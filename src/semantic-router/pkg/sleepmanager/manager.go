// Package sleepmanager provides functionality for managing vLLM endpoint sleep/wake states.
// It enables automatic waking of sleeping endpoints when requests are routed to them,
// and optional automatic sleeping of endpoints after a period of inactivity.
package sleepmanager

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// EndpointState represents the current state of an endpoint
type EndpointState string

const (
	// StateUnknown indicates the endpoint state is not yet known
	StateUnknown EndpointState = "unknown"
	// StateAwake indicates the endpoint is ready to serve requests
	StateAwake EndpointState = "awake"
	// StateSleeping indicates the endpoint is in sleep mode
	StateSleeping EndpointState = "sleeping"
	// StateWakingUp indicates the endpoint is currently waking up
	StateWakingUp EndpointState = "waking_up"
	// StateGoingToSleep indicates the endpoint is transitioning to sleep
	StateGoingToSleep EndpointState = "going_to_sleep"
)

// EndpointInfo holds information about an endpoint's sleep state
type EndpointInfo struct {
	Name            string
	Address         string
	State           EndpointState
	LastActivity    time.Time
	SleepConfig     *config.EndpointSleepConfig
	WakeUpStartTime time.Time
	mu              sync.RWMutex
}

// Manager manages sleep/wake states for vLLM endpoints
type Manager struct {
	endpoints      map[string]*EndpointInfo // key: endpoint address
	mu             sync.RWMutex
	httpClient     *http.Client
	inactivityChan chan string
	stopChan       chan struct{}
	wg             sync.WaitGroup
}

// singleton instance
var (
	instance *Manager
	once     sync.Once
)

// GetManager returns the singleton Manager instance
// Must call InitManager first to configure the manager
func GetManager() *Manager {
	return instance
}

// InitManager initializes the sleep manager with endpoint configurations
func InitManager(endpoints []config.VLLMEndpoint) *Manager {
	once.Do(func() {
		instance = &Manager{
			endpoints: make(map[string]*EndpointInfo),
			httpClient: &http.Client{
				Timeout: 10 * time.Second,
			},
			inactivityChan: make(chan string, 100),
			stopChan:       make(chan struct{}),
		}
	})

	// Update endpoints
	instance.mu.Lock()
	for _, ep := range endpoints {
		if ep.SleepMode != nil && ep.SleepMode.Enabled {
			addr := fmt.Sprintf("%s:%d", ep.Address, ep.Port)
			instance.endpoints[addr] = &EndpointInfo{
				Name:         ep.Name,
				Address:      addr,
				State:        StateUnknown,
				LastActivity: time.Now(),
				SleepConfig:  ep.SleepMode,
			}
			logging.Infof("Sleep manager: Registered endpoint %s (%s) with sleep mode enabled", ep.Name, addr)
		}
	}
	instance.mu.Unlock()

	// Start background workers
	instance.wg.Add(1)
	go instance.inactivityMonitor()

	return instance
}

// Stop gracefully stops the manager
func (m *Manager) Stop() {
	close(m.stopChan)
	m.wg.Wait()
}

// IsSleepModeEnabled checks if an endpoint has sleep mode enabled
func (m *Manager) IsSleepModeEnabled(address string) bool {
	m.mu.RLock()
	defer m.mu.RUnlock()

	info, exists := m.endpoints[address]
	return exists && info.SleepConfig != nil && info.SleepConfig.Enabled
}

// GetEndpointState returns the current state of an endpoint
func (m *Manager) GetEndpointState(address string) EndpointState {
	m.mu.RLock()
	defer m.mu.RUnlock()

	info, exists := m.endpoints[address]
	if !exists {
		return StateUnknown
	}

	info.mu.RLock()
	defer info.mu.RUnlock()
	return info.State
}

// RecordActivity records activity on an endpoint (resets inactivity timer)
func (m *Manager) RecordActivity(address string) {
	m.mu.RLock()
	info, exists := m.endpoints[address]
	m.mu.RUnlock()

	if !exists {
		return
	}

	info.mu.Lock()
	info.LastActivity = time.Now()
	info.mu.Unlock()
}

// EnsureAwake ensures the endpoint is awake before routing requests to it
// This is the main entry point for the request routing flow
// Returns nil if the endpoint is ready, error if wake-up failed
func (m *Manager) EnsureAwake(ctx context.Context, address string) error {
	if !m.IsSleepModeEnabled(address) {
		return nil // Sleep mode not enabled for this endpoint
	}

	m.mu.RLock()
	info := m.endpoints[address]
	m.mu.RUnlock()

	if info == nil {
		return nil
	}

	info.mu.Lock()
	currentState := info.State
	info.mu.Unlock()

	switch currentState {
	case StateAwake:
		// Endpoint is awake, record activity and proceed
		m.RecordActivity(address)
		return nil

	case StateWakingUp:
		// Already waking up, wait for it
		return m.waitForWakeUp(ctx, info)

	case StateSleeping, StateUnknown:
		// Need to wake up the endpoint
		return m.wakeUp(ctx, info)

	case StateGoingToSleep:
		// Cancel the sleep and wake up
		return m.wakeUp(ctx, info)

	default:
		return nil
	}
}

// wakeUp sends a wake-up request to the endpoint
func (m *Manager) wakeUp(ctx context.Context, info *EndpointInfo) error {
	info.mu.Lock()
	if info.State == StateAwake {
		info.mu.Unlock()
		return nil
	}
	info.State = StateWakingUp
	info.WakeUpStartTime = time.Now()
	info.mu.Unlock()

	startTime := time.Now()
	logging.Infof("Sleep manager: Waking up endpoint %s (%s)", info.Name, info.Address)
	metrics.RecordEndpointWakeUpAttempt(info.Name, info.Address)

	// Send wake-up request
	wakeUpURL := fmt.Sprintf("http://%s/wake_up", info.Address)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, wakeUpURL, nil)
	if err != nil {
		m.setEndpointState(info, StateUnknown)
		return fmt.Errorf("failed to create wake-up request: %w", err)
	}

	resp, err := m.httpClient.Do(req)
	if err != nil {
		m.setEndpointState(info, StateUnknown)
		metrics.RecordEndpointWakeUpFailure(info.Name, info.Address)
		return fmt.Errorf("failed to send wake-up request to %s: %w", info.Address, err)
	}
	defer resp.Body.Close()

	// Read response body for logging
	body, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusOK {
		m.setEndpointState(info, StateUnknown)
		metrics.RecordEndpointWakeUpFailure(info.Name, info.Address)
		return fmt.Errorf("wake-up request failed with status %d: %s", resp.StatusCode, string(body))
	}

	// Wait for the endpoint to be fully awake
	if err := m.waitForWakeUp(ctx, info); err != nil {
		return err
	}

	duration := time.Since(startTime)
	logging.Infof("Sleep manager: Endpoint %s (%s) woke up successfully in %v", info.Name, info.Address, duration)
	metrics.RecordEndpointWakeUpSuccess(info.Name, info.Address, duration)

	return nil
}

// waitForWakeUp waits for an endpoint to become awake
func (m *Manager) waitForWakeUp(ctx context.Context, info *EndpointInfo) error {
	timeout := 60 * time.Second
	retryInterval := 500 * time.Millisecond

	if info.SleepConfig != nil {
		if info.SleepConfig.WakeUpTimeoutSeconds > 0 {
			timeout = time.Duration(info.SleepConfig.WakeUpTimeoutSeconds) * time.Second
		}
		if info.SleepConfig.WakeUpRetryIntervalMs > 0 {
			retryInterval = time.Duration(info.SleepConfig.WakeUpRetryIntervalMs) * time.Millisecond
		}
	}

	deadline := time.Now().Add(timeout)
	ticker := time.NewTicker(retryInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			if time.Now().After(deadline) {
				m.setEndpointState(info, StateUnknown)
				return fmt.Errorf("timeout waiting for endpoint %s to wake up", info.Address)
			}

			// Check if endpoint is ready by calling health endpoint
			if m.isEndpointHealthy(info.Address) {
				m.setEndpointState(info, StateAwake)
				m.RecordActivity(info.Address)
				return nil
			}
		}
	}
}

// isEndpointHealthy checks if an endpoint is healthy and ready
func (m *Manager) isEndpointHealthy(address string) bool {
	healthURL := fmt.Sprintf("http://%s/health", address)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, healthURL, nil)
	if err != nil {
		return false
	}

	resp, err := m.httpClient.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()

	return resp.StatusCode == http.StatusOK
}

// Sleep puts an endpoint to sleep
func (m *Manager) Sleep(ctx context.Context, address string) error {
	if !m.IsSleepModeEnabled(address) {
		return fmt.Errorf("sleep mode not enabled for endpoint %s", address)
	}

	m.mu.RLock()
	info := m.endpoints[address]
	m.mu.RUnlock()

	if info == nil {
		return fmt.Errorf("endpoint %s not found", address)
	}

	info.mu.Lock()
	if info.State == StateSleeping {
		info.mu.Unlock()
		return nil // Already sleeping
	}
	info.State = StateGoingToSleep
	info.mu.Unlock()

	level := 1
	if info.SleepConfig != nil && info.SleepConfig.SleepLevel > 0 {
		level = info.SleepConfig.SleepLevel
	}

	logging.Infof("Sleep manager: Putting endpoint %s (%s) to sleep (level %d)", info.Name, info.Address, level)

	// Send sleep request
	sleepURL := fmt.Sprintf("http://%s/sleep?level=%d", info.Address, level)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, sleepURL, nil)
	if err != nil {
		m.setEndpointState(info, StateUnknown)
		return fmt.Errorf("failed to create sleep request: %w", err)
	}

	resp, err := m.httpClient.Do(req)
	if err != nil {
		m.setEndpointState(info, StateUnknown)
		return fmt.Errorf("failed to send sleep request to %s: %w", info.Address, err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusOK {
		m.setEndpointState(info, StateUnknown)
		return fmt.Errorf("sleep request failed with status %d: %s", resp.StatusCode, string(body))
	}

	m.setEndpointState(info, StateSleeping)
	logging.Infof("Sleep manager: Endpoint %s (%s) is now sleeping", info.Name, info.Address)
	metrics.RecordEndpointSleep(info.Name, info.Address)

	return nil
}

// setEndpointState safely sets the endpoint state
func (m *Manager) setEndpointState(info *EndpointInfo, state EndpointState) {
	info.mu.Lock()
	info.State = state
	info.mu.Unlock()
}

// inactivityMonitor monitors endpoints for inactivity and puts them to sleep
func (m *Manager) inactivityMonitor() {
	defer m.wg.Done()

	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-m.stopChan:
			return
		case <-ticker.C:
			m.checkInactivity()
		}
	}
}

// checkInactivity checks all endpoints for inactivity
func (m *Manager) checkInactivity() {
	m.mu.RLock()
	endpoints := make([]*EndpointInfo, 0, len(m.endpoints))
	for _, info := range m.endpoints {
		endpoints = append(endpoints, info)
	}
	m.mu.RUnlock()

	for _, info := range endpoints {
		if info.SleepConfig == nil || info.SleepConfig.InactivityTimeoutSeconds <= 0 {
			continue // No automatic sleep configured
		}

		info.mu.RLock()
		state := info.State
		lastActivity := info.LastActivity
		info.mu.RUnlock()

		if state != StateAwake {
			continue // Only sleep awake endpoints
		}

		inactivityDuration := time.Since(lastActivity)
		timeout := time.Duration(info.SleepConfig.InactivityTimeoutSeconds) * time.Second

		if inactivityDuration >= timeout {
			logging.Infof("Sleep manager: Endpoint %s (%s) inactive for %v, putting to sleep",
				info.Name, info.Address, inactivityDuration)
			go func(addr string) {
				ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
				defer cancel()
				if err := m.Sleep(ctx, addr); err != nil {
					logging.Warnf("Failed to put endpoint %s to sleep: %v", addr, err)
				}
			}(info.Address)
		}
	}
}

// GetAllEndpointStates returns the state of all managed endpoints
func (m *Manager) GetAllEndpointStates() map[string]EndpointState {
	m.mu.RLock()
	defer m.mu.RUnlock()

	states := make(map[string]EndpointState, len(m.endpoints))
	for addr, info := range m.endpoints {
		info.mu.RLock()
		states[addr] = info.State
		info.mu.RUnlock()
	}
	return states
}

// GetEndpointInfo returns detailed information about an endpoint
func (m *Manager) GetEndpointInfo(address string) *EndpointInfo {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.endpoints[address]
}

