package sleepmanager

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestManagerInit(t *testing.T) {
	// Reset singleton for testing
	once = sync.Once{}
	instance = nil

	endpoints := []config.VLLMEndpoint{
		{
			Name:    "test-endpoint",
			Address: "127.0.0.1",
			Port:    8000,
			SleepMode: &config.EndpointSleepConfig{
				Enabled:              true,
				SleepLevel:           1,
				WakeUpTimeoutSeconds: 5,
			},
		},
		{
			Name:    "no-sleep-endpoint",
			Address: "127.0.0.1",
			Port:    8001,
			// No SleepMode configured
		},
	}

	manager := InitManager(endpoints)

	if manager == nil {
		t.Fatal("Expected manager to be initialized")
	}

	// Check that only the endpoint with sleep mode is registered
	if !manager.IsSleepModeEnabled("127.0.0.1:8000") {
		t.Error("Expected sleep mode to be enabled for 127.0.0.1:8000")
	}

	if manager.IsSleepModeEnabled("127.0.0.1:8001") {
		t.Error("Expected sleep mode NOT to be enabled for 127.0.0.1:8001")
	}

	manager.Stop()
}

func TestEndpointState(t *testing.T) {
	// Reset singleton for testing
	once = sync.Once{}
	instance = nil

	endpoints := []config.VLLMEndpoint{
		{
			Name:    "test-endpoint",
			Address: "127.0.0.1",
			Port:    8000,
			SleepMode: &config.EndpointSleepConfig{
				Enabled:              true,
				SleepLevel:           1,
				WakeUpTimeoutSeconds: 5,
			},
		},
	}

	manager := InitManager(endpoints)
	defer manager.Stop()

	// Initial state should be unknown
	state := manager.GetEndpointState("127.0.0.1:8000")
	if state != StateUnknown {
		t.Errorf("Expected initial state to be %s, got %s", StateUnknown, state)
	}

	// Record activity and check state
	manager.RecordActivity("127.0.0.1:8000")

	info := manager.GetEndpointInfo("127.0.0.1:8000")
	if info == nil {
		t.Fatal("Expected endpoint info to exist")
	}

	if info.Name != "test-endpoint" {
		t.Errorf("Expected endpoint name to be 'test-endpoint', got '%s'", info.Name)
	}
}

func TestWakeUpWithMockServer(t *testing.T) {
	// Reset singleton for testing
	once = sync.Once{}
	instance = nil

	// Create a mock vLLM server
	wakeUpCalled := false
	healthyCalled := 0

	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/wake_up":
			wakeUpCalled = true
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(`{"status": "ok"}`))
		case "/health":
			healthyCalled++
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(`{"status": "healthy"}`))
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer mockServer.Close()

	// Parse server address
	addr := mockServer.Listener.Addr().String()

	// Create endpoint with mock server address
	endpoints := []config.VLLMEndpoint{
		{
			Name:    "mock-endpoint",
			Address: "localhost", // Will be overwritten
			Port:    8000,
			SleepMode: &config.EndpointSleepConfig{
				Enabled:                true,
				SleepLevel:             1,
				WakeUpTimeoutSeconds:   5,
				WakeUpRetryIntervalMs:  100,
			},
		},
	}

	manager := InitManager(endpoints)
	defer manager.Stop()

	// Manually add the mock endpoint with correct address
	manager.mu.Lock()
	manager.endpoints[addr] = &EndpointInfo{
		Name:         "mock-endpoint",
		Address:      addr,
		State:        StateSleeping,
		LastActivity: time.Now(),
		SleepConfig: &config.EndpointSleepConfig{
			Enabled:               true,
			SleepLevel:            1,
			WakeUpTimeoutSeconds:  5,
			WakeUpRetryIntervalMs: 100,
		},
	}
	manager.mu.Unlock()

	// Try to ensure the endpoint is awake
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	err := manager.EnsureAwake(ctx, addr)
	if err != nil {
		t.Fatalf("EnsureAwake failed: %v", err)
	}

	if !wakeUpCalled {
		t.Error("Expected wake_up endpoint to be called")
	}

	if healthyCalled < 1 {
		t.Error("Expected health endpoint to be called at least once")
	}

	// Check state is now awake
	state := manager.GetEndpointState(addr)
	if state != StateAwake {
		t.Errorf("Expected state to be %s, got %s", StateAwake, state)
	}
}

func TestSleepWithMockServer(t *testing.T) {
	// Reset singleton for testing
	once = sync.Once{}
	instance = nil

	// Create a mock vLLM server
	sleepCalled := false
	sleepLevel := 0

	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/sleep":
			sleepCalled = true
			q := r.URL.Query()
			if l := q.Get("level"); l != "" {
				if l == "1" {
					sleepLevel = 1
				} else if l == "2" {
					sleepLevel = 2
				}
			}
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(`{"status": "sleeping"}`))
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer mockServer.Close()

	addr := mockServer.Listener.Addr().String()

	endpoints := []config.VLLMEndpoint{
		{
			Name:    "mock-endpoint",
			Address: "localhost",
			Port:    8000,
			SleepMode: &config.EndpointSleepConfig{
				Enabled:    true,
				SleepLevel: 2,
			},
		},
	}

	manager := InitManager(endpoints)
	defer manager.Stop()

	// Manually add the mock endpoint
	manager.mu.Lock()
	manager.endpoints[addr] = &EndpointInfo{
		Name:         "mock-endpoint",
		Address:      addr,
		State:        StateAwake,
		LastActivity: time.Now(),
		SleepConfig: &config.EndpointSleepConfig{
			Enabled:    true,
			SleepLevel: 2,
		},
	}
	manager.mu.Unlock()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	err := manager.Sleep(ctx, addr)
	if err != nil {
		t.Fatalf("Sleep failed: %v", err)
	}

	if !sleepCalled {
		t.Error("Expected sleep endpoint to be called")
	}

	if sleepLevel != 2 {
		t.Errorf("Expected sleep level 2, got %d", sleepLevel)
	}

	state := manager.GetEndpointState(addr)
	if state != StateSleeping {
		t.Errorf("Expected state to be %s, got %s", StateSleeping, state)
	}
}

func TestGetAllEndpointStates(t *testing.T) {
	// Reset singleton for testing
	once = sync.Once{}
	instance = nil

	endpoints := []config.VLLMEndpoint{
		{
			Name:    "endpoint1",
			Address: "127.0.0.1",
			Port:    8000,
			SleepMode: &config.EndpointSleepConfig{
				Enabled: true,
			},
		},
		{
			Name:    "endpoint2",
			Address: "127.0.0.1",
			Port:    8001,
			SleepMode: &config.EndpointSleepConfig{
				Enabled: true,
			},
		},
	}

	manager := InitManager(endpoints)
	defer manager.Stop()

	states := manager.GetAllEndpointStates()

	if len(states) != 2 {
		t.Errorf("Expected 2 endpoints, got %d", len(states))
	}

	for _, state := range states {
		if state != StateUnknown {
			t.Errorf("Expected all initial states to be %s, got %s", StateUnknown, state)
		}
	}
}

