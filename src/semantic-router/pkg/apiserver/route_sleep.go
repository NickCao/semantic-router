//go:build !windows && cgo

package apiserver

import (
	"context"
	"encoding/json"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sleepmanager"
)

// EndpointStateResponse represents the response for endpoint state queries
type EndpointStateResponse struct {
	Address      string `json:"address"`
	Name         string `json:"name,omitempty"`
	State        string `json:"state"`
	LastActivity string `json:"last_activity,omitempty"`
}

// EndpointSleepRequest represents a request to put an endpoint to sleep
type EndpointSleepRequest struct {
	Address string `json:"address"`
}

// EndpointWakeRequest represents a request to wake up an endpoint
type EndpointWakeRequest struct {
	Address string `json:"address"`
}

// AllEndpointsStateResponse represents the response for all endpoints state
type AllEndpointsStateResponse struct {
	Endpoints []EndpointStateResponse `json:"endpoints"`
}

// SleepActionResponse represents the response for sleep/wake actions
type SleepActionResponse struct {
	Success bool   `json:"success"`
	Message string `json:"message"`
	Address string `json:"address"`
	State   string `json:"state"`
}

// handleGetEndpointStates returns the current state of all managed endpoints
func (s *ClassificationAPIServer) handleGetEndpointStates(w http.ResponseWriter, r *http.Request) {
	manager := sleepmanager.GetManager()
	if manager == nil {
		s.writeJSONResponse(w, http.StatusOK, AllEndpointsStateResponse{
			Endpoints: []EndpointStateResponse{},
		})
		return
	}

	states := manager.GetAllEndpointStates()
	response := AllEndpointsStateResponse{
		Endpoints: make([]EndpointStateResponse, 0, len(states)),
	}

	for addr, state := range states {
		info := manager.GetEndpointInfo(addr)
		epResponse := EndpointStateResponse{
			Address: addr,
			State:   string(state),
		}
		if info != nil {
			epResponse.Name = info.Name
			epResponse.LastActivity = info.LastActivity.Format(time.RFC3339)
		}
		response.Endpoints = append(response.Endpoints, epResponse)
	}

	s.writeJSONResponse(w, http.StatusOK, response)
}

// handleGetEndpointState returns the current state of a specific endpoint
func (s *ClassificationAPIServer) handleGetEndpointState(w http.ResponseWriter, r *http.Request) {
	address := r.URL.Query().Get("address")
	if address == "" {
		s.writeErrorResponse(w, http.StatusBadRequest, "MISSING_ADDRESS", "Address query parameter is required")
		return
	}

	manager := sleepmanager.GetManager()
	if manager == nil {
		s.writeErrorResponse(w, http.StatusNotFound, "SLEEP_MANAGER_NOT_INITIALIZED", "Sleep manager is not initialized")
		return
	}

	if !manager.IsSleepModeEnabled(address) {
		s.writeErrorResponse(w, http.StatusNotFound, "ENDPOINT_NOT_FOUND", "Endpoint not found or sleep mode not enabled")
		return
	}

	state := manager.GetEndpointState(address)
	info := manager.GetEndpointInfo(address)

	response := EndpointStateResponse{
		Address: address,
		State:   string(state),
	}
	if info != nil {
		response.Name = info.Name
		response.LastActivity = info.LastActivity.Format(time.RFC3339)
	}

	s.writeJSONResponse(w, http.StatusOK, response)
}

// handleSleepEndpoint puts an endpoint to sleep
func (s *ClassificationAPIServer) handleSleepEndpoint(w http.ResponseWriter, r *http.Request) {
	var req EndpointSleepRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_REQUEST", "Invalid JSON request body")
		return
	}
	defer r.Body.Close()

	if req.Address == "" {
		s.writeErrorResponse(w, http.StatusBadRequest, "MISSING_ADDRESS", "Address is required")
		return
	}

	manager := sleepmanager.GetManager()
	if manager == nil {
		s.writeErrorResponse(w, http.StatusServiceUnavailable, "SLEEP_MANAGER_NOT_INITIALIZED", "Sleep manager is not initialized")
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()

	logging.Infof("API: Putting endpoint %s to sleep", req.Address)

	if err := manager.Sleep(ctx, req.Address); err != nil {
		logging.Errorf("API: Failed to put endpoint %s to sleep: %v", req.Address, err)
		s.writeErrorResponse(w, http.StatusInternalServerError, "SLEEP_FAILED", err.Error())
		return
	}

	state := manager.GetEndpointState(req.Address)
	response := SleepActionResponse{
		Success: true,
		Message: "Endpoint successfully put to sleep",
		Address: req.Address,
		State:   string(state),
	}

	s.writeJSONResponse(w, http.StatusOK, response)
}

// handleWakeEndpoint wakes up an endpoint
func (s *ClassificationAPIServer) handleWakeEndpoint(w http.ResponseWriter, r *http.Request) {
	var req EndpointWakeRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_REQUEST", "Invalid JSON request body")
		return
	}
	defer r.Body.Close()

	if req.Address == "" {
		s.writeErrorResponse(w, http.StatusBadRequest, "MISSING_ADDRESS", "Address is required")
		return
	}

	manager := sleepmanager.GetManager()
	if manager == nil {
		s.writeErrorResponse(w, http.StatusServiceUnavailable, "SLEEP_MANAGER_NOT_INITIALIZED", "Sleep manager is not initialized")
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 120*time.Second)
	defer cancel()

	logging.Infof("API: Waking up endpoint %s", req.Address)

	if err := manager.EnsureAwake(ctx, req.Address); err != nil {
		logging.Errorf("API: Failed to wake up endpoint %s: %v", req.Address, err)
		s.writeErrorResponse(w, http.StatusInternalServerError, "WAKE_FAILED", err.Error())
		return
	}

	state := manager.GetEndpointState(req.Address)
	response := SleepActionResponse{
		Success: true,
		Message: "Endpoint successfully woken up",
		Address: req.Address,
		State:   string(state),
	}

	s.writeJSONResponse(w, http.StatusOK, response)
}

