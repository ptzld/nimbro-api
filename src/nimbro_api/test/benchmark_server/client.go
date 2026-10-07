package main

import (
	"bytes"
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"sync"
	"sync/atomic"
	"time"
)

type clientConfig struct {
	Endpoint          string  `json:"endpoint"`
	CAFile            string  `json:"ca_file"`
	HTTP              string  `json:"http"`
	Processes         int     `json:"processes"`
	Threads           int     `json:"threads_per_process"`
	PromptBytes       int     `json:"prompt_bytes"`
	ResponseBodyBytes int64   `json:"response_body_bytes"`
	RequestTimeout    float64 `json:"request_timeout"`
}

type command struct {
	Kind     string  `json:"kind"`
	Duration float64 `json:"duration"`
	Probe    bool    `json:"probe"`
}

type clientStats struct {
	submitted, completed, succeeded, failed atomic.Int64
	latencySum, latencyMax                  atomic.Int64
	histogram                               [2048]atomic.Int64
	errorLock                               sync.Mutex
	errors                                  []string
}

func (s *clientStats) finish(duration time.Duration, err error) {
	n := duration.Nanoseconds()
	s.latencySum.Add(n)
	peak(&s.latencyMax, n)
	bucket := min(2047, max(0, int(math.Log(max(float64(n)/1000, 1))/math.Log(1.02))))
	s.histogram[bucket].Add(1)
	if err == nil {
		s.succeeded.Add(1)
	} else {
		s.failed.Add(1)
		s.errorLock.Lock()
		if len(s.errors) < 3 {
			message := err.Error()
			if len(message) > 1000 {
				message = message[:1000]
			}
			s.errors = append(s.errors, message)
		}
		s.errorLock.Unlock()
	}
	s.completed.Add(1)
}

func collect(groups []*clientStats, histogram bool) map[string]any {
	var submitted, completed, succeeded, failed, sum, maximum int64
	buckets := map[int]int64{}
	errors := []string{}
	for _, s := range groups {
		completed += s.completed.Load()
		submitted += s.submitted.Load()
		succeeded += s.succeeded.Load()
		failed += s.failed.Load()
		sum += s.latencySum.Load()
		maximum = max(maximum, s.latencyMax.Load())
		if histogram {
			for bucket := range s.histogram {
				if count := s.histogram[bucket].Load(); count > 0 {
					buckets[bucket] += count
				}
			}
		}
		s.errorLock.Lock()
		for _, message := range s.errors {
			if len(errors) < 5 {
				errors = append(errors, message)
			}
		}
		s.errorLock.Unlock()
	}
	failures := map[string]int64{}
	if failed > 0 {
		failures["HTTPFailure"] = failed
	}
	return map[string]any{
		"submitted": submitted, "completed": completed, "succeeded": succeeded, "failed": failed,
		"outstanding": submitted - completed, "latency_sum_seconds": float64(sum) / 1e9,
		"max_latency_seconds": float64(maximum) / 1e9,
		"histogram": buckets, "errors": failures, "error_samples": errors,
	}
}

func post(ctx context.Context, client *http.Client, config clientConfig, body []byte) error {
	r, err := http.NewRequestWithContext(ctx, "POST", config.Endpoint, bytes.NewReader(body))
	if err != nil {
		return err
	}
	r.Header.Set("Content-Type", "application/json")
	response, err := client.Do(r)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	n, err := io.Copy(io.Discard, response.Body)
	if err != nil {
		return err
	}
	if response.StatusCode != 200 {
		return fmt.Errorf("unexpected HTTP status %d", response.StatusCode)
	}
	if n != config.ResponseBodyBytes {
		return fmt.Errorf("expected %d response bytes, got %d", config.ResponseBodyBytes, n)
	}
	if (config.HTTP == "h2" && response.ProtoMajor != 2) || (config.HTTP == "h1" && response.ProtoMajor != 1) {
		return fmt.Errorf("requested %s, negotiated %s", config.HTTP, response.Proto)
	}
	return nil
}

func nativePhase(ctx context.Context, config clientConfig, phase command, commands <-chan command,
	clients []*http.Client, body []byte, output *json.Encoder) error {
	groups := make([]*clientStats, len(clients))
	gate := make(chan struct{})
	var ready, finished sync.WaitGroup
	var deadline time.Time
	for i, client := range clients {
		s := new(clientStats)
		groups[i] = s
		for range config.Threads {
			ready.Add(1)
			finished.Add(1)
			go func() {
				defer finished.Done()
				ready.Done()
				select {
				case <-ctx.Done():
					return
				case <-gate:
				}
				for attempted := 0; ctx.Err() == nil; attempted++ {
					if (phase.Probe && attempted > 0) || (!phase.Probe && !time.Now().Before(deadline)) {
						return
					}
					s.submitted.Add(1)
					start := time.Now()
					err := post(ctx, client, config, body)
					s.finish(time.Since(start), err)
				}
			}()
		}
	}
	ready.Wait()
	if err := output.Encode(map[string]any{"kind": "prepared", "worker": 0}); err != nil {
		return err
	}
	select {
	case <-ctx.Done():
		return ctx.Err()
	case cmd, ok := <-commands:
		if !ok || cmd.Kind != "go" {
			return fmt.Errorf("expected go command")
		}
	}
	deadline = time.Now().Add(time.Duration(phase.Duration * float64(time.Second)))
	cpuStart := runtimeCPU()
	close(gate)
	done := make(chan struct{})
	go func() { finished.Wait(); close(done) }()
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-done:
			return output.Encode(map[string]any{
				"kind": "done", "worker": 0, "statistics": collect(groups, true),
				"cpu_seconds": max(0, runtimeCPU()-cpuStart),
			})
		case <-ticker.C:
			if err := output.Encode(map[string]any{"kind": "progress", "worker": 0, "statistics": collect(groups, false)}); err != nil {
				return err
			}
		}
	}
}

func runClient() error {
	input := json.NewDecoder(os.Stdin)
	output := json.NewEncoder(os.Stdout)
	var config clientConfig
	if err := input.Decode(&config); err != nil {
		return err
	}
	if config.Processes < 1 || config.Threads < 1 || config.PromptBytes < 1 || config.RequestTimeout <= 0 {
		return fmt.Errorf("invalid native client configuration")
	}
	certificate, err := os.ReadFile(config.CAFile)
	if err != nil {
		return err
	}
	roots := x509.NewCertPool()
	if !roots.AppendCertsFromPEM(certificate) {
		return fmt.Errorf("invalid benchmark certificate")
	}
	protocols := new(http.Protocols)
	protocols.SetHTTP1(config.HTTP == "h1")
	protocols.SetHTTP2(config.HTTP == "h2")
	clients := make([]*http.Client, config.Processes)
	for i := range clients {
		transport := http.DefaultTransport.(*http.Transport).Clone()
		transport.Proxy = nil
		transport.TLSClientConfig = &tls.Config{RootCAs: roots, MinVersion: tls.VersionTLS12}
		transport.Protocols = protocols
		transport.HTTP2 = &http.HTTP2Config{StrictMaxConcurrentRequests: false}
		transport.MaxIdleConns = config.Threads
		transport.MaxIdleConnsPerHost = config.Threads
		transport.DisableCompression = true
		defer transport.CloseIdleConnections()
		clients[i] = &http.Client{Transport: transport, Timeout: time.Duration(config.RequestTimeout * float64(time.Second)),
			CheckRedirect: func(_ *http.Request, _ []*http.Request) error { return http.ErrUseLastResponse }}
	}
	body, err := json.Marshal(map[string]any{
		"model": "fake-model", "messages": []any{map[string]any{
			"role": "user", "content": []any{map[string]string{"type": "text", "text": string(bytes.Repeat([]byte("x"), config.PromptBytes))}},
		}},
		"temperature": json.Number("1.0"), "top_p": json.Number("1.0"), "max_completion_tokens": 10000,
		"presence_penalty": json.Number("0.0"), "frequency_penalty": json.Number("0.0"), "response_format": map[string]string{"type": "text"},
		"n": 1, "stream": false, "reasoning_effort": "none",
	})
	if err != nil {
		return err
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	commands := make(chan command, 1)
	go func() {
		defer close(commands)
		for {
			var cmd command
			if input.Decode(&cmd) != nil || cmd.Kind == "shutdown" {
				cancel()
				return
			}
			select {
			case commands <- cmd:
			case <-ctx.Done():
				return
			}
		}
	}()
	if err := output.Encode(map[string]any{"kind": "ready"}); err != nil {
		return err
	}
	for {
		select {
		case <-ctx.Done():
			return nil
		case phase, ok := <-commands:
			if !ok {
				return nil
			}
			if phase.Kind != "phase" {
				return fmt.Errorf("expected phase command")
			}
			if err := nativePhase(ctx, config, phase, commands, clients, body, output); err != nil {
				if ctx.Err() != nil {
					return nil
				}
				return err
			}
		}
	}
}
