package main

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/json"
	"encoding/pem"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"math/big"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"runtime/metrics"
	"strings"
	"sync/atomic"
	"time"
)

type counters struct {
	entered, received, active, peak, holding     atomic.Int64
	completed, canceled, readErrors, writeErrors atomic.Int64
	requestBytes, responseBytes, h1, h2          atomic.Int64
	connectionBaseline                           int64
}

type benchmark struct {
	current                                       atomic.Pointer[counters]
	connections, openConnections, peakConnections atomic.Int64
	body                                          []byte
	hold                                          time.Duration
	config                                        map[string]any
}

func peak(target *atomic.Int64, value int64) {
	for old := target.Load(); value > old; old = target.Load() {
		if target.CompareAndSwap(old, value) {
			return
		}
	}
}

func runtimeCPU() float64 {
	samples := []metrics.Sample{
		{Name: "/cpu/classes/user:cpu-seconds"},
		{Name: "/cpu/classes/gc/total:cpu-seconds"},
		{Name: "/cpu/classes/scavenge/total:cpu-seconds"},
	}
	metrics.Read(samples)
	total := 0.0
	for _, sample := range samples {
		if sample.Value.Kind() == metrics.KindFloat64 {
			total += sample.Value.Float64()
		}
	}
	return total
}

func (b *benchmark) snapshot() map[string]any {
	c := b.current.Load()
	return map[string]any{
		"entered": c.entered.Load(), "received": c.received.Load(),
		"active": c.active.Load(), "peak_active": c.peak.Load(), "holding": c.holding.Load(),
		"completed": c.completed.Load(), "canceled": c.canceled.Load(),
		"read_errors": c.readErrors.Load(), "write_errors": c.writeErrors.Load(),
		"request_bytes": c.requestBytes.Load(), "response_bytes": c.responseBytes.Load(),
		"http1_requests": c.h1.Load(), "http2_requests": c.h2.Load(),
		"connections_created": b.connections.Load() - c.connectionBaseline,
		"open_connections":    b.openConnections.Load(), "peak_open_connections": b.peakConnections.Load(),
		"runtime_cpu_seconds_estimate": runtimeCPU(), "goroutines": runtime.NumGoroutine(),
		"config": b.config,
	}
}

func (b *benchmark) completion(w http.ResponseWriter, r *http.Request) {
	c := b.current.Load()
	c.entered.Add(1)
	peak(&c.peak, c.active.Add(1))
	defer c.active.Add(-1)
	if r.ProtoMajor == 2 {
		c.h2.Add(1)
	} else {
		c.h1.Add(1)
	}
	n, err := io.Copy(io.Discard, r.Body)
	c.requestBytes.Add(n)
	if err != nil {
		c.readErrors.Add(1)
		return
	}
	c.received.Add(1)
	if b.hold > 0 {
		c.holding.Add(1)
		timer := time.NewTimer(b.hold)
		select {
		case <-timer.C:
		case <-r.Context().Done():
			timer.Stop()
			c.holding.Add(-1)
			c.canceled.Add(1)
			return
		}
		c.holding.Add(-1)
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Length", fmt.Sprint(len(b.body)))
	written, err := w.Write(b.body)
	c.responseBytes.Add(int64(written))
	if err == nil {
		err = http.NewResponseController(w).Flush()
	}
	if err != nil {
		c.writeErrors.Add(1)
	} else {
		c.completed.Add(1)
	}
}

func writeJSON(w http.ResponseWriter, value any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(value)
}

func certificate(directory string) (string, string, error) {
	if err := os.MkdirAll(directory, 0700); err != nil {
		return "", "", err
	}
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return "", "", err
	}
	serial, err := rand.Int(rand.Reader, new(big.Int).Lsh(big.NewInt(1), 128))
	if err != nil {
		return "", "", err
	}
	template := &x509.Certificate{
		SerialNumber: serial, Subject: pkix.Name{CommonName: "NimbRo bandwidth benchmark"},
		NotBefore: time.Now().Add(-time.Hour), NotAfter: time.Now().Add(7 * 24 * time.Hour),
		KeyUsage:              x509.KeyUsageDigitalSignature | x509.KeyUsageCertSign,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true, IsCA: true,
	}
	for _, host := range []string{"localhost", "127.0.0.1", "::1"} {
		if ip := net.ParseIP(host); ip != nil {
			template.IPAddresses = append(template.IPAddresses, ip)
		} else {
			template.DNSNames = append(template.DNSNames, host)
		}
	}
	der, err := x509.CreateCertificate(rand.Reader, template, template, &key.PublicKey, key)
	if err != nil {
		return "", "", err
	}
	keyDER, err := x509.MarshalPKCS8PrivateKey(key)
	if err != nil {
		return "", "", err
	}
	certPath, _ := filepath.Abs(filepath.Join(directory, "cert.pem"))
	keyPath, _ := filepath.Abs(filepath.Join(directory, "key.pem"))
	if err = os.WriteFile(certPath, pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}), 0600); err != nil {
		return "", "", err
	}
	err = os.WriteFile(keyPath, pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: keyDER}), 0600)
	return certPath, keyPath, err
}

func run() error {
	certDir := flag.String("cert-dir", "", "Generated certificate directory")
	responseBytes := flag.Int("response-bytes", 2048, "Completion text bytes")
	holdMS := flag.Float64("hold-ms", 0, "Delay after body receipt in milliseconds")
	streams := flag.Int("h2-max-streams", 250, "Maximum concurrent HTTP/2 streams per connection")
	flag.Parse()
	if *certDir == "" {
		return fmt.Errorf("cert-dir is required")
	}
	if *responseBytes < 1 || *streams < 1 || uint64(*streams) > 1<<32-1 {
		return fmt.Errorf("response-bytes must be positive; h2-max-streams must be in [1, 4294967295]")
	}
	if *holdMS < 0 || math.IsNaN(*holdMS) || math.IsInf(*holdMS, 0) || *holdMS >= float64(1<<63-1)/float64(time.Millisecond) {
		return fmt.Errorf("hold-ms must be a finite nonnegative duration representable in nanoseconds")
	}
	body, err := json.Marshal(map[string]any{
		"id": "fake-completion", "object": "chat.completion", "model": "fake-model",
		"choices": []any{map[string]any{
			"index": 0, "message": map[string]string{"role": "assistant", "content": strings.Repeat("x", *responseBytes)},
			"finish_reason": "stop",
		}},
		"usage": map[string]int{"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
	})
	if err != nil {
		return err
	}
	b := &benchmark{body: body, hold: time.Duration(*holdMS * float64(time.Millisecond)), config: map[string]any{
		"response_bytes": *responseBytes, "response_body_bytes": len(body), "hold_ms": *holdMS,
		"h2_max_streams": *streams, "go_version": runtime.Version(),
		"cpu_count": runtime.NumCPU(), "gomaxprocs": runtime.GOMAXPROCS(0), "pid": os.Getpid(),
	}}
	b.current.Store(&counters{})
	api := http.NewServeMux()
	api.HandleFunc("POST /v1/chat/completions", b.completion)
	api.HandleFunc("GET /v1/models", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, map[string]any{"object": "list", "data": []any{map[string]string{"id": "fake-model", "object": "model"}}})
	})
	protocols := new(http.Protocols)
	protocols.SetHTTP1(true)
	protocols.SetHTTP2(true)
	server := &http.Server{
		Handler: api, Protocols: protocols, HTTP2: &http.HTTP2Config{MaxConcurrentStreams: *streams},
		TLSConfig: &tls.Config{MinVersion: tls.VersionTLS12}, ReadHeaderTimeout: 10 * time.Second,
		ConnState: func(_ net.Conn, state http.ConnState) {
			switch state {
			case http.StateNew:
				b.connections.Add(1)
				peak(&b.peakConnections, b.openConnections.Add(1))
			case http.StateClosed, http.StateHijacked:
				b.openConnections.Add(-1)
			}
		},
	}
	admin := http.NewServeMux()
	admin.HandleFunc("GET /metrics", func(w http.ResponseWriter, _ *http.Request) { writeJSON(w, b.snapshot()) })
	admin.HandleFunc("POST /reset", func(w http.ResponseWriter, _ *http.Request) {
		if b.current.Load().active.Load() != 0 {
			http.Error(w, "Drain requests before resetting metrics", http.StatusConflict)
			return
		}
		b.current.Store(&counters{connectionBaseline: b.connections.Load()})
		b.peakConnections.Store(b.openConnections.Load())
		writeJSON(w, b.snapshot())
	})
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt)
	defer signal.Stop(stop)
	shutdown := make(chan struct{}, 1)
	admin.HandleFunc("POST /shutdown", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, map[string]bool{"ok": true})
		select {
		case shutdown <- struct{}{}:
		default:
		}
	})
	control := &http.Server{Handler: admin, ReadHeaderTimeout: 10 * time.Second}
	cert, key, err := certificate(*certDir)
	if err != nil {
		return err
	}
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return err
	}
	defer listener.Close()
	adminListener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return err
	}
	defer adminListener.Close()
	errors := make(chan error, 2)
	go func() { errors <- server.ServeTLS(listener, cert, key) }()
	go func() { errors <- control.Serve(adminListener) }()
	if err = json.NewEncoder(os.Stdout).Encode(map[string]any{
		"endpoint":  "https://" + listener.Addr().String() + "/v1/chat/completions",
		"admin_url": "http://" + adminListener.Addr().String(), "ca_file": cert, "config": b.config,
	}); err != nil {
		return err
	}
	select {
	case <-stop:
	case <-shutdown:
	case err = <-errors:
		return err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = server.Shutdown(ctx)
	return control.Shutdown(ctx)
}

func main() {
	var err error
	if len(os.Args) == 2 && os.Args[1] == "client" {
		err = runClient()
	} else {
		err = run()
	}
	if err != nil {
		log.Fatal(err)
	}
}
