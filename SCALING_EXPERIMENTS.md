# Throughput tuning: settings, threads, and processes

## Choose a layout from workload and compute budget

Let `X` be the logical CPUs available to the job, `P` processes, `T` request threads
per process, and `C = P × T` configured request slots. The layouts below assume
independent clients for simultaneous requests; a shared instance serializes them.

| Task / objective | Initial layout | Constraint or measured reason |
|---|---|---|
| Immediate/short responses, maximum req/s | `P ≈ X`, `T = 5`; compare lower counts and `T = 10` | On 32 CPUs with optimized HTTP/1.1, 5/10/25 threads gave about 17,400/14,100/12,600 req/s. Five was the best tested count, not a proven optimum |
| Responses lasting seconds, specified target rate `R` | Estimate `C = ceil(R × L)` using mean end-to-end latency `L` at the intended load; start with `P = min(X, ceil(C / 50))`, `T = ceil(C / P)` | Rate is bounded near active concurrency divided by latency; this sizing rule is a starting heuristic, not a measured law for other machines |
| Responses lasting seconds, maximum rate with lower CPU/tail cost | `P ≈ X`, `T = 100–200` | Optimized HTTP/1.1 at 32 × 200/five seconds reached about 1,230 req/s, 2 ms CPU per success, and 6-second p99 |
| Higher HTTP/1.1 rate, accepting higher CPU/tail cost | Compare `P ≈ X`, `T = 300–400` with `P ≈ 2X` and fewer threads at the same `C`; use dependency optimizations | At 32 × 400/five seconds, about 2,180 req/s required 3.3 ms CPU per success and 9.4-second p99. Stock dependencies at that layout gave only 396 req/s |
| HTTP/2, one connection per process to the same origin | Stock httpcore 1.0.9: `T ≤ 100`. With corrected negotiation: `T ≤ S`, the server's advertised stream limit | Above these limits, extra threads wait rather than raise active server concurrency. The reference server allowed `S = 250` |

For example, an eight-CPU job targeting 200 req/s at five-second mean latency needs
about 1,000 slots before further dispatch overhead. Compare 8 × 125 with 16 × 63.
With stock HTTP/2's 100-stream cap, 8 × 125 provides only 800 active streams to one
origin; with one connection per process, at least ten processes are needed for
1,000 active streams.
These smaller-host layouts are extrapolations, not measured capacities.

If the provider's request/token quota or server capacity is below `R`, increasing
`C` does not meet the target. If latency rises while successful req/s stops rising,
the additional slots are increasing waiting time, not throughput. Compare process
layouts at fixed `C` to distinguish distribution overhead from additional capacity.

## Settings that determine the result

Transport settings are process-wide (`nimbro_api.set_settings()`); apply them in
each worker before traffic. Other settings in the table belong to ChatCompletions.

| Setting | Throughput-oriented configuration | Scope / limitation |
|---|---|---|
| `stream` | `False` for the non-streaming capacity figures below | This path uses notification waits and no per-request completion helper/pipe. Streaming has no measured capacity recommendation here |
| `http_use_http2` | `False` for HTTP/1.1; `True` for HTTP/2 | Requires `httpx[http2]` and server support. Enabling it does not remove httpcore 1.0.9's outgoing-stream cap |
| `http_max_connections` | For HTTP/1.1 to one origin, start with at least `T`, or `None` to remove the pool bound | With `T = 200` and a limit of 100, no more than 100 HTTP/1.1 requests are in service through that pool. Limits cover all origins in the process, not each client |
| `http_max_keepalive_connections` | For repeated HTTP/1.1 bursts of `T` requests, retain at least `T` idle connections if the resource budget permits; keep this ≤ a finite active limit | Default 100 retains at most 100 idle connections after a 400-connection burst. `None` removes the bound; `0` disables idle retention. Optimal finite limits were not measured |
| `http_keepalive_expiry` | Use 30.0 seconds to match reference settings; choose a lifetime longer than gaps between bursts when connection reuse is desired | Seconds; `None` removes expiry, `0` expires idle connections immediately. The optimal lifetime for other traffic patterns was not measured |
| `timeout_connect`, `timeout_read`, `timeout_completion` | Reference comparisons used 60 seconds for all three; choose production deadlines from measured latency and the task's deadline | Connect also bounds transport pool waiting; read bounds individual reads, not total response time. Five-second optimized work had tails near ten seconds; a five-second completion deadline leaves no dispatch margin |
| `timeout_chunk_first`, `timeout_chunk_next` | For streaming, exceed the allowed first-token/inter-chunk pause | These deadlines are independent of total completion time; non-streaming comparisons do not determine their values |
| `retry` | `False` when determining failure-free capacity; then measure the required retry policy | Retries add attempts to the same provider quota and extend public-call latency. Report successful logical requests separately from attempts |
| Logging | `logger_severity="off"` for an overhead-isolated comparison; then measure with production logging | Reference throughput excludes diagnostic logging; it does not include the cost of printing prompts/completions |
| `validate_model`, `correction`, `parser` | Reference settings were `False`, `False`, and `[]` | Use those values only if the task does not require those behaviors. Validation/correction requests and parsing work are outside the reference workload |

**Resource budget:** HTTP/1.1 at `P × T` active requests requires approximately
`T` client sockets per process and `P × T` accepted sockets at the endpoint, plus
other descriptors. With one HTTP/2 connection per process, those socket counts
are approximately one and `P`; stream capacity remains a separate limit. At
12,800 HTTP/1.1 slots, the endpoint needs thousands of descriptors even though
each 400-thread client process needs only hundreds. Reference runs used a
65,536 descriptor limit. Check both sides' limits and the available TCP port range.

Non-streaming uses `T` request threads plus shared HTTP machinery per process;
streaming adds roughly one helper thread and pipe per active completion. Per-request
memory and per-process RSS were not quantified, so no memory-based maximum is
claimed for large contexts, media, or smaller hosts.

## Measured capacity: stock versus optimized dependencies

Reference: Linux, 32 logical CPUs, Python 3.14.3, package 0.2.5 with the non-streaming
path specified above, httpx 0.28.1, httpcore 1.0.9, AnyIO 4.14.0, h2 4.4.1. Workload:
local HTTPS, 2 KiB prompt and completion text, `stream=False`, unlimited active/idle
connections, 30-second keepalive expiry, 60-second deadlines, retries/validation/
correction/parsing extensions/logging disabled. These figures do not cover real
inference, streaming, or other payload sizes. They require the non-streaming fast
path and notification-based HTTP waits, not an implementation retaining per-request
completion helpers/pipes.

"Optimized" means pool-scan/socket-lookup changes for HTTP/1.1, and corrected
outgoing-stream negotiation for HTTP/2. The HTTP/2 figures do not depend on the
HTTP/1.1 changes. All rows were failure-free. Rates include final drain; immediate,
one-second, and five-second submission windows were 12, 20, and 30 seconds.

| Service time | Protocol | Processes × threads | Slots | Stock req/s | Optimized req/s | Peak active requests: stock / optimized |
|---|---|---|---:|---:|---:|---:|
| Immediate | HTTP/1.1 | 32 × 5 | 160 | 16,468 | 17,368 | 39 / 52 |
| Immediate | HTTP/2 | 32 × 5 | 160 | 15,489 | — | 85 / — |
| 1 s | HTTP/1.1 | 32 × 100 | 3,200 | 3,048 | 3,111 | 3,200 / 3,200 |
| 1 s | HTTP/2 | 32 × 100 | 3,200 | 2,829 | — | 3,200 / — |
| 5 s | HTTP/1.1 | 32 × 200 | 6,400 | 994 | 1,221 | 6,400 / 6,400 |
| 5 s | HTTP/1.1 | 32 × 400 | 12,800 | 396 | 2,179 | 11,324 / 12,800 |
| 5 s | HTTP/2 | 32 × 200 | 6,400 | 534 | 1,167 | 3,200 / 6,400 |
| 5 s | HTTP/2 | 64 × 200 | 12,800 | 1,040 | 2,284 | 6,400 / 12,800 |

**Largest demonstrated capacities:** approximately **17,400 successful req/s** for
immediate work and **12,800 peak active requests** for five-second work. No hardware
maximum was established: the largest optimized slow-request layouts used about
7.2 average worker CPU cores for HTTP/1.1 and 3.5 for HTTP/2, below the 32-CPU budget.
This aggregate headroom does not measure saturation of an individual event loop.

Stock HTTP/1.1 at 32 × 400 illustrates why peak concurrency alone is insufficient:
11,324 active requests at peak accompanied only 396 req/s and approximately
36-second p99 for five-second service. Optimized dependencies at the same layout
gave 2,179 req/s and 9.4-second p99. The HTTP/1.1 and HTTP/2 largest-rate examples
use different process layouts, so they do not establish a same-layout protocol winner.

## Identify the constraint before raising concurrency

| Scenario | Constraint established for this configuration | Response |
|---|---|---|
| Immediate responses, 32 processes, more than five threads | Five → ten → 25 threads reduced optimized HTTP/1.1 rate and increased p99 from about 18 → 59 → 202 ms; exact dominant CPU hotspot is unresolved | Reduce threads; profile the remaining CPU work before proposing another transport change |
| One-second HTTP/1.1, 32 × 100, stock dependencies | 3,048 req/s is close to the 3,200-slot/service-time bound; dependency changes added only about 2% | More slots are the relevant capacity change if the endpoint supports them |
| Five-second HTTP/1.1, 32 × 200–400, stock dependencies | Pool/socket overhead: dependency changes raised 994 → 1,221 and 396 → 2,179 req/s at these layouts | Apply the optimizations before treating 400 threads/process as a useful configuration |
| Stock HTTP/2, `T > 100`, one connection/process | Active streams ≤ `P × min(T, 100, S)`; 32 × 200 reached only 3,200 active requests | Use more processes or corrected stream negotiation; extra waiting threads do not raise this bound |
| Corrected HTTP/2, `T > S` | Active streams ≤ `P × min(T, S)`; with `S = 250`, 32 × 300 reached 8,000 rather than 9,600 | Increase process/connection count or reduce threads; changing the client's incoming-stream advertisement does not increase outgoing capacity |
| Corrected HTTP/2, five-second service below `S` | Approximately ten-second tails remain; their cause is unresolved | Allow deadline margin and measure dispatch/transport latency before attributing it to pooling or flow control |
| HTTP/1.1, finite connection limit below `T` | Connection count, rather than request-thread count, bounds active transport work | Raise the process-wide limit within the descriptor budget or lower `T` |
| Provider quota/server capacity reached | Additional slots increase waiting once successful req/s plateaus | Stop increasing concurrency; size to the provider's measured capacity and quota |
| Streaming, larger payloads/contexts, required tools/parsers | No measured thread/process optimum or capacity for these workloads | Use workload-specific measurements; the non-streaming 2 KiB figures are not estimates for them |

## Dependency setup for the optimized layouts

`patch_dependencies.py` supports **httpcore 1.0.9**. The default selection removes
a redundant nested pool scan and caches raw-socket lookup without skipping
readability checks. `--http2` additionally honors negotiated outgoing-stream
capacity; the script bundles both selections rather than offering HTTP/2 alone.

Run with the interpreter of the target environment:

```bash
python patch_dependencies.py apply
python patch_dependencies.py status
python patch_dependencies.py restore
```

For the HTTP/2 selection, add `--http2` consistently to all three commands. Stop
clients first and start fresh processes afterwards. Start from stock sources;
restore one selection before applying another. Backups are outside the checkout;
source/hash checks protect intervening edits. Custom `--backup-dir` paths must
differ between selections. These dependency edits are opt-in, not installed by
the package automatically.

## Already-tested optimization directions

The following conclusions apply to the reference versions and non-streaming
2 KiB workload with unlimited pools. They bound what is known, not all future workloads.

| Direction | Tested conditions / result | Implication |
|---|---|---|
| Completion helper/pipe removal and notification waits | Already present in the specified non-streaming path | This overhead is not an unimplemented optimization; streaming still uses its helper/pipe |
| Pool-scan and raw-socket changes for HTTP/2 | At 32 processes: `T = 5`/immediate, `T = 100`/one-second, and `T = 200`/five-second service gave rate differences of about −1% to +1% | No meaningful throughput gain established for those HTTP/2 layouts; stream negotiation is the demonstrated restriction above 100 threads |
| Four HTTP/1.1 transport pools | With the non-streaming fast path and pool/socket optimizations, 32 × 10/immediate gained about 1%; 32 × 200/five seconds lost about 2% and used more CPU per success | No throughput justification for default sharding in those workloads |
| Socket-health checks moved to HTTP/1.1 connection reuse | With pool/socket optimizations, 32 × 10/immediate and 32 × 200/five seconds established no general rate improvement | No demonstrated reason to change the health-check policy |

The exact remaining CPU hotspot for short requests and the corrected HTTP/2 tail
cause have not been identified. Existing low-sample profiles are insufficient to
attribute them to the GIL, parsing, pool locks, or HTTP/2 flow control.

## Measure another task or configuration

The benchmark defaults to one process per available CPU, 100 threads/process,
HTTP/1.1, five-second service, five-second warm-up, and 30-second measurement.
It inherits the package's pool settings and separate ChatCompletions timeouts
(currently 10 seconds connect, 20 read, 30 completion); `--request-timeout` overrides
all three. Omit it or pass `none` to inherit them. Diagnostic severity remains `off`.
This is one fixed slow-request layout, not a search for maximum throughput.

Vary `T` on both sides of the candidate value, then compare `P` at fixed `C`.
Record successful logical req/s, failures, mean/p99 latency, CPU per success, and
active server requests. Repeat close results; the reference windows do not support
claims of precise percentage improvements for another installation. For code
optimizations, compare against the already-optimized configuration, not just stock
dependencies or a slower completion path.

For the packaged local benchmark, install the checkout and Go 1.26+ on `PATH`:

```bash
python -m pip install -e . 'httpx[http2]'
python src/nimbro_api/test/benchmark.py --processes 32 --threads-per-process 5 \
    --duration 12 --warmup 2 --hold-ms 0 --request-timeout 60 \
    --max-keepalive-connections none --severity off
```

Use `--threads-per-process 200 --hold-ms 5000 --duration 30` for the slow reference
configuration; add `--http h2 --h2-max-streams 250` for HTTP/2. Scale `--processes`
to the job's CPU budget. Default prompt/response sizes are 2 KiB; reference pool
settings are `http_max_connections=None`, `http_keepalive_expiry=30.0`. The callable
`nimbro_api.benchmark()` inherits process-wide pool settings and accepts the same
workload controls. The printed settings identify the actual configuration.

Go is an endpoint-capacity reference, not equivalent client work. Throughput includes
drain, and approximate latency percentiles include failures. `severity="off"` keeps
statistics but hides diagnostics; use `"error"` for failure details. Results overwrite
`benchmark.txt` in the package test directory. This benchmark does not measure streaming.
