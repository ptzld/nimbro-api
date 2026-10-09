# NimbRo API Agent Guide

## Scope and Priorities

`nimbro-api` is a Python API-client framework with consistent settings, result
tuples, retries, logging, and shared HTTP transport. Keep integrations small and
consistent with existing clients rather than adding a framework for each provider.

- Read the closest client, its base implementation, and its tests before editing.
- Preserve prompts, model defaults, request fields, call ordering, retry behavior,
  context handling, and result schemas unless the requested change requires it.
- Treat those contracts as public behavior: downstream applications depend on
  stable settings, payloads, and context lifecycles.
- Prefer the smallest correct change. Avoid speculative helpers, provider
  branches, compatibility aliases, dependencies, and unrelated refactors.
- Preserve user edits and untracked files. Check the worktree before and after;
  do not stage, revert, delete, or reformat unrelated work.

Use the recurring patterns below for new code. Existing clients contain some
historical inconsistencies; copy their architecture and applicable behavior,
not stale descriptions, unrelated capabilities, or incidental quirks.

## Package and Layout

- Distribution: `nimbro-api`; import package: `nimbro_api`.
- Python 3.9+, Setuptools, BSD-3-Clause. Metadata lives in `pyproject.toml`.
- `httpx` is the only required runtime dependency. Keep acceleration and visual
  dependencies optional; preserve installation and imports with the minimal set.
- `src/nimbro_api/client/`: public `Client` facade and `ClientBase` machinery.
- `src/nimbro_api/core/`: process-wide settings, credentials, caches, deferred jobs.
- `src/nimbro_api/utility/`: HTTP, assertions, logging, I/O, strings, and visuals.
- `src/nimbro_api/api/<family>/client/`: public clients and their defaults.
- `src/nimbro_api/api/<family>/base/`: API implementations.
- `src/nimbro_api/api/<family>/test/`: API-specific test functions.
- `src/nimbro_api/api/openai/parser/`: completion parsing extensions.
- `src/nimbro_api/test/`: custom test runner and shared assets.

Public usage follows `import nimbro_api` and names such as
`nimbro_api.openai.ChatCompletions` or `nimbro_api.misc.No`. Root exports
are assembled from API families and a process-wide Core instance; avoid adding
alternative import paths or manually constructing Core objects.

Keep client imports, `__all__`, and README tables in matching alphabetical order.
Do not edit generated build, distribution, egg-info, or cache files as source.

### Choosing a Reference

Choose a client with the same kind of operation, then read its client, base, and
test together. Paths below are relative to `src/nimbro_api/api/`:

- `misc/{client/no.py,base/no_base.py,test/no_test.py}`: small, unauthenticated GET.
- `openai/{client/classify.py,base/classify_base.py,test/classify_test.py}`:
  authenticated JSON requests and optional request fields.
- `openai/{client/transcriptions.py,base/transcriptions_base.py,test/transcriptions_test.py}`:
  media conversion, multipart uploads, and format-dependent results.
- `nimbro_vision_servers/{client/kosmos2.py,base/kosmos2_base.py,test/kosmos2_test.py}`:
  health/model lifecycle helpers and image inference.

Use `src/nimbro_api/client/client.py` and `src/nimbro_api/client/client_base.py`
for the shared contract. Consult the larger chat/composite clients when working
on contexts, streaming, batching, or multi-stage operations; their extra machinery
is not a basic client template.

## Client and Base Contract

A new integration normally adds three files:

```text
api/<family>/client/<name>.py
api/<family>/base/<name>_base.py
api/<family>/test/<name>_test.py
```

Export its public class in the family's `__init__.py` and add its README row.

- Use snake_case module names, a PascalCase public class, and the same class name
  suffixed with `Base` for its implementation.
- The public class inherits `Client`. It owns API documentation and a module-level
  `default_settings` dictionary, not validation or request-processing logic.
- Its constructor is `__init__(self, settings=None, **kwargs)` and passes the
  concrete base class, settings, defaults, and keyword arguments to `Client`.
- The implementation inherits `ClientBase`, binds shared helpers where needed,
  and sets `self._initialized = True` after successful initialization.
- `ClientBase.__init__()` already invokes the concrete `set_settings()`. That
  method must work before helper bindings or members assigned after `super()`
  exist. Bind helpers such as `self.get_api_key = get_api_key.__get__(self)` after
  `super()`, then emit the established initialization debug message and set the flag.
- Public operational methods delegate through
  `self._base.wrap(payload_count, self._base.method, *args, **kwargs)`.
  `get_settings()` uses the established direct delegation.
- Base methods return `(success, message, *payloads)`. `payload_count` counts only
  the values after the boolean and message. Typical request failures return
  `(False, message, None)`; preserve specialized contracts where they exist.
- Keep constructors free of live API requests.

Public method order is constructor, `get_settings()`, `set_settings()`, applicable
credential/discovery/lifecycle methods, then the main operations. Base classes
start with constructor and `set_settings()`, followed by implementation methods.
Explicitly expose and document settings methods in each concrete public client.

The delegation shape is:

```python
def get_settings(self, name=None):
    return self._base.get_settings(name)

def set_settings(self, settings=None, **kwargs):
    return self._base.wrap(0, self._base.set_settings, settings, **kwargs)

def classify(self, prompt, **kwargs):
    return self._base.wrap(1, self._base.classify, prompt, **kwargs)
```

These bodies still need the public docstrings described below. Pass operation
arguments positionally through `wrap()` in the base method's signature order.
The remaining `**kwargs` are settings overrides and wrapper controls, not an
arbitrary provider payload. The base operation takes its explicit arguments and
reads configuration from `self._settings`; it does not repeat `**kwargs` handling.

`wrap()` serializes operations on each instance with a non-reentrant lock. Inside
a base operation, call other base methods directly; do not re-enter the same
instance through its public facade or another `wrap()` call.

### Settings

Every concrete default dictionary includes `logger_severity`, `logger_name`, and
`retry`. Ordinary API clients use `logger_severity=None` to inherit process-wide
severity, a descriptive `logger_name`, and normally `retry=2`.

Keep defaults, the `Settings` docstring, and per-setting validation in matching
logical order: logging/display controls, endpoint registry and selection,
model/validation controls, operation-specific settings, timeouts, then retries.
Use established setting names such as `message_results`, `endpoints`, `endpoint`,
`model`, `timeout_connect`, and `timeout_read` when their meanings apply.

- Keep persistent configuration in settings and per-request data in method
  arguments. Do not expose the same option through competing mechanisms.
- Defaults define the allowed top-level settings. Add a setting to defaults,
  documentation, validation, and its consuming code together.
- Write nested defaults as nested dictionaries, not dotted keys. Dotted names
  are accepted in updates; literal dots in default-setting keys are rejected.
- `mute` and `persist` are reserved wrapper controls, not settings.

Base `set_settings(self, settings, mode="set")` follows this order:

1. Obtain complete target settings through `_introduce_settings()`.
2. Validate settings and their combinations with the shared assertion helpers.
3. Update derived members such as `_endpoint` after validation.
4. Return `_apply_settings(settings, mode)`.

Use one `# setting_name` block per setting and finish with `# apply settings`.
The shared layer validates `logger_severity`, `logger_name`, and `retry`; concrete
bases validate their additional settings. Pass `mode` through unchanged. Validate
the complete candidate, including cross-setting constraints, before changing
derived instance state. During validation use the local `settings` dictionary,
not the previous `self._settings` or the locking public getter.

Use native partial updates and dotted-key support. Do not add custom merging or
mutate the public object's `_base._settings` directly. A nonempty `settings`
dictionary and setting keyword arguments are mutually exclusive.

- Per-call setting overrides are temporary unless `persist=True`.
- `mute=True` suppresses the wrapped operation's logs.
- `set_settings(None)` resets to initialization settings, not necessarily defaults.
- `get_settings()` returns a defensive copy. With endpoints, its full snapshot
  consolidates the selected endpoint into a dictionary with `name` and omits the
  endpoint registry. `get_settings('endpoint')` returns the stored endpoint name.
  Preserve the `set_settings(get_settings())` round trip.
- Numeric connect/read timeouts must be positive; `None` means no timeout.
  Document units, ranges, empty values, and sentinels for every setting that uses
  them. Distinguish `None`, `False`, zero, and an empty collection explicitly.

### Retries and Errors

`ClientBase.wrap()` owns serialization of operations, temporary settings,
exception handling, retries, and result logging. Reuse it.

- `retry=False` or `0` disables retries; `True` permits unlimited retries;
  a positive integer sets the number of additional attempts.
- Expected invalid arguments/settings raise `UnrecoverableError` through shared
  assertions. The wrapper does not retry these errors.
- During construction, initialization failure raises `UnrecoverableError` to the
  caller. After initialization, wrapped failures normally become failure tuples
  with the declared number of payload slots. Direct `get_settings()` can raise.
- Ordinary unsuccessful operation tuples follow the existing wrapper retry path.
- Do not add client-specific retry loops, backoff, status-code policies, or an
  SDK's competing retry layer unless explicitly requested.
- Shared wrapper changes affect every client and require cross-client checks.

Keep all success and failure branches the same tuple length. Unpack helper
results as `success, message, ...`, propagate useful failure messages, and clear
failed payloads to `None` unless the operation has an explicit partial-result
contract. A successful empty result is not automatically a failure.

## Endpoints and Transport

Use `utility.api.validate_endpoint()`, `get_api_key()`, `post_request()`,
`get_request()`, and `http_request()` as appropriate. Do not create another HTTP
client/session per integration. Public APIs are generally synchronous; shared
HTTP machinery manages an asynchronous client in a background event loop.
Preserve its cancellation, connection reuse, and process lifecycle behavior.

- Endpoint dictionaries identify `api_url` and, where required, `key_type` and
  `key_value`. A directly supplied endpoint also has `name`.
- Validate a nonempty registry and its names, then each registry entry with
  `require_name=False`. Validate an inline selection with `require_name=True`,
  insert/update it in the registry, select its name, and remove `name` from the
  stored definition. Derive `self._endpoint` only after settings validation.
- `key_type` is `"environment"` or `"plain"`; an empty plain key supports
  unauthenticated servers. Omit the Authorization header for an empty key.
- Reuse the standard Bearer, `HTTP-Referer`, and `X-Title` header pattern. For JSON
  requests include `Content-Type: application/json`; for multipart uploads let
  the transport generate the content type and boundary.
- Supply all keyword-only policies to `validate_endpoint()`: `flavors`,
  `require_key`, `require_name`, `allow_models`, and the diagnostic `setting_name`.
  `allow_models` controls whether `models_url` is permitted. Apply the same API
  contract to registry entries and inline endpoints.
- Use `flavors=None` for one shared wire contract. URL, credentials, or model-name
  differences alone do not justify API flavors or provider-specific classes.
- Preserve existing API flavors where they implement real protocol differences.
- Add model discovery only when needed and supported by the actual catalog
  contract. Reuse family helpers when their catalog and validation semantics fit.
  Preserve existing model-validation behavior; do not assume every compatible
  inference endpoint also supplies a complete Models API.
- Never embed credentials or bypass the framework's credential handling.

### Request Implementation

Follow the reference client's procedural stages, with short section comments:

1. Start elapsed timing with `stamp = time.perf_counter()` where reported.
2. `# parse arguments` / `# parse image` / `# parse audio`: validate and convert
   inputs with shared helpers.
3. Retrieve credentials and perform applicable connection/health/model checks.
   Preserve the established ordering when changing an existing client.
4. `# construct payload`: build headers and data explicitly from validated inputs
   and settings. Do not send the entire settings dictionary to the provider.
5. `# use API`: call the shared request helper with `api_name`, `api_url`, headers,
   data, `timeout=(connect, read)`, and `logger=self._logger` as named arguments.
6. `# parse response` / `# parse API response`: decode and validate the successful
   response according to the endpoint's actual formats.
7. `# finalize response`: build the success message or clear failed payloads,
   then return `success, message, result`.

The transport returns an HTTP response object, not decoded JSON. Check its
success flag before parsing. Keep JSON/text fallbacks only where the API supports
both. Catch decoding/response-contract errors around parsing and normalize them
to a failure tuple; malformed server data should not be classified as invalid
caller input and inadvertently made unretryable.

Omit optional fields only according to their documented sentinel semantics.
Do not use a truthiness filter that drops valid `False`, zero, or empty values.
Likewise, do not turn omission into JSON `null` unless the wire contract calls for
it. Preserve full response metadata when returning a provider dictionary; any
normalized or extracted result must match the public method's documented schema.

### Shared I/O and Caches

- Use `utility.io` for existing JSON, Base64, download, image, and audio parsing
  operations rather than duplicating conversion or file-handling code.
- Describe the accepted media forms precisely. Different helpers accept paths,
  URLs, Base64, bytes, or image objects; do not promise every form indiscriminately.
- Use process-wide `nimbro_api.query_cache()` / `update_cache()` for shared cached
  data. Match the helper's category and identifier convention, usually the
  resource URL, so caches do not mix unrelated endpoints.
- For discovery methods, `age=0` forces a request, a positive age permits cached
  data up to that many seconds old, and `age=None` permits any cache age. A cache
  miss still requires a request. Keep this distinct from boolean validation
  controls that enable or disable checking.
- Internal Core calls commonly use `mute=True`, with their messages logged at
  debug/warn level by the calling base. Do not overwrite the main operation's
  `success` and `message` with incidental cache-update results.
- Keep optional libraries behind the existing lazy-import or availability/fallback
  patterns. Importing the package must work with only required dependencies.

## Validation and Coding Style

- Use `assert_type_value`, `assert_keys`, and `assert_log` for expected contract
  violations; they raise `UnrecoverableError`. Prefer these to bare assertions
  in production paths.
- Use named helper arguments: `obj=`, `type_or_value=`, `name=`, `expression=`,
  and `message=`. `type_or_value=[str, None]` accepts types/sentinels;
  `type_or_value=["input", "messages"]` validates enumerated values.
- Validate type before length, range, membership-dependent indexing, or nested
  contents. Name the precise location, e.g. `setting 'model'`, `argument 'prompt'`,
  or `key 'text' in transcription response`.
- `assert_keys(..., mode="required")` allows extra keys; `"whitelist"` allows
  only listed keys without requiring all of them; `"match"` requires exactly the
  given keys. Use strict keys for owned configuration and required keys for
  extensible provider responses.
- Validate fields the operation consumes, relies on, or produces. Avoid turning
  a focused integration into a validator for unrelated provider metadata.
- Distinguish booleans from numeric values; the type assertion helper rejects
  booleans as integers. Check finiteness where required by the contract.
- Keep extra response metadata when the API returns full response dictionaries.
- Catch exceptions at the boundary that owns parsing or failure normalization.
  Do not disguise programming errors as ordinary invalid user input.
- Use snake_case for functions, variables, and modules; PascalCase for classes;
  leading underscores for internal members. Keep public types in docstrings
  rather than introducing annotation/model infrastructure for a plain-dict API.
- Use four-space indentation and one blank line between module-level definitions
  and between methods. Preserve long signatures/assertion calls when readable;
  do not impose a formatter's line wrapping or two-blank-line style.
- Preserve formatting of unchanged expressions when editing nearby code. Do not
  add line breaks, change quoting, or otherwise reformat code unless required by
  the requested change. Keep diffs focused on meaningful changes.
- In code, use single quotes for dictionary keys, key lists, and subscriptions;
  double quotes for string values, URLs, enum values, messages, and f-strings:
  `{'key_type': "environment"}` and `f"Model '{settings['model']}'."`.
- Put settings and substantial payload dictionaries on multiple lines, one entry
  per line, with matching nested indentation. Preserve nearby trailing-comma
  style. Use parentheses and adjacent string literals for long text constants.
- Group standard-library, third-party, and internal imports with blank lines.
  Clients import `Client` via `nimbro_api.client` and their base relatively via
  `..base.<name>_base`; bases use shared package helpers and relative family
  utilities. Avoid wildcard imports. Preserve local symbol ordering rather than
  sorting unrelated imports; exported client lists are alphabetical.
- Keep comments sparse. Follow the existing section comments for settings
  validation, payload construction, response parsing, and finalization.
- Runtime syntax must support Python 3.9. Docstring union notation such as
  `str | None` does not imply that the same annotation is safe to evaluate there.
- Copy at settings, ownership, mutation, and public-return boundaries. Avoid
  unnecessary copies of large images, masks, or batches.

## Documentation and Logging

### Public Docstrings

- Use triple double quotes on their own lines, including the opening delimiter.
  Start with a short summary, then a blank line before sections. The class
  introduces the API and its supported capabilities; the constructor says
  "Create a client implementing ..." and documents initialization.
- Use the section order `Args`, `Settings` (on `get_settings()`), `Raises` when
  applicable, `Returns`, then `Notes`. Omit sections that do not apply.
- Section headings end in `:`. Entries are indented four more spaces; their
  descriptions another four. Use `name (type):` or `name (type, optional):`.
  `optional` describes an argument with a default, not merely one allowing `None`.
- Use built-in type spellings, `|` for alternatives, and containers such as
  `list[str]` or `tuple[bool, str, dict | None]` in type descriptions.
- In prose, single quotes identify arguments, settings, variables, attributes,
  and dictionary keys: 'settings', 'endpoint', 'api_url', 'default_settings'.
  Double quotes identify literal string values: "environment", "plain", "json",
  and "". Backticks mark Python types, non-string literals, expressions, and
  callable references: `dict`, `None`, `True`, `get_settings()`,
  `result['data'][0]`. Do not treat these three forms as interchangeable emphasis.
- Document all settings in `get_settings()` in defaults order, including nested
  structures and conditional applicability. Refer to module-level
  'default_settings' for setting defaults; describe argument defaults explicitly
  with "Defaults to ..." and explain special values with "Use ... to ...".
- Constructor/settings docstrings explain partial dictionaries, dotted keys,
  and the mutual exclusion with setting keyword arguments. Operational methods
  document setting overrides plus `persist` and `mute` under `**kwargs`.
- `Returns` spells out the full tuple type followed by "A tuple containing:" and
  a bullet for each element, in order. Describe payload structure and failure
  values, not just "the result". Document raised exceptions only where they can
  actually reach the caller; account for `wrap()` converting ordinary failures.
- Describe accepted shapes, ranges, units, return structures, state changes,
  cache behavior, and media formats wherever applicable. Avoid boilerplate
  claims about models, caching, or conversions the client does not implement.

For example, preserve this distinction between a setting name and its values:

```text
The value of 'key_type' must be either "environment" or "plain".
Use `None` to return all settings as a dictionary. Defaults to `None`.
```

Keep README entries concise and consistent with the public class and constructor:
link the implementation and the authoritative API reference for the implemented
contract. Provider/setup explanations belong in the appropriate client docs.

### Runtime Messages

- Use the native logger and `escape` colors, not a custom logging abstraction.
- Respect severity, process-wide mute, per-call mute, and `message_results`.
- Validation messages should identify the exact setting/argument and use
  expectation-first phrasing: `Expected <contract> but got '<observed value>'.`
  Runtime messages generally single-quote names and interpolated values, unlike
  the docstring distinction between names and string literals.
- Keep ordinary messages concise, action-first, and grammatical for zero, one,
  and multiple items. Format elapsed time as `'{duration:.3f}s'`.
- Let `wrap()` log the final operation result; avoid logging the same summary in
  the base and returning it for another log. Use debug for intermediate progress
  and shared-helper messages, with warnings/errors where appropriate.
- Separate machine-readable payloads from readable messages. `message_results`
  controls result details in the message, not the returned data. Use `format_obj`
  for generic object display and `escape` for existing color conventions; reset
  styles where their intended span ends. Formatting must not mutate results.

## Testing and Verification

Use `~/ws/venv/bin/python` when available. Otherwise use an available interpreter
with the required dependencies. Ensure verification imports this checkout, for
example with `PYTHONPATH=src`.

This repository uses its own runner, `nimbro_api.test()`:

- API tests are zero-argument `test_*` functions in `<name>_test.py`, commonly
  ordered/named `test_1_utilities`, `test_2_endpoint`, `test_3_inference`.
- Use the shared assertion helpers and established success/message/payload checks.
- The default `function="utilities"` filter selects offline utility tests.
  These must not make outgoing connections. The runner also executes common
  client checks and package utility checks by default (`common=True`,
  `utilities=True`); these switches are independent of the function filter.
- Live inference/endpoint tests require credentials and services. Do not run them
  unless the user explicitly authorizes it. `function=None` selects all test
  functions, including live ones.
- Keep committed API tests in the existing function-based style and runner rather
  than introducing another test framework. Keep temporary verification scripts
  outside the repository.
- The runner writes `src/nimbro_api/test/test.txt` and `test_pretty.txt`. Account
  for those side effects and do not include generated logs in a source change.

Tests unpack and check the public contract in this order:

```python
success, message, result = client.classify(prompt="This is a test.")
assert_type_value(obj=success, type_or_value=bool, name="success")
assert_type_value(obj=message, type_or_value=str, name="message")
assert_log(expression=success, message=message)
assert_type_value(obj=result, type_or_value=dict, name="result")
```

Add meaningful payload assertions for the API contract. Use package test assets
for media, portable paths, and restore changed global settings/environment when
testing utilities. Do not make offline tests depend on external credentials.

Run the narrowest relevant offline checks and pylint for the changed scope.
Shared helper or wrapper changes justify broader common-client verification.
CI runs pylint and exercises the default offline test runner across supported
Python versions on Linux, Windows, and macOS. Representative commands from the root:

```bash
PYTHONPATH=src python -m pylint src/nimbro_api
PYTHONPATH=src python -c "import nimbro_api, sys; sys.exit(nimbro_api.test())"
git diff --check
```

For focused changes, select the relevant files for pylint and the relevant
`provider`, `module`, and offline `function` filters for `nimbro_api.test()`.
Filters are case-sensitive substring matches; `module` matches test filenames
for API tests but class names for common checks. Run common checks separately
without a filename filter when needed. The runner returns a failure count.
Do not run the full suite for a local formatting or documentation edit.

## Before Finishing

- Compare the change with the closest existing client/base/test pattern.
- Confirm defaults, validation, docstrings, and actual payload behavior agree.
- Check export/README ordering and that documentation links describe the actual API.
- When changing a shared helper signature, update and check every call site.
- Preserve retry, settings restoration, response data, and context lifecycles.
- Review only the intended diff, including accidental generated files.
- Report which checks ran and which live tests were deliberately left unrun.
