# Meta Watcher documentation

Meta Watcher is a person-first desktop video overlay app powered by SAM 3.1. It captures from a webcam, RTSP stream, or local file; runs text-prompted segmentation on either Apple Silicon (MLX) or NVIDIA CUDA; keeps a stable empty-scene inventory of named objects; switches to tracked-people overlays when someone is present; and records one annotated MP4 plus JSON sidecar per occupancy event.

These docs are organized by the job the reader is trying to do.

## Getting started

- [Getting started](guides/getting-started.md) — install, configure, and run the CLI end to end.
- [CLI reference](reference/cli.md) — every flag supported by `meta-watcher`.
- [Configuration reference](reference/configuration.md) — every field of `config.default.json`, its default, and what it does.

## Architecture

- [Architecture overview](architecture/overview.md) — components, runtime threads, and the frame lifecycle.
- [State machine](architecture/state-machine.md) — inventory / person-present / cooldown transitions and the recording decisions tied to them.
- [Recording pipeline](architecture/recording.md) — pre-roll buffer, event start/finish, sink lifecycle, and metadata.
- [Inference backends](architecture/inference-backends.md) — how MLX and CUDA providers are built and where the subprocess boundary lives.

## How-to guides

- [Operate the web UI](guides/operator-ui.md) — controls, status panel, and inventory list.
- [Frontend development](guides/frontend.md) — build and iterate on the Astro bundle.
- [Testing](guides/testing.md) — run the suite, and what each module covers.
- [Troubleshooting](guides/troubleshooting.md) — common failures around cameras, model access, and performance.

## Reference

- [HTTP API reference](reference/http-api.md) — every endpoint exposed by `meta_watcher.web.server.build_app`.
- [Python API reference](reference/python-api.md) — classes and functions intended to be used from embedding code.
- [Configuration reference](reference/configuration.md) — config schema and defaults.
- [CLI reference](reference/cli.md) — `meta-watcher` flags.

## Internals (for maintainers)

- [Pipeline internals](internals/pipeline.md) — `StreamProcessor` and `StreamRuntime` in detail.
- [Web runtime state](internals/web-state.md) — the threading contract behind `RuntimeState`.
- [Inference patches and provider lifecycle](internals/inference.md) — the `sam3.perflib` fp32 patch and the MLX subprocess RPC protocol.

## Agent skills

Meta Watcher ships repo-local agent skills so Claude Code and compatible LLM tools can operate the project consistently. Start at the index:

- [Agent skills index](../.claude/skills/README.md).

## Machine-readable bundles

- [`../llms.txt`](../llms.txt) — compact LLM-facing index.
- [`../llms-full.txt`](../llms-full.txt) — single-file documentation bundle for ingestion.
