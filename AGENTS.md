# AGENTS.md — ai/ monorepo root

> Conventions for any agent operating in this repository.

## Zones

This repo is split into three zones by **purpose**, not by language:

| Zone        | Path        | Purpose                                       |
|-------------|-------------|-----------------------------------------------|
| Harness     | `agents/`   | Agent implementations (azathoth, yog-sothoth) |
| Knowledge   | `research/` | Karpathy 3-layer research (raw/wiki/outputs)  |
| Shared libs | `src/lib/`  | Lab-wide utility package                      |
| Dev probes  | `scripts/`  | One-shot dev scripts                          |

## Dependency direction

```
agents/ ──▸ src/lib/
research/raw/ notebooks ──▸ src/lib/
src/lib/ ──▸ (nothing in this repo)
agents/ ✗ research/   (never cross-import)
```

## Toolchain

- **Shell**: nushell only. Justfiles use `set shell := ["nu", "-c"]`.
- **Task runner**: `just`.
- **Python**: `uv` + `ruff` + `ty`. Never pip, black, or mypy.
- **Nix**: `alejandra` + `statix` + `deadnix` + `nil`. Rebuilds via `nh`.
- **CLI preferences**: Rust replacements (`eza`, `fd`, `rg`, `sd`, `bat`, `bottom`).

## Style rules

- Functional over imperative.
- No speculative abstractions — build what's needed now.
- No drive-by reformatting of files outside the current task.
- Comments explain **why**, not what.
- All typed code is fully type-hinted. Non-negotiable.

## Files at this level

- `README.md` — project overview.
- `justfile` — root task runner.
- `pyproject.toml` + `uv.lock` — Python workspace config.
- `ai.code-workspace` — editor workspace.
- `LICENSE` — MIT.
