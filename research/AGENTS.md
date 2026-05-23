# AGENTS.md — research/

> Schema for the Karpathy 3-layer knowledge territory.
> Inherits: root `AGENTS.md`.

## Structure

```
research/
├── raw/        — immutable source material (notebooks, papers, data dumps)
├── wiki/       — synthesized concept pages (one topic per file)
└── outputs/    — generated content (gitignored, disposable)
```

## Rules by layer

### `raw/` — Immutable sources

- **Read-only for agents.** Agents may read from `raw/` but must never edit
  files here.
- **Append-only workflow.** New material is added; existing material is never
  modified or deleted.
- Contains: notebooks (`.ipynb`), scripts (`.py`), PDFs, data dumps.
- Organized by topic: `cv/`, `evo/`, `ga/`, `neural-net/`, `notebooks/`
  (orphan notebooks), `sys/` (system diagnostics).

### `wiki/` — Synthesized knowledge

- **One topic per file.** Filename is a kebab-case slug of the topic.
- **Cite sources.** Every wiki page should reference its `raw/` sources at
  the bottom of the page.
- **Update index.** `wiki/index.md` maps every slug to its human-readable
  topic name. Update it on every page add or rename.
- Contains: synthesized markdown summaries of AI/ML concepts.

### `outputs/` — Generated content

- **Disposable.** Everything here is gitignored and can be regenerated.
- Contains: exported analyses, agent-generated reports, rendered content.

## Wiki ingestion

The mechanism for propagating new `raw/` files to `wiki/` updates is
**deferred**. For v1.0, maintenance is manual. This decision will be
revisited after 1–2 months of use to assess whether automation is warranted
based on actual update cadence.

## Naming conventions

- Wiki filenames: kebab-case slugs (e.g. `cnn.md`, `data-science.md`).
- The slug ↔ topic mapping is in `wiki/index.md`.
