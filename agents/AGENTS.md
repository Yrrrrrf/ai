# AGENTS.md — agents/

> Conventions shared by all agent harnesses in this directory.
> Inherits: root `AGENTS.md`.

## Harness philosophy

Each agent is an **identity-and-action** harness:

- **A2A side** (skills, conventions, agent card) defines what the agent **is**.
- **MCP side** (tools, resources, server callbacks) defines what the agent **can do**.
- A **swappable model** sits at the bottom — provider choice is a pricing-and-fit decision.

## Provider abstraction

- No vendor strings in core logic. All model access goes through a provider
  abstraction layer (e.g. `azathoth/src/azathoth/providers/`).
- Provider swap is a configuration change, not a code change.

## Standards adopted

| Standard | Status         | Details                                               |
|----------|----------------|-------------------------------------------------------|
| A2A v1.x | Cards only     | Static `.well-known/agent-card.json`; no runtime yet. |
| MCP      | In use         | Agent-to-tool integration.                            |
| SKILL.md | Adopted        | One directory per skill, YAML frontmatter required.   |

## SKILL.md format

Each skill lives in `<agent>/skills/<skill-name>/SKILL.md`:

- `name`: kebab-case, ≤64 characters, lowercase + digits + hyphens only.
- `description`: ≤1024 characters, deliberately "pushy" (anti-undertriggering).
  Describe both *what* the skill does and *when* an agent should reach for it.
- Body: ≤500 lines.

## Agent card format

Located at `<agent>/.well-known/agent-card.json`. Follows A2A v1.x schema:
`name`, `description`, `url`, `version`, `protocolVersion`, `skills[]`,
`capabilities`. No auth section (localhost-only stub for now).

## Directory pattern

Each agent directory contains at minimum:

```
<agent>/
├── AGENTS.md              — agent-specific conventions
├── .well-known/agent-card.json
├── skills/                — SKILL.md-formatted skills
├── src/                   — implementation
└── ...
```
