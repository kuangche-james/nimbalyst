# Project skills

The skills in this folder are copied from [mattpocock/skills](https://github.com/mattpocock/skills) at commit `4ae2317`, under the MIT licence reproduced in [LICENSE-mattpocock-skills](./LICENSE-mattpocock-skills). They are plain copies so they can be edited for this repository; they do not update themselves.

## Included

From `skills/engineering/`: `ask-matt`, `codebase-design`, `diagnosing-bugs`, `domain-modeling`, `grill-with-docs`, `implement-spec`, `improve-codebase-architecture`, `pr`, `prototype`, `research`, `retro`, `setup-matt-pocock-skills`, `tdd`, `to-spec`, `to-tickets`, `triage`, `wayfinder`, `wizard`.

From `skills/productivity/`: `grill-me`, `grilling`, `handoff`, `teach`, `to-questionnaire`, `wait-what`, `writing-for-agents`.

## Left out, and what that changes

- `implement`: the name collides with this repository's `/implement` command (`.claude/commands/implement.md`), which executes a plan document. Where `ask-matt` and `triage` point to `/implement`, they now reach this repository's command, not Matt's skill.
- `code-review`: the name collides with the `code-review` skill already available in Claude Code. `implement-spec` and `tdd` call `code-review`, which now resolves to that skill.
- `skills/misc/`: `setup-pre-commit` and `git-guardrails-claude-code` would install hooks alongside this repository's own pre-commit and pre-push hooks; `migrate-to-shoehorn` and `scaffold-exercises` are specific to the author's course tooling.
- `skills/in-progress/`: marked unfinished upstream.

The `agents/openai.yaml` files are Codex metadata; Claude Code ignores them.

## Setup

Run `/setup-matt-pocock-skills` once before using `to-tickets`, `triage`, `to-spec` or `wayfinder`. It asks where issues are tracked and how domain docs are laid out, writes `docs/agents/`, and adds a section to `CLAUDE.md`. Review its `CLAUDE.md` change against this repository's rules before committing it.

## Updating

Re-copy the folders from a newer upstream commit and update the commit hash above. Local edits are overwritten, so re-apply them afterwards.
