# Agent instructions

## Cross-repository work tracking

For work on this repository for Jack, maintain the private [jhacksman knowledge wiki](https://github.com/jhacksman/jhacksman-knowledge-wiki). Read its `AGENTS.md`, `_index.md`, and `_schema.md` before updating it; re-check task-critical facts at their sources.

- At task start, when blocked, at handoff, and at completion, create or update the task's single `work/items/<id>.json` record. Reuse its stable ID rather than creating duplicate records.
- Follow the existing schema: `id`, `title`, `repos`, `status`, `owner`, `updated_on`, `next_step`, and `evidence`. Include this repository in `repos`, identify the project in `title`, and use `active`, `blocked`, `paused`, or `done` accurately.
- In the existing optional `notes` field, record the execution machine (and any separate target machine) and decisions needed from Jack, explicitly saying `none` when there are none. Use `blocker` for the exact obstacle when blocked. At handoff, record the recipient and concrete next action; at completion, link the result and set `next_step` to `none`.
- Support the record with dated, direct evidence links. A push or open PR alone does not establish that a task is active. Keep secrets and unnecessary personal data out of the wiki and public repository.
- From the wiki root run `python3 scripts/sync_github.py --offline`, then `python3 scripts/sync_github.py --offline --check`. Commit the work record and generated views together. Use a branch and PR for concurrent wiki edits; reconcile generated-view conflicts by rerunning the renderer after merging records.
- If wiki access or publication is blocked, report the exact failure and retain the pending update in the task's working artifacts for handoff. Do not claim the wiki was updated, invent a parallel tracking schema, or let the tracking failure stop independent authorized work.

Keep repository-specific instructions in effect alongside this tracking requirement.
