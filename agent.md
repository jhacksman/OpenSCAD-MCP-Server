# Agent guide

<!-- portfolio-guide:start -->
## Shared tools for repository documentation

Use these for work on this repository for Jack. Keep local project rules in effect.

- **Find context and track confirmed work.** Open the private [knowledge wiki](https://github.com/jhacksman/jhacksman-knowledge-wiki), read its `AGENTS.md`, `_index.md` and `_schema.md`, and check material facts at their sources. At start, status change, handoff and completion, update the same `work/items/<id>.json`. Include this repo, task, owner, date, status, next step and dated evidence. Use notes for the execution/target machines, owner decisions, and any handoff recipient. Record the exact blocker if blocked. On completion link the result and set the next step to `none`. From the wiki root run `python3 scripts/sync_github.py --offline` and then `python3 scripts/sync_github.py --offline --check`. Commit the record and generated views together. Coordinate concurrent wiki edits through a branch and PR.
- **Fix repeated documentation friction.** Use [Code Gardener](https://github.com/jhacksman/code-gardener/blob/main/PRACTICE.md) to identify a recurring setup or handoff problem, fix its source example, and verify it on the next use. The [delegation worksheet](https://github.com/jhacksman/code-gardener/blob/main/DELEGATION.md) helps record what an agent completed independently and which decision needed a person.
- **Bank an idea when asked.** Use the [idea handoff brief](https://github.com/jhacksman/code-gardener/blob/main/IDEA_HANDOFF.md) to name the reader, smallest trial and evidence of success. Keep the brief in its owning project at the intended visibility. Link to it from other records.
- **Hand off between agents.** In the owner's private coordination vault, use the [Second Brain protocol](https://github.com/jhacksman/second-brain/blob/main/Shared/PROTOCOL.md). Give the recipient a link to the work record, the next action and the evidence needed to finish. Keep the task status in the wiki.
- **Keep one source for each fact.** Use the private [routing guide](https://github.com/jhacksman/jhacksman-knowledge-wiki/blob/main/projects/meta-tools.md) for durable project context and handoffs. Keep credentials and private notes out of public files. If private tools are unavailable, retain a local pending note, identify the access failure and continue authorized project work.
<!-- portfolio-guide:end -->
