# Jira handoff: live board verified, column configuration remaining

Verified 9 November 2026. Do not run Git commands or push.

## Completed and verified

- Site: https://bokhariadil.atlassian.net; project My Software Team (SCRUM), ID 10000.
- Live **Signature Identifier** Kanban board: https://bokhariadil.atlassian.net/secure/RapidBoard.jspa?rapidView=2 (board ID 2).
- Saved filter: ID 10001, name Signature Identifier; JQL `project = "SCRUM" AND labels = "signature-identifier" ORDER BY Rank ASC`.
- Board location is personal/user; the task records remain in SCRUM. The original SCRUM board is ID 1.
- All 20 prepared tasks exist as SCRUM-5 through SCRUM-24. The named board's API returns all 20.
- SCRUM-5 through SCRUM-23 are Done; SCRUM-24 remains To Do.
- Local documentation, snapshots and archived CSV now record the real board and filter.

## Remaining configuration

Current columns, left to right: Backlog (no status), To Do (To Do), In Progress (In Progress and In Review), Done (Done), In Review (no status).

1. Move the existing In Review status from In Progress to the In Review column.
2. Place In Review before Done, leaving Done as the final column. Preserve the project's existing workflow and all tasks.
3. Verify task visibility and completed-task retention in the browser. Current verification is through the Jira API.
4. Once setup is verified, complete SCRUM-24 in Jira and refresh the local snapshot, mapping, archived CSV and README status counts.

Do not recreate the board, saved filter or imported tasks. `jira_import.csv` is an archive after API import: DO NOT RE-IMPORT.

Earlier installation, tool-loading and board-creation restrictions are historical context. The board and filter now exist.
This documentation update used read-only Jira checks and did not change remote configuration or issue statuses.
