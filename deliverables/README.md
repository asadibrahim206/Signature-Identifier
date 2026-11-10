# Submission deliverables

Prepared **9 November 2026**. Each pending task has its own folder.

| Task | Deliverable | State |
| --- | --- | --- |
| 1–2 slide deck | [slide_deck](slide_deck/README.md): exactly two editable slides plus PDF | Complete |
| Architecture diagram | [architecture_diagram](architecture_diagram/README.md): SVG, PNG, DOT and Mermaid | Complete |
| Jira board | [jira_board](jira_board/README.md): live Kanban board, 20 tasks, offline snapshot, key mapping and archived CSV | Board and filter verified; 19 Done, SCRUM-24 To Do; column configuration needs correction |
| Working demo | Existing FastAPI/Docker application; [testing instructions](../docs/README.md) | Previously implemented and verified; not rerun for this documentation task |
| FastAPI Swagger | `http://127.0.0.1:8000/docs` after starting the service | Previously implemented and HTTP-verified |

The [Signature Identifier Kanban board](https://bokhariadil.atlassian.net/secure/RapidBoard.jspa?rapidView=2) now exists as board ID 2, using saved filter 10001. It returns all 20 signature-project tasks in My Software Team (SCRUM), SCRUM-5 through SCRUM-24. Nineteen are Done; SCRUM-24 remains To Do. The In Review status currently maps to In Progress, and an empty In Review column follows Done; the remaining setup is to correct that mapping and make Done the final column. See [Jira results and mapping](jira_board/README.md). The CSV is an archive: do not re-import.

The deck and diagram use saved project outputs. Test metrics are not a new training run; API verification evidence dates from 5 November 2026. No Git commands or pushes were performed for this task.

See each folder's README for usage, evidence and limitations.
