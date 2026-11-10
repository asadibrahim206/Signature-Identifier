# Jira task import and board setup

Verified **9 November 2026** through the connected Atlassian integration at https://bokhariadil.atlassian.net, project **My Software Team (SCRUM)**.

**The [Signature Identifier Kanban board is live](https://bokhariadil.atlassian.net/secure/RapidBoard.jspa?rapidView=2).** Board ID **2** uses saved filter **10001**, also named Signature Identifier. Its location is the owner's personal board location; the issues remain in My Software Team (SCRUM).

All 20 tasks exist as SCRUM-5 through SCRUM-24 and are returned by the named board's API. SCRUM-5 through SCRUM-23 are Done. [SCRUM-24](https://bokhariadil.atlassian.net/browse/SCRUM-24), Create the live Jira board, remains To Do because column configuration still needs correction. These values were rechecked through the connected integration on 9 November 2026.

Before creation, all four existing SCRUM issues were compared by summary and description with the backlog titles and SD references. None matched. A site-wide label search returned no matches. The visible board list contained only SCRUM board (ID 1), and saved-filter search returned no visible filters. Unrelated issues SCRUM-1 through SCRUM-4 and the existing board were not modified.

## Verified import

Every task preserves its local reference, workstream, acceptance criteria and evidence, uses Task, and has label `signature-identifier`. Medium and High were available and used as proposed. All assignees remain unassigned. No historical completion dates or Jira events were invented: Jira creation and Done transitions occurred during import on 9 November 2026. Earlier API verification remains dated 5 November 2026.

The original team-managed SCRUM board (ID 1) is separate from the requested named Kanban board (ID 2). Verification of board 2 confirms the saved filter, its column mappings and all 20 imported task records. Interactive UI visibility and completed-task retention have not been verified in a browser during this documentation update.

## Remaining board setup

The board and filter have already been created. Use the existing filter rather than creating or importing them again:

```jql
project = "SCRUM" AND labels = "signature-identifier" ORDER BY Rank ASC
```

Current board configuration, verified through Jira:

| Column, left to right | Mapped statuses |
| --- | --- |
| Backlog | None |
| To Do | To Do |
| In Progress | In Progress, In Review |
| Done | Done |
| In Review | None |

Move the existing In Review status into the In Review column and position that column before Done. Keep Done as the final column. The intended working sequence is **To Do → In Progress → In Review → Done**. Jira's [column configuration guide](https://support.atlassian.com/jira-software-cloud/docs/configure-columns/) explains the mapping and final-column behavior.

After correcting the columns, verify task visibility and completed-task retention in the browser, then complete SCRUM-24 in Jira. The task remains To Do in this local snapshot because that is its verified live status. This documentation update does not change Jira configuration or issue statuses.

## Showing the board

Open the live board link above and sign in. During the project demonstration, share that browser window, show the workflow columns, and open a task such as [Dockerize the prediction service](https://bokhariadil.atlassian.net/browse/SCRUM-17) to show its completion criteria and evidence. For a submission, provide the board URL and a screenshot after correcting the columns. Reviewers need access to the Jira project and board filter.

The board tracks project work, including dataset preparation, training, evaluation, API implementation and Docker deployment. Most tasks were imported after development, so it serves as a retrospective record of completed work and remaining setup.

## Files

- `board.html`: offline snapshot grouped by issue status, with the live board URL and real Jira issue links; it does not update itself from Jira.
- `backlog.json`: imported backlog, actual keys, verified statuses, board/filter IDs and remaining configuration notes.
- `jira_mapping.json`: SD reference to real issue-key mapping.
- `jira_verification.json`: duplicate-check results, verified import and concrete board blockers.
- **`jira_import.csv`: archive after API import. DO NOT RE-IMPORT.** Re-importing can create duplicates. The archive includes real keys and an explicit notice on every row.
- `HANDOFF.md`: remaining work only; do not recreate the tasks.

## Issue mapping


| Local reference | Jira task | Status |
| --- | --- | --- |
| SD001 | [SCRUM-5](https://bokhariadil.atlassian.net/browse/SCRUM-5) - Audit three local datasets | Done |
| SD002 | [SCRUM-6](https://bokhariadil.atlassian.net/browse/SCRUM-6) - Clean images and bounding boxes | Done |
| SD003 | [SCRUM-7](https://bokhariadil.atlassian.net/browse/SCRUM-7) - Create related-image grouped splits | Done |
| SD004 | [SCRUM-8](https://bokhariadil.atlassian.net/browse/SCRUM-8) - Configure single-class YOLOX-Tiny | Done |
| SD005 | [SCRUM-9](https://bokhariadil.atlassian.net/browse/SCRUM-9) - Optimize training for the RTX 3060 laptop | Done |
| SD006 | [SCRUM-10](https://bokhariadil.atlassian.net/browse/SCRUM-10) - Complete training and select the checkpoint | Done |
| SD007 | [SCRUM-11](https://bokhariadil.atlassian.net/browse/SCRUM-11) - Evaluate the untouched test split | Done |
| SD008 | [SCRUM-12](https://bokhariadil.atlassian.net/browse/SCRUM-12) - Document data, training and results | Done |
| SD009 | [SCRUM-13](https://bokhariadil.atlassian.net/browse/SCRUM-13) - Package the serving model | Done |
| SD010 | [SCRUM-14](https://bokhariadil.atlassian.net/browse/SCRUM-14) - Implement annotated-image prediction | Done |
| SD011 | [SCRUM-15](https://bokhariadil.atlassian.net/browse/SCRUM-15) - Expose Swagger and health endpoints | Done |
| SD012 | [SCRUM-16](https://bokhariadil.atlassian.net/browse/SCRUM-16) - Validate and bound uploaded images | Done |
| SD013 | [SCRUM-17](https://bokhariadil.atlassian.net/browse/SCRUM-17) - Dockerize the prediction service | Done |
| SD014 | [SCRUM-18](https://bokhariadil.atlassian.net/browse/SCRUM-18) - Run API regression checks | Done |
| SD015 | [SCRUM-19](https://bokhariadil.atlassian.net/browse/SCRUM-19) - Verify the packaged Docker model | Done |
| SD016 | [SCRUM-20](https://bokhariadil.atlassian.net/browse/SCRUM-20) - Demonstrate held-out image prediction | Done |
| SD017 | [SCRUM-21](https://bokhariadil.atlassian.net/browse/SCRUM-21) - Document API setup and testing | Done |
| SD018 | [SCRUM-22](https://bokhariadil.atlassian.net/browse/SCRUM-22) - Prepare the two-slide presentation | Done |
| SD019 | [SCRUM-23](https://bokhariadil.atlassian.net/browse/SCRUM-23) - Prepare the architecture diagram | Done |
| SD020 | [SCRUM-24](https://bokhariadil.atlassian.net/browse/SCRUM-24) - Create the live Jira board | To Do |

No Git commands or pushes were performed.
