# GitHub Actions deployment to the Windows laptop

Prepared 10 November 2026 for the existing CPU FastAPI application and Docker Desktop.

## What happens after a push

1. A GitHub-hosted Ubuntu runner installs Python 3.12 and the locked dependencies with uv.
2. It checks model integrity, runs Ruff and the API tests, builds the Linux/AMD64 Docker image, then checks health, Swagger, prediction and upload errors against that container.
3. For a branch push or manual run, it publishes the **tested image** to `ghcr.io/OWNER/REPOSITORY:sha-COMMIT`. Pull requests only test; they do not publish or deploy.
4. When deployment is enabled and the run is on your configured deployment branch, your Windows self-hosted runner pulls that image **by digest**. Docker Compose starts it, then the deployment script checks model readiness, the expected model hash, Swagger and a real PNG prediction.

Your laptop does not train the model or rebuild the image. The packaged model and vendor code are already in the image. A failed image pull leaves the running service untouched. If replacing an existing deployment fails verification, the script attempts to restore its previous image and port and still reports the workflow as failed. A failed first deployment has no previous image to restore.

The laptop is the host: it must remain awake with Docker Desktop and the runner running. The endpoint is bound to **127.0.0.1**, so it is available on that laptop. This setup does not create an internet URL. Swagger will be `http://127.0.0.1:8000/docs`, or the port you configure.

## Files to push

Push from the repository root, including:

- `.github/workflows/api.yml`
- `deploy/compose.laptop.yaml` and `deploy/deploy-laptop.ps1`
- The existing `Dockerfile`, `.dockerignore`, `.gitignore`, `.python-version`, `pyproject.toml` and `uv.lock`
- `app/`, `vendor/`, `scripts/`, `tests/`
- **`signature_yolox/api_model/model.pth` and `metadata.json`**

Only the serving model folder is needed from `signature_yolox`; the existing ignore exceptions already allow its two files. The notebook, datasets and training outputs are not CI/deployment inputs. Check that both serving files appear in GitHub's Code view after pushing. A Git LFS pointer or missing model will fail the integrity check.

The workflow lives at the repository root, not inside the earlier `temp` export. No runner token, personal access token, `.env` or Docker login file belongs in the repository.

## Browser setup after your first push

### 1. Check the workflow

Open your repository on GitHub and select **Actions → API CI and laptop deployment**. Your first relevant branch push starts the workflow automatically. **Test, build and publish** should run on GitHub. **Deploy to Windows laptop** is skipped until the variables below are set.

If Actions is disabled, open **Settings → Actions → General** and enable the actions used by this repository under your account or organization's policy. The workflow declares its own job permissions: package write access for publishing and package read access for deployment. It uses GitHub's automatically supplied `GITHUB_TOKEN`; no manually created registry token or repository secret is required for a package associated with this repository. See GitHub's [container publishing guide](https://docs.github.com/en/actions/tutorials/publish-packages/publish-docker-images).

### 2. Register the laptop runner

In the browser, open **Settings → Actions → Runners → New self-hosted runner**, then select **Windows** and **x64**. GitHub displays the download, extraction and configuration commands for your repository. Run those commands in a normal PowerShell window on your laptop, in a separate folder such as `C:\actions-runner-signature`—outside this project and its `temp` export. The page's registration token is temporary; use the generated commands rather than saving the token in the code. [GitHub runner instructions](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/add-runners).

During configuration:

- Give it an identifiable name, such as `signature-laptop`.
- Add the custom label **`signature-laptop`**. Keep the automatic `self-hosted`, `Windows` and `X64` labels.
- For this Docker Desktop setup, decline installation as a Windows service and run it interactively under the same Windows user that runs Docker Desktop.
- Start Docker Desktop with **Linux containers**, then start the runner using the generated `run.cmd` instruction. Keep that PowerShell window open.

Refresh **Settings → Actions → Runners** and confirm the runner is **Idle/online** with the required labels. The one-time runner installation requires PowerShell commands; it cannot be completed entirely in the browser.

A self-hosted runner executes repository workflow code on your laptop. Use a repository and contributors you trust; GitHub [recommends private repositories for self-hosted runners](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/add-runners). This workflow keeps pull-request jobs on GitHub-hosted runners, but that does not protect against someone authorized to change your workflow itself.

### 3. Add repository variables

Open **Settings → Secrets and variables → Actions → Variables → New repository variable**. Add these as **variables**, not secrets:

| Name | Value |
| --- | --- |
| `DEPLOY_BRANCH` | The exact branch you push for deployment, for example `yolox-tiny`. Do not include `refs/heads/`. |
| `DEPLOY_ENABLED` | `true`, once the runner is online and Docker Desktop is ready. Any other value or no value skips deployment. |
| `DEPLOY_PORT` | Optional; default `8000`. Use `8001` if your manually started local API already occupies port 8000. |
| `DOCKER_EXE` | Optional full path to `docker.exe` if the runner cannot find Docker. Most installs work through PATH or the script's Docker Desktop fallback paths. |

GitHub documents repository-variable setup under [configuration variables](https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/use-variables).

Do not change package visibility to make this work. The deployment job uses its short-lived package-read token. If an existing GHCR package is not connected to this repository, check the package's **Package settings → Manage Actions access** and grant this repository the required access; see [package Actions access](https://docs.github.com/en/packages/learn-github-packages/configuring-a-packages-access-control-and-visibility).

### 4. Trigger deployment

After setup, push a change to an API, workflow or deployment file on `DEPLOY_BRANCH`. The push runs CI, publishes the tested image and starts the laptop deployment automatically. Documentation-only and notebook-only changes do not trigger this workflow.

You can also open an existing run in **Actions**, select **Re-run jobs → Re-run all jobs** after configuring the variables. Check the new run's deploy job; if it remains skipped, verify the exact variable values and branch and trigger a new relevant push. [GitHub rerun instructions](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/re-run-workflows-and-jobs).

For a fresh manual run, choose **Actions → API CI and laptop deployment → Run workflow**, select `DEPLOY_BRANCH`, and run it. **The Run workflow button requires this workflow file to also exist on the repository's default branch.** If you only pushed it to a feature branch, use the automatic push run or rerun until it is on the default branch. [GitHub manual-run requirements](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow).

### 5. Open the deployed API

When both jobs succeed, open **http://127.0.0.1:8000/docs** on your laptop, adjusting the port if you set `DEPLOY_PORT`. Upload a document through `POST /predict`; the response is the annotated PNG. The Actions run summary records the deployed image digest and local URL.

The Compose project is **`signature-api-laptop`**, visible in Docker Desktop. It is separate from the original manually started Compose project. If another application occupies the selected port, choose a free `DEPLOY_PORT` or stop that application's container yourself; this script does not stop unrelated services. Container updates briefly interrupt this single-instance API.

## Later updates and common problems

After setup, pushing relevant code to the configured branch repeats the same pipeline. Setting `DEPLOY_ENABLED` to `false` prevents future deployment jobs; it does not stop the currently running container. `restart: unless-stopped` allows that container to restart when Docker Desktop's engine starts again; it does not start Docker Desktop or wake the laptop.

| What you see | Check |
| --- | --- |
| Deploy is skipped | `DEPLOY_ENABLED` must be exactly `true`; `DEPLOY_BRANCH` must match the run's branch; pull requests never deploy. |
| Deploy waits for a runner | Start `run.cmd`, confirm the laptop is awake, and check all four runner labels. |
| Docker not found or cannot connect | Docker Desktop must run under the runner's Windows user, using Linux containers. Set `DOCKER_EXE` if needed and restart the runner after PATH changes. |
| Port is already allocated | Set a free `DEPLOY_PORT`, for example `8001`; open that same port in the browser. |
| Model check fails | Confirm both packaged serving files were pushed and the model is the real binary matching its metadata. |
| Registry push/pull denied | Check workflow/package policy and repository access to the GHCR package. |
| Deploy fails after replacement | Read the deployment log for its health check and rollback result. A rollback attempt does not make the failed release successful. |
| Swagger works only on the laptop | Expected: the published address is loopback. A public hosting/tunnel setup is a separate step. |

## Validation boundary

Local validation on **10 November 2026** passed:

- All **16 API tests**, using the saved model.
- Ruff lint and formatting checks.
- Workflow and Compose YAML parsing, pinned action references, deployment-branch guards, test-before-publish ordering, model integrity and documentation links.
- Windows PowerShell syntax parsing.
- The deployment script's actual HTTP verification function against a temporary local API: readiness, expected model hash, Swagger HTML and PNG prediction. The temporary server was stopped after the check.

These files prepare the pipeline; creating them does not register a runner, trigger a GitHub run, publish an image or deploy a Docker service. The Linux image build, GHCR push/pull and real runner/Docker Desktop deployment remain to be verified by the first GitHub run after your push and setup. Docker replacement/rollback was not exercised locally during this change.
