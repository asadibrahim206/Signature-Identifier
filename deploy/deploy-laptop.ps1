[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^ghcr\.io/[a-z0-9._/-]+@sha256:[a-f0-9]{64}$')]
    [string]$Image,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[a-f0-9]{64}$')]
    [string]$ExpectedModelSha256,

    [ValidateRange(1, 65535)]
    [int]$Port = 8000
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$projectName = 'signature-api-laptop'
$composeFile = Join-Path $PSScriptRoot 'compose.laptop.yaml'
$composeArgs = @('compose', '--project-name', $projectName, '--file', $composeFile)

# Docker Desktop may be installed for this user or for all users.
$dockerCommand = Get-Command docker -ErrorAction SilentlyContinue
if ($env:DOCKER_EXE) {
    $script:DockerExecutable = (Get-Item -LiteralPath $env:DOCKER_EXE).FullName
} elseif ($dockerCommand) {
    $script:DockerExecutable = $dockerCommand.Source
} else {
    $candidates = @(
        (Join-Path $env:ProgramFiles 'Docker\Docker\resources\bin\docker.exe'),
        (Join-Path $env:LOCALAPPDATA 'Programs\DockerDesktop\resources\bin\docker.exe')
    )
    $script:DockerExecutable = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $script:DockerExecutable) {
        throw 'Docker CLI not found. Add Docker to the runner PATH or set the DOCKER_EXE repository variable to docker.exe.'
    }
}

function Invoke-Docker {
    param([string[]]$DockerArguments)
    & $script:DockerExecutable @DockerArguments
    if ($LASTEXITCODE -ne 0) {
        throw ('Docker command failed (exit {0}): {1}' -f $LASTEXITCODE, ($DockerArguments -join ' '))
    }
}

function Confirm-Prediction {
    param([int]$ApiPort, [string]$ModelHash)
    $url = 'http://127.0.0.1:' + $ApiPort
    $health = Invoke-RestMethod -Uri ($url + '/health') -TimeoutSec 30
    if ($health.status -ne 'ready' -or $health.device -ne 'cpu') {
        throw 'The deployed API is not ready on CPU.'
    }
    if ($ModelHash -and $health.model_sha256 -ne $ModelHash) {
        throw 'The deployed model hash does not match the image tested by CI.'
    }
    $swagger = Invoke-WebRequest -UseBasicParsing -Uri ($url + '/docs') -TimeoutSec 30
    if ($swagger.Content -notmatch 'SwaggerUIBundle') { throw 'Swagger UI was not served.' }

    Add-Type -AssemblyName System.Net.Http
    $handler = New-Object System.Net.Http.HttpClientHandler
    $handler.UseProxy = $false
    $client = New-Object System.Net.Http.HttpClient($handler)
    $client.Timeout = [TimeSpan]::FromSeconds(120)
    $form = New-Object System.Net.Http.MultipartFormDataContent
    try {
        # A valid, white 1x1 PNG exercises real inference without uploading a personal document.
        $png = [Convert]::FromBase64String('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4//8/AAX+Av4N70a4AAAAAElFTkSuQmCC')
        $file = New-Object System.Net.Http.ByteArrayContent -ArgumentList (,$png)
        $file.Headers.ContentType = New-Object System.Net.Http.Headers.MediaTypeHeaderValue('image/png')
        $form.Add($file, 'file', 'deployment-check.png')
        $response = $client.PostAsync(($url + '/predict'), $form).GetAwaiter().GetResult()
        try {
            $null = $response.EnsureSuccessStatusCode()
            if ($response.Content.Headers.ContentType.MediaType -ne 'image/png') {
                throw 'Prediction did not return image/png.'
            }
            $bytes = $response.Content.ReadAsByteArrayAsync().GetAwaiter().GetResult()
            if ($bytes.Length -lt 24 -or [BitConverter]::ToString($bytes[0..7]) -ne '89-50-4E-47-0D-0A-1A-0A') {
                throw 'Prediction response is not a PNG.'
            }
            $detections = [int](@($response.Headers.GetValues('X-Detection-Count'))[0])
            if ($detections -lt 0) { throw 'Invalid detection count.' }
        } finally { $response.Dispose() }
    } finally {
        $form.Dispose()
        $client.Dispose()
        $handler.Dispose()
    }
    Write-Host ('Verified health, model hash, Swagger and PNG prediction: {0}/docs' -f $url)
}

$engine = (Invoke-Docker -DockerArguments @('info', '--format', '{{.OSType}}') | Out-String).Trim()
if ($engine -ne 'linux') { throw 'Start Docker Desktop using Linux containers, under the same Windows user as the runner.' }
$null = Invoke-Docker -DockerArguments @('compose', 'version')
if (-not $env:GHCR_USERNAME -or -not $env:GHCR_TOKEN) {
    throw 'GHCR_USERNAME and GHCR_TOKEN are required; the workflow supplies the short-lived GITHUB_TOKEN.'
}

$tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
if ($env:RUNNER_TEMP) { $tempRoot = [IO.Path]::GetFullPath($env:RUNNER_TEMP) }
$authDirectory = [IO.Path]::GetFullPath((Join-Path $tempRoot ('signature-registry-' + [guid]::NewGuid().ToString('N'))))
$tempPrefix = $tempRoot.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
if (-not $authDirectory.StartsWith($tempPrefix, [StringComparison]::OrdinalIgnoreCase)) { throw 'Invalid temporary registry directory.' }
$previousDockerConfig = $env:DOCKER_CONFIG
$previousImage = $null
$previousPort = $Port
$deploymentStarted = $false
New-Item -ItemType Directory -Path $authDirectory | Out-Null
$env:DOCKER_CONFIG = $authDirectory
$env:SIGNATURE_IMAGE = $Image
$env:PORT = [string]$Port

try {
    $env:GHCR_TOKEN | & $script:DockerExecutable login ghcr.io --username $env:GHCR_USERNAME --password-stdin
    if ($LASTEXITCODE -ne 0) { throw 'GHCR login failed. Check the package permission for this repository.' }
    $env:GHCR_TOKEN = ''

    # Pull first: a registry/network failure leaves the running service untouched.
    Invoke-Docker -DockerArguments @('pull', $Image)
    $ids = @(Invoke-Docker -DockerArguments ($composeArgs + @('ps', '--all', '--quiet', 'api')) | Where-Object { $_ })
    if ($ids.Count -gt 1) { throw 'Expected at most one existing deployment container.' }
    if ($ids.Count -eq 1) {
        $previous = (Invoke-Docker -DockerArguments @('inspect', $ids[0]) | Out-String) | ConvertFrom-Json
        $previousImage = $previous[0].Config.Image
        $bindings = $previous[0].HostConfig.PortBindings.'8000/tcp'
        if ($bindings) { $previousPort = [int]$bindings[0].HostPort }
    }

    $deploymentStarted = $true
    Invoke-Docker -DockerArguments ($composeArgs + @('up', '--detach', '--no-build', '--pull', 'never', '--wait', '--wait-timeout', '180'))
    Confirm-Prediction -ApiPort $Port -ModelHash $ExpectedModelSha256
    if ($env:GITHUB_STEP_SUMMARY) {
        @(
            '### Laptop deployment verified',
            ('Image: `{0}`' -f $Image),
            ('Local Swagger UI: http://127.0.0.1:{0}/docs' -f $Port),
            'Health, expected model hash, Swagger HTML and PNG prediction passed.'
        ) | Out-File -FilePath $env:GITHUB_STEP_SUMMARY -Encoding utf8 -Append
    }
} catch {
    $failure = $_
    if ($deploymentStarted) {
        try { Invoke-Docker -DockerArguments ($composeArgs + @('logs', '--tail', '100', 'api')) } catch { Write-Warning 'Could not read container logs.' }
    }
    if ($deploymentStarted -and $previousImage) {
        Write-Warning 'Deployment failed; restoring the previous container image and port.'
        $env:SIGNATURE_IMAGE = $previousImage
        $env:PORT = [string]$previousPort
        try {
            Invoke-Docker -DockerArguments ($composeArgs + @('up', '--detach', '--no-build', '--pull', 'never', '--wait', '--wait-timeout', '180'))
            Confirm-Prediction -ApiPort $previousPort -ModelHash ''
        } catch { Write-Warning ('Rollback also failed: ' + $_.Exception.Message) }
    }
    throw $failure
} finally {
    $env:GHCR_TOKEN = ''
    try { $null = Invoke-Docker -DockerArguments @('logout', 'ghcr.io') } catch { Write-Warning 'Could not log out of the temporary registry configuration.' }
    # Delete only our temporary auth file and then the directory if empty; never recurse.
    $authFile = Join-Path $authDirectory 'config.json'
    if (Test-Path -LiteralPath $authFile) { Remove-Item -LiteralPath $authFile -Force }
    if (@(Get-ChildItem -LiteralPath $authDirectory -Force).Count -eq 0) { Remove-Item -LiteralPath $authDirectory -Force }
    $env:DOCKER_CONFIG = $previousDockerConfig
}
