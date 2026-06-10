param(
    [string]$Python = "python",
    [string]$CloudflaredPath = "",
    [string]$VercelPublicUrl = "https://vercel-proxy-nine-beta.vercel.app",
    [switch]$SkipBackendRestart,
    [switch]$ReuseTunnel,
    [switch]$RunChatSmoke,
    [switch]$NoDeploy,
    [int]$TunnelTimeoutSeconds = 60
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$LogDir = Join-Path $RepoRoot "logs\tablet-server"
$VercelDir = Join-Path $RepoRoot "vercel-proxy"
$ServiceScript = Join-Path $PSScriptRoot "start_tablet_server.ps1"
$PublicUrlFile = Join-Path $LogDir "public_url.txt"
$TunnelPidFile = Join-Path $LogDir "cloudflared.pid"
$TunnelStdout = Join-Path $LogDir "cloudflared.out.log"
$TunnelStderr = Join-Path $LogDir "cloudflared.err.log"
$DeployLog = Join-Path $LogDir "vercel_deploy.log"
$StatusFile = Join-Path $LogDir "deploy_status.txt"
$TunnelTarget = "http://127.0.0.1:8000"
$TabletPath = "/tablet_legacy.html?singleOrigin=1&publicMode=1"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host ("==> {0}" -f $Message)
}

function Invoke-External {
    param(
        [string]$FilePath,
        [string[]]$Arguments,
        [string]$WorkingDirectory,
        [switch]$AllowFailure
    )

    Push-Location $WorkingDirectory
    $oldErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $output = @(& $FilePath @Arguments 2>&1)
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $oldErrorActionPreference
        Pop-Location
    }

    if ($exitCode -ne 0 -and -not $AllowFailure) {
        Write-Host ($output -join "`n")
        throw ("Command failed ({0}): {1} {2}" -f $exitCode, $FilePath, ($Arguments -join " "))
    }
    return [pscustomobject]@{
        ExitCode = $exitCode
        Output = $output
    }
}

function Invoke-Foreground {
    param(
        [string]$FilePath,
        [string[]]$Arguments,
        [string]$WorkingDirectory
    )

    Push-Location $WorkingDirectory
    $oldErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $FilePath @Arguments
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $oldErrorActionPreference
        Pop-Location
    }

    if ($exitCode -ne 0) {
        throw ("Command failed ({0}): {1} {2}" -f $exitCode, $FilePath, ($Arguments -join " "))
    }
}

function Test-HttpOk {
    param(
        [string]$Url,
        [int]$TimeoutSec = 30
    )

    try {
        $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec $TimeoutSec
        return ($response.StatusCode -ge 200 -and $response.StatusCode -lt 300)
    } catch {
        return $false
    }
}

function Wait-HttpOk {
    param(
        [string]$Url,
        [int]$TimeoutSec = 60
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    do {
        if (Test-HttpOk -Url $Url -TimeoutSec 15) {
            return
        }
        Start-Sleep -Seconds 2
    } while ((Get-Date) -lt $deadline)

    throw "Timed out waiting for HTTP 2xx: $Url"
}

function Get-CloudflaredExecutable {
    if ($CloudflaredPath.Trim()) {
        if (Test-Path $CloudflaredPath) {
            return (Resolve-Path $CloudflaredPath).Path
        }
        throw "CloudflaredPath does not exist: $CloudflaredPath"
    }

    if ($env:CLOUDFLARED_PATH -and (Test-Path $env:CLOUDFLARED_PATH)) {
        return (Resolve-Path $env:CLOUDFLARED_PATH).Path
    }

    $command = Get-Command cloudflared -ErrorAction SilentlyContinue
    if ($command -and $command.Source) {
        return $command.Source
    }

    $running = Get-CimInstance Win32_Process -Filter "name='cloudflared.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match [regex]::Escape($TunnelTarget) } |
        Select-Object -First 1
    if ($running -and $running.ExecutablePath -and (Test-Path $running.ExecutablePath)) {
        return $running.ExecutablePath
    }

    $knownPaths = @(
        "D:\xuchengwei\vocal_server\tools\cloudflared.exe",
        (Join-Path $RepoRoot "tools\cloudflared.exe")
    )
    foreach ($path in $knownPaths) {
        if (Test-Path $path) {
            return (Resolve-Path $path).Path
        }
    }

    throw "cloudflared.exe was not found. Set CLOUDFLARED_PATH or pass -CloudflaredPath."
}

function Stop-ExistingTabletTunnel {
    $escapedTarget = [regex]::Escape($TunnelTarget)
    $processes = Get-CimInstance Win32_Process -Filter "name='cloudflared.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match "tunnel" -and $_.CommandLine -match $escapedTarget }

    foreach ($process in $processes) {
        Write-Host ("Stopping old tablet tunnel pid={0}" -f $process.ProcessId)
        Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
    }

    Remove-Item -LiteralPath $TunnelPidFile -ErrorAction SilentlyContinue
}

function Get-TunnelUrlFromLogs {
    $pattern = "https://[-a-z0-9]+\.trycloudflare\.com"
    foreach ($file in @($TunnelStdout, $TunnelStderr)) {
        if (-not (Test-Path $file)) {
            continue
        }
        $matches = Select-String -Path $file -Pattern $pattern -AllMatches -ErrorAction SilentlyContinue
        if ($matches) {
            $last = $matches | Select-Object -Last 1
            return $last.Matches[$last.Matches.Count - 1].Value.TrimEnd("/")
        }
    }
    return ""
}

function Start-NewTunnel {
    $cloudflared = Get-CloudflaredExecutable

    Remove-Item -LiteralPath $TunnelStdout, $TunnelStderr -ErrorAction SilentlyContinue

    $process = Start-Process `
        -FilePath $cloudflared `
        -ArgumentList @("tunnel", "--url", $TunnelTarget, "--no-autoupdate", "--protocol", "http2") `
        -WorkingDirectory $RepoRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $TunnelStdout `
        -RedirectStandardError $TunnelStderr `
        -PassThru

    Set-Content -Path $TunnelPidFile -Value $process.Id
    Write-Host ("Started cloudflared pid={0}" -f $process.Id)

    $deadline = (Get-Date).AddSeconds($TunnelTimeoutSeconds)
    do {
        $process.Refresh()
        if ($process.HasExited) {
            $log = @()
            if (Test-Path $TunnelStdout) { $log += Get-Content $TunnelStdout -Tail 40 }
            if (Test-Path $TunnelStderr) { $log += Get-Content $TunnelStderr -Tail 40 }
            throw ("cloudflared exited before publishing a URL.`n{0}" -f ($log -join "`n"))
        }

        $url = Get-TunnelUrlFromLogs
        if ($url) {
            Set-Content -Path $PublicUrlFile -Value $url
            return $url
        }
        Start-Sleep -Seconds 1
    } while ((Get-Date) -lt $deadline)

    throw "Timed out waiting for cloudflared public URL. See $TunnelStdout and $TunnelStderr"
}

function Get-OrStartTunnel {
    if ($ReuseTunnel -and (Test-Path $PublicUrlFile)) {
        $existingUrl = (Get-Content $PublicUrlFile -ErrorAction SilentlyContinue | Select-Object -First 1).Trim()
        if ($existingUrl -and (Test-HttpOk -Url ($existingUrl + $TabletPath) -TimeoutSec 20)) {
            Write-Host ("Reusing live tunnel: {0}" -f $existingUrl)
            return $existingUrl.TrimEnd("/")
        }
    }

    Stop-ExistingTabletTunnel
    return Start-NewTunnel
}

function Set-VercelEnv {
    param(
        [string]$Target,
        [string]$BackendUrl
    )

    Write-Host ("Updating Vercel BACKEND_BASE_URL for {0}" -f $Target)
    [void](Invoke-External -FilePath "npx" -Arguments @("--yes", "vercel", "env", "add", "BACKEND_BASE_URL", $Target, "--value", $BackendUrl, "--yes", "--force") -WorkingDirectory $VercelDir)
}

function Deploy-VercelProxy {
    $result = Invoke-External -FilePath "npx" -Arguments @("--yes", "vercel", "deploy", "--prod", "--yes") -WorkingDirectory $VercelDir
    $result.Output | Set-Content -Path $DeployLog

    $aliased = $result.Output |
        Select-String -Pattern "Aliased\s+(https://\S+?\.vercel\.app)" |
        Select-Object -Last 1
    if ($aliased) {
        return $aliased.Matches[0].Groups[1].Value.TrimEnd("/")
    }
    return $VercelPublicUrl.TrimEnd("/")
}

function Test-VercelProxy {
    param([string]$PublicUrl)

    $pageUrl = $PublicUrl.TrimEnd("/") + $TabletPath
    Wait-HttpOk -Url $pageUrl -TimeoutSec 90

    $randomWav = $PublicUrl.TrimEnd("/") + "/api/tts-total/random-wav"
    Wait-HttpOk -Url $randomWav -TimeoutSec 90

    if ($RunChatSmoke) {
        $tmp = Join-Path $env:TEMP "xiaosita-chat-smoke.json"
        [System.IO.File]::WriteAllText($tmp, '{"message":"你好","text":"你好"}', [System.Text.Encoding]::UTF8)
        $output = @(& curl.exe -s -X POST ($PublicUrl.TrimEnd("/") + "/api/chat") -H "Content-Type: application/json; charset=utf-8" --data-binary "@$tmp" 2>&1)
        if ($LASTEXITCODE -ne 0) {
            throw ("Chat smoke request failed: {0}" -f ($output -join "`n"))
        }
        $json = ($output -join "`n") | ConvertFrom-Json
        if (-not $json.text) {
            throw ("Chat smoke response did not contain text: {0}" -f ($output -join "`n"))
        }
    }
}

Write-Step "Starting local backend services"
if ($SkipBackendRestart) {
    Wait-HttpOk -Url "http://127.0.0.1:8000/tablet_legacy.html" -TimeoutSec 20
} else {
    Invoke-Foreground -FilePath "powershell" -Arguments @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $ServiceScript, "-Python", $Python) -WorkingDirectory $RepoRoot
}

Write-Step "Starting public Cloudflare tunnel"
$backendUrl = Get-OrStartTunnel
Wait-HttpOk -Url ($backendUrl + $TabletPath) -TimeoutSec 90
Write-Host ("Backend public URL: {0}" -f $backendUrl)

if (-not $NoDeploy) {
    Write-Step "Updating Vercel backend URL"
    Set-VercelEnv -Target "production" -BackendUrl $backendUrl
    Set-VercelEnv -Target "development" -BackendUrl $backendUrl

    Write-Step "Deploying Vercel proxy"
    $VercelPublicUrl = Deploy-VercelProxy
}

Write-Step "Verifying public tablet URL"
Test-VercelProxy -PublicUrl $VercelPublicUrl

Write-Host ""
Write-Host "Done."
Write-Host ("Backend tunnel: {0}" -f $backendUrl)
Write-Host ("Tablet public URL: {0}{1}" -f $VercelPublicUrl.TrimEnd("/"), $TabletPath)
Write-Host ("Logs: {0}" -f $LogDir)

@(
    "updated_at=$(Get-Date -Format o)",
    "backend_tunnel=$backendUrl",
    "tablet_public_url=$($VercelPublicUrl.TrimEnd('/'))$TabletPath",
    "logs=$LogDir"
) | Set-Content -Path $StatusFile
