param(
    [string]$HostIp = "",
    [string]$Python = "python",
    [switch]$Stop
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$ProjectRoot = Join-Path $RepoRoot "oh-my-live2d-main (2)\oh-my-live2d-main"
$BackendRoot = Join-Path $ProjectRoot "backend"
$LogDir = Join-Path $RepoRoot "logs\tablet-server"
$PidFile = Join-Path $LogDir "pids.txt"
$ServicePorts = @(8000, 8001, 8002, 8003, 8004, 8005, 8006, 8010)

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Stop-PreviousServices {
    $pids = New-Object System.Collections.Generic.HashSet[int]

    if (Test-Path $PidFile) {
        Get-Content $PidFile | ForEach-Object {
            if ($_ -match "^\d+$") {
                [void]$pids.Add([int]$_)
            }
        }
    }

    $portPattern = ($ServicePorts | ForEach-Object { ":$_" }) -join "|"
    netstat -ano | Select-String -Pattern "LISTENING" | ForEach-Object {
        $line = $_.Line
        if ($line -notmatch $portPattern) {
            return
        }
        $columns = $line -split "\s+"
        if ($columns.Length -lt 5) {
            return
        }
        $localAddress = $columns[2]
        $pidText = $columns[-1]
        foreach ($port in $ServicePorts) {
            if ($localAddress -match ":$port$" -and $pidText -match "^\d+$") {
                [void]$pids.Add([int]$pidText)
            }
        }
    }

    foreach ($processId in $pids) {
        Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
    }

    foreach ($port in $ServicePorts) {
        $deadline = (Get-Date).AddSeconds(8)
        while ((Get-Date) -lt $deadline) {
            $stillListening = netstat -ano | Select-String -Pattern "LISTENING" | Where-Object { $_.Line -match ":$port\s" }
            if (-not $stillListening) {
                break
            }
            Start-Sleep -Milliseconds 250
        }
    }
    Remove-Item -Path $PidFile -ErrorAction SilentlyContinue
}

function Get-DefaultLanIp {
    if ($HostIp.Trim()) {
        return $HostIp.Trim()
    }

    $addresses = [System.Net.Dns]::GetHostAddresses([System.Net.Dns]::GetHostName()) |
        Where-Object {
            $_.AddressFamily -eq "InterNetwork" -and
            $_.IPAddressToString -notlike "127.*" -and
            $_.IPAddressToString -notlike "169.254.*"
        } |
        ForEach-Object { $_.IPAddressToString }

    if ($addresses) {
        return $addresses[0]
    }
    return "127.0.0.1"
}

function Test-EnvValue {
    param([string]$Name)

    if ([Environment]::GetEnvironmentVariable($Name)) {
        return $true
    }

    foreach ($envFile in @(
        Join-Path $RepoRoot ".env",
        Join-Path $ProjectRoot ".env",
        Join-Path $BackendRoot ".env"
    )) {
        if ((Test-Path $envFile) -and (Select-String -Path $envFile -Pattern "^\s*$Name\s*=" -Quiet)) {
            return $true
        }
    }
    return $false
}

function Start-ServerProcess {
    param(
        [string]$Name,
        [string]$WorkingDirectory,
        [string]$Command
    )

    $stdout = Join-Path $LogDir "$Name.out.log"
    $stderr = Join-Path $LogDir "$Name.err.log"
    $process = Start-Process `
        -FilePath "powershell" `
        -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $Command) `
        -WorkingDirectory $WorkingDirectory `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -PassThru

    Add-Content -Path $PidFile -Value $process.Id
    Write-Host ("started {0} pid={1}" -f $Name, $process.Id)
}

function Wait-ForPorts {
    param(
        [int[]]$Ports,
        [int]$TimeoutSeconds = 45
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $pending = New-Object System.Collections.Generic.HashSet[int]
    foreach ($port in $Ports) {
        [void]$pending.Add($port)
    }

    while ($pending.Count -gt 0 -and (Get-Date) -lt $deadline) {
        foreach ($port in @($pending)) {
            $connection = Test-NetConnection -ComputerName "127.0.0.1" -Port $port -WarningAction SilentlyContinue
            if ($connection.TcpTestSucceeded) {
                [void]$pending.Remove($port)
            }
        }
        if ($pending.Count -gt 0) {
            Start-Sleep -Seconds 1
        }
    }

    if ($pending.Count -gt 0) {
        throw "These ports did not start listening: $($pending -join ', ')"
    }
}

if ($Stop) {
    Stop-PreviousServices
    Write-Host "tablet server processes stopped"
    exit 0
}

Stop-PreviousServices

if (-not (Test-EnvValue "DASHSCOPE_API_KEY")) {
    Write-Warning "DASHSCOPE_API_KEY is not set. Chat, TTS, and ASR services will not work until you create .env or set the variable."
}

$LanIp = Get-DefaultLanIp

Start-ServerProcess "web-page-8000" $ProjectRoot "`$env:WEB_PORT='8000'; `$env:WEB_MODE='page'; & '$Python' 'mao_demo_server.py'"
Start-ServerProcess "live2d-assets-8010" $ProjectRoot "`$env:WEB_PORT='8010'; `$env:WEB_MODE='assets'; & '$Python' 'mao_demo_server.py'"
Start-ServerProcess "chat-page-8001" $ProjectRoot "`$env:WEB_PORT='8001'; `$env:WEB_MODE='page'; & '$Python' 'mao_demo_server.py'"
Start-ServerProcess "gateway-8002" $BackendRoot "& '$Python' 'main.py'"
Start-ServerProcess "chat-8003" $BackendRoot "& '$Python' 'qwen_chat_server.py'"
Start-ServerProcess "tts-8004" $BackendRoot "& '$Python' 'tts_ws_server.py'"
Start-ServerProcess "scoring-8005" $BackendRoot "& '$Python' 'asr_server.py'"
Start-ServerProcess "asr-8006" $BackendRoot "& '$Python' 'asr_new.py'"

Wait-ForPorts -Ports $ServicePorts

$TabletUrl = "http://$LanIp`:8000/tablet_legacy.html?singleOrigin=1&publicMode=1"
Write-Host ""
Write-Host "Tablet URL:"
Write-Host $TabletUrl
Write-Host ""
Write-Host "Logs:"
Write-Host $LogDir
Write-Host ""
Write-Host "Stop command:"
Write-Host "powershell -ExecutionPolicy Bypass -File scripts\start_tablet_server.ps1 -Stop"
