# start_demo.ps1
# One command to run the offline CPU demo of the deployed model.
# Starts the local llama.cpp server, waits until it is healthy, launches the demo,
# and stops the server when the demo exits. No second terminal, no manual steps.
#
#   .\start_demo.ps1              speak to it (Ryan voice replies)
#   .\start_demo.ps1 -Text        type instead of speaking
#   .\start_demo.ps1 -Text -NoTts type, no audio out (quietest, most reliable)
#   .\start_demo.ps1 -Model llama3 -Quant Q4_K_M   use a different model

param(
    [string]$Model = "qwen0.5b",
    [string]$Quant = "Q4_K_M",
    [switch]$Text,
    [switch]$NoTts
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "Starting $Model ($Quant) server..."
$server = Start-Process -FilePath "python" `
    -ArgumentList @("scripts/03_cpu_server.py", "--model", $Model, "--quant", $Quant) `
    -PassThru -WindowStyle Minimized

$healthy = $false
for ($i = 0; $i -lt 60; $i++) {
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:8080/health" -TimeoutSec 2 -UseBasicParsing
        if ($r.Content -match "ok") { $healthy = $true; break }
    } catch { }
    Start-Sleep -Seconds 1
}

if (-not $healthy) {
    Write-Host "Server did not become healthy in time. Stopping."
    Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue
    Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    exit 1
}

Write-Host "Server ready. Launching demo. Type 'quit' in the demo to finish."

$demoArgs = @("demo.py", "--llm", "cpu", "--family", $Model)
if ($Text)  { $demoArgs += "--text" }
if ($NoTts) { $demoArgs += "--no-tts" }

try {
    & python @demoArgs
}
finally {
    Write-Host "Stopping server..."
    # Kill the wrapper first so it cannot respawn the child, then the server itself.
    Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue
    Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Write-Host "Done."
}
