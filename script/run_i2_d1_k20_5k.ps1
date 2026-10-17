param(
    [ValidateSet("All", "Control", "Release", "Reallocation")]
    [string]$Arm = "All",
    [switch]$Execute
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = "D:\Users\huashuo\anaconda3\envs\fsgs\python.exe"
$armValue = $Arm.ToLowerInvariant()
$arguments = @(
    "-u",
    "tools\run_i2_d1_k20_5k.py",
    "--arm",
    $armValue
)

if (-not $Execute) {
    $arguments += "--preflight-only"
    Write-Host "Preflight only. No GPU training will be started."
    Write-Host "After review, add -Execute to launch the selected arm(s)."
}

Push-Location $projectRoot
try {
    & $python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "I2-D1-K20 5k launcher failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
