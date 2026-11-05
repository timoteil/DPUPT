Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    Write-Error "Python launcher 'py' was not found. Install Python 3.11 or 3.12 from https://www.python.org/downloads/ and check 'Add python.exe to PATH'."
}

py -3 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
# CPU PyTorch is enough for 324x244 frames and stays smaller than the CUDA wheel.
& .\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host ""
Write-Host "Ready. Stay on a network with internet until this finishes."
Write-Host "Then activate the venv and start a script from this folder:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python opencv-viewer.py"
Write-Host "  python follow_me.py MOTIVE_SERVER_IP --client THIS_PC_IP --body-id 4 --address 0xE7E7E7E7E7"
