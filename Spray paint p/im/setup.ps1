$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw 'Python launcher not found. Install 64-bit Python 3.12 and enable the Python launcher.'
}
$installed = py -0p
if (-not ($installed -match '3\.12')) {
    throw 'Python 3.12 was not found. Install 64-bit Python 3.12, then reopen PowerShell and run setup again.'
}
py -3.12 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw 'Could not create the virtual environment.' }
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'Could not upgrade pip.' }
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Resolve the pip error above, then run setup again.' }
& .\.venv\Scripts\python.exe .\generate_audio.py
if ($LASTEXITCODE -ne 0) { throw 'Could not generate audio assets.' }
Write-Host "Setup complete. Start the canvas with: .\.venv\Scripts\python.exe .\spray_canvas.py"
