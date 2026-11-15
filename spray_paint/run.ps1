$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not (Test-Path .\.venv\Scripts\python.exe)) { throw 'Environment missing. Run .\setup.ps1 first.' }
& .\.venv\Scripts\python.exe .\spray_canvas.py @args
