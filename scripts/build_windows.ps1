$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path "$PSScriptRoot\.."
$Python = "$ProjectRoot\.venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) { throw "Execute setup_windows.ps1 primeiro." }
Push-Location $ProjectRoot
try {
    & $Python -m pytest
    if ($LASTEXITCODE -ne 0) { throw "Os testes falharam; build cancelado." }
    & $Python -m PyInstaller --noconfirm --clean --windowed --onedir --name CortaFlowAI --paths src src\cortaflow\main.py
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller falhou." }
} finally { Pop-Location }
