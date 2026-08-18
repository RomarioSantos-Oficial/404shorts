$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path "$PSScriptRoot\.."
$Python311 = "$env:LocalAppData\Programs\Python\Python311\python.exe"

if (-not (Test-Path -LiteralPath $Python311)) {
    throw "Python 3.11 x64 não encontrado em $Python311"
}

if (-not (Test-Path -LiteralPath "$ProjectRoot\.venv")) {
    & $Python311 -m venv "$ProjectRoot\.venv"
}

& "$ProjectRoot\.venv\Scripts\python.exe" -m pip install --use-feature=truststore "setuptools>=75"
& "$ProjectRoot\.venv\Scripts\python.exe" -m pip install --use-feature=truststore --no-build-isolation -e "${ProjectRoot}[dev]"
