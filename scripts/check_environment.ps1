$ErrorActionPreference = "Continue"
$PythonPath = "$PSScriptRoot\..\.venv\Scripts\python.exe"

Write-Host "CortaFlow AI - diagnóstico"
Write-Host "Python:"
& $PythonPath --version

Write-Host "GPU:"
& nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

Write-Host "FFmpeg:"
$FFmpegPath = & $PythonPath -c "from cortaflow.infrastructure.ffmpeg import find_executable; print(find_executable('ffmpeg'))"
& $FFmpegPath -version | Select-Object -First 1

Write-Host "FFprobe:"
$FFprobePath = & $PythonPath -c "from cortaflow.infrastructure.ffmpeg import find_executable; print(find_executable('ffprobe'))"
& $FFprobePath -version | Select-Object -First 1

Write-Host "Deno/EJS:"
& "$PSScriptRoot\..\.venv\Scripts\deno.exe" --version | Select-Object -First 1
& $PythonPath -c "import importlib.metadata as m; print('yt-dlp-ejs', m.version('yt-dlp-ejs'))"

Write-Host "Dispositivo de transcrição realmente utilizável:"
& $PythonPath -c "from cortaflow.services.transcription import diagnose_compute_device; s=diagnose_compute_device(); print(f'{s.device.upper()} · {s.compute_type} · {s.detail}')"

Write-Host "IA semântica local:"
& $PythonPath -c "from cortaflow.services.semantic_models import find_ollama_assets; a=find_ollama_assets(); print(f'Ollama · {a.model_name} · pronto' if a else 'Ollama/modelo indisponível · fallback heurístico pronto')"
