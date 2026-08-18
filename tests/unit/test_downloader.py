from pathlib import Path
from threading import Event

import pytest
from yt_dlp.utils import DownloadError

from cortaflow.services import downloader
from cortaflow.services.downloader import (
    DownloadCancelled,
    download_media,
    inspect_url,
    validate_public_url,
)


@pytest.mark.parametrize(
    "url",
    ["https://youtube.com/watch?v=abc", "http://example.com/video"],
)
def test_accepts_public_urls(url: str) -> None:
    assert validate_public_url(url) == url


@pytest.mark.parametrize(
    "url",
    [
        "file:///c:/secret",
        "javascript:alert(1)",
        "http://localhost/video",
        "http://sub.localhost/video",
        "https://user:pass@example.com/video",
        "http://127.0.0.1/video",
        "http://10.0.0.1/video",
        "http://[::1]/video",
        "https://example.com:99999/video",
    ],
)
def test_rejects_unsafe_urls(url: str) -> None:
    with pytest.raises(ValueError):
        validate_public_url(url)


def test_metadata_formats_add_audio_to_video_only(monkeypatch: pytest.MonkeyPatch) -> None:
    info = {
        "id": "abc",
        "title": "Vídeo autorizado",
        "duration": 12,
        "width": 1920,
        "height": 1080,
        "extractor_key": "YouTube",
        "thumbnail": "https://example.com/thumb.jpg",
        "formats": [
            {"format_id": "22", "vcodec": "avc1", "acodec": "aac", "height": 720, "width": 1280, "ext": "mp4", "fps": 30},
            {"format_id": "137", "vcodec": "avc1", "acodec": "none", "height": 1080, "width": 1920, "ext": "mp4", "fps": 30},
            {"format_id": "140", "vcodec": "none", "acodec": "aac", "ext": "m4a"},
        ],
    }

    class FakeYoutubeDL:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, url, download):
            assert download is False
            return info

        def sanitize_info(self, extracted):
            return extracted

    monkeypatch.setattr(downloader, "YoutubeDL", FakeYoutubeDL)
    metadata = inspect_url("https://example.com/video")
    selectors = {item.format_id: item.selector for item in metadata.formats}
    assert selectors["22"] == "22"
    assert selectors["137"] == "137+ba/137"
    assert "140" not in selectors


def test_download_returns_final_merged_file_and_reports_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    final_path = tmp_path / "vídeo com espaço [abc].mp4"

    class FakeYoutubeDL:
        options = {}

        def __init__(self, options):
            type(self).options = options

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, url, download):
            final_path.write_bytes(b"fake media")
            payload = {"status": "finished", "filename": str(final_path), "info_dict": {"filepath": str(final_path)}}
            for hook in self.options["progress_hooks"]:
                hook(payload)
            for hook in self.options["postprocessor_hooks"]:
                hook({"status": "finished", "info_dict": {"filepath": str(final_path)}})
            return {"id": "abc", "filepath": str(final_path)}

        def prepare_filename(self, info):
            return str(tmp_path / "arquivo anterior.webm")

    monkeypatch.setattr(downloader, "YoutubeDL", FakeYoutubeDL)
    fake_ffmpeg = tmp_path / "Ferramentas FFmpeg" / "ffmpeg.exe"
    monkeypatch.setattr(downloader, "find_executable", lambda _name: fake_ffmpeg)
    events = []
    result = download_media("https://example.com/video", tmp_path, "137+ba/137", "vídeo: teste?", events.append, Event())
    assert result == final_path.resolve()
    assert FakeYoutubeDL.options["format"] == "137+ba/137"
    assert FakeYoutubeDL.options["outtmpl"]["default"].startswith("vídeo_ teste_")
    assert FakeYoutubeDL.options["overwrites"] is False
    assert FakeYoutubeDL.options["compat_opts"] == ["no-certifi"]
    assert FakeYoutubeDL.options["ffmpeg_location"] == str(fake_ffmpeg)
    assert any(event["status"] == "postprocessing" for event in events)


def test_download_cancellation_is_cooperative(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeYoutubeDL:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, url, download):
            self.options["progress_hooks"][0]({"status": "downloading"})

    monkeypatch.setattr(downloader, "YoutubeDL", FakeYoutubeDL)
    monkeypatch.setattr(downloader, "find_executable", lambda _name: tmp_path / "ffmpeg.exe")
    cancelled = Event()
    cancelled.set()
    with pytest.raises(DownloadCancelled):
        download_media("https://example.com/video", tmp_path, None, "Vídeo", lambda _: None, cancelled)


def test_metadata_network_error_is_private(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingYoutubeDL:
        def __init__(self, options):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, url, download):
            raise DownloadError("secret query content")

    monkeypatch.setattr(downloader, "YoutubeDL", FailingYoutubeDL)
    with pytest.raises(RuntimeError, match="Não foi possível consultar") as error:
        inspect_url("https://example.com/video?token=secret")
    assert "token" not in str(error.value)


def test_metadata_uses_windows_certificate_store(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict = {}

    class FakeYoutubeDL:
        def __init__(self, options):
            observed.update(options)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, url, download):
            return {"title": "Teste", "formats": []}

        def sanitize_info(self, extracted):
            return extracted

    monkeypatch.setattr(downloader, "YoutubeDL", FakeYoutubeDL)
    inspect_url("https://example.com/video")

    assert observed["compat_opts"] == ["no-certifi"]


def test_download_uses_youtube_android_client_for_compatibility(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(downloader, "find_executable", lambda _name: tmp_path / "ffmpeg.exe")
    monkeypatch.setattr(downloader, "find_javascript_runtime", lambda: ("deno", tmp_path / "deno.exe"))
    monkeypatch.setattr(downloader.importlib.util, "find_spec", lambda _name: object())

    options = downloader._download_options(
        tmp_path, None, "Teste", lambda _state: None, lambda _state: None
    )

    assert options["extractor_args"] == {"youtube": {"player_client": ["android", "mweb"]}}


def test_download_configures_project_javascript_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    deno = tmp_path / "deno.exe"
    deno.write_bytes(b"runtime")
    monkeypatch.setattr(downloader, "find_javascript_runtime", lambda: ("deno", deno))
    monkeypatch.setattr(downloader.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(downloader, "find_executable", lambda _name: tmp_path / "ffmpeg.exe")

    options = downloader._download_options(
        tmp_path, None, "Teste", lambda _state: None, lambda _state: None
    )

    assert options["js_runtimes"] == {"deno": {"path": str(deno)}}


def test_download_explains_missing_javascript_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_runtime():
        raise downloader.JavaScriptRuntimeNotFound("runtime ausente")

    monkeypatch.setattr(downloader, "find_javascript_runtime", missing_runtime)
    monkeypatch.setattr(downloader, "find_executable", lambda _name: tmp_path / "ffmpeg.exe")
    with pytest.raises(downloader.JavaScriptRuntimeNotFound, match="runtime ausente"):
        downloader._download_options(
            tmp_path, None, "Teste", lambda _state: None, lambda _state: None
        )
