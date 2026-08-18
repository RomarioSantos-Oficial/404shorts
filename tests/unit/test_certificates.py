from pathlib import Path
import ssl

from cortaflow.infrastructure import certificates


def test_configures_windows_certificates_as_pem(tmp_path: Path, monkeypatch) -> None:
    der = b"certificate-der"
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.setattr(certificates.os, "name", "nt")
    monkeypatch.setattr(
        certificates.ssl,
        "enum_certificates",
        lambda _store: [(der, "x509_asn", True)],
    )
    monkeypatch.setattr(
        certificates.ssl,
        "DER_cert_to_PEM_cert",
        lambda value: f"PEM:{value.decode('ascii')}\n",
    )

    result = certificates.configure_system_certificates(tmp_path)

    assert result == tmp_path / "windows-ca-bundle.pem"
    assert result.read_text(encoding="ascii") == "PEM:certificate-der\n"
    assert certificates.os.environ["SSL_CERT_FILE"] == str(result)


def test_preserves_explicit_certificate_bundle(tmp_path: Path, monkeypatch) -> None:
    configured = tmp_path / "custom.pem"
    configured.write_text("custom", encoding="ascii")
    monkeypatch.setenv("SSL_CERT_FILE", str(configured))
    monkeypatch.setattr(
        certificates.ssl,
        "enum_certificates",
        lambda _store: (_ for _ in ()).throw(AssertionError("must not enumerate")),
    )

    assert certificates.configure_system_certificates(tmp_path / "generated") == configured
