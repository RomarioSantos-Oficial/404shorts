"""Bridge the Windows certificate stores to libraries that expect a PEM file."""

from __future__ import annotations

import os
from pathlib import Path
import ssl
import tempfile


def configure_system_certificates(cache_directory: Path) -> Path | None:
    """Expose trusted Windows roots through ``SSL_CERT_FILE`` without disabling TLS."""
    configured = os.environ.get("SSL_CERT_FILE")
    if configured and Path(configured).is_file():
        return Path(configured)
    if os.name != "nt" or not hasattr(ssl, "enum_certificates"):
        return None

    certificates: set[bytes] = set()
    for store_name in ("ROOT", "CA"):
        for certificate, encoding, _trust in ssl.enum_certificates(store_name):
            if encoding == "x509_asn":
                certificates.add(certificate)
    if not certificates:
        return None

    cache_directory.mkdir(parents=True, exist_ok=True)
    bundle_path = cache_directory / "windows-ca-bundle.pem"
    content = "".join(
        ssl.DER_cert_to_PEM_cert(certificate)
        for certificate in sorted(certificates)
    ).encode("ascii")
    if not bundle_path.is_file() or bundle_path.read_bytes() != content:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=cache_directory, prefix="ca-", suffix=".tmp", delete=False
        ) as temporary:
            temporary.write(content)
            temporary_path = Path(temporary.name)
        temporary_path.replace(bundle_path)

    os.environ["SSL_CERT_FILE"] = str(bundle_path)
    return bundle_path
