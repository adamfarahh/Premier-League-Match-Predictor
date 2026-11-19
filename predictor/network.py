"""Fetch public data with normal HTTPS verification on each platform."""

import shutil
import ssl
import subprocess
from urllib.error import URLError
from urllib.request import urlopen


def read_url(url: str) -> bytes:
    try:
        with urlopen(url, timeout=25) as response:
            return response.read()
    except URLError as error:
        # Windows curl can use an issuer in the system trust store that Python missed.
        curl = shutil.which("curl.exe")
        if not curl or not isinstance(error.reason, ssl.SSLCertVerificationError):
            raise
        result = subprocess.run(
            [curl, "--location", "--fail", "--silent", "--show-error", "--max-time", "25", url],
            capture_output=True, check=False, timeout=30,
        )
        if result.returncode:
            raise OSError(f"Verified HTTPS download failed for {url}: {result.stderr.decode(errors='replace')}") from error
        return result.stdout
