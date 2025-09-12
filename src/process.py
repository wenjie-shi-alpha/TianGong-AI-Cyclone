"""NOAA public S3 download helper.

This module provides a single function ``download_from_noaa`` to download one
object from the public NOAA OAR MLWP S3 bucket (anonymous access).

Basic usage:
    from src.download import download_from_noaa
    path = download_from_noaa("gpm/IMERG/2024/01/01/file.nc")
    print(path)

Temporary file download:
    tmp_path = download_from_noaa("gpm/IMERG/2024/01/01/file.nc", to_temp=True)
    # use file, then optionally delete
    tmp_path.unlink()
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import tempfile

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError

_BUCKET = "noaa-oar-mlwp-data"
_S3_CLIENT = None  # lazy init

__all__ = ["download_from_noaa"]


def _get_s3():
    global _S3_CLIENT
    if _S3_CLIENT is None:
        _S3_CLIENT = boto3.client(
            "s3", region_name="us-east-1", config=Config(signature_version=UNSIGNED)
        )
    return _S3_CLIENT


def download_from_noaa(
    key: str,
    destination: Optional[str | Path] = None,
    bucket: str = _BUCKET,
    to_temp: bool = False,
) -> Path:
    """Download a file from the NOAA MLWP public S3 bucket.

    中文说明: 从 NOAA 公共 S3 桶下载指定对象到本地。

    Parameters
    ----------
    key : str
        Object key (path inside the bucket).
    destination : str | Path | None, optional
        Local file path or directory. If a directory or endswith('/'), filename
        derived from key. If None, downloads into current directory preserving
        key's basename. Ignored when ``to_temp=True``.
    bucket : str, optional
        S3 bucket name (default: noaa-oar-mlwp-data)
    to_temp : bool, optional
        If True, download into a new temporary file (system tmp dir) and return
        its path. The suffix (extension) is preserved based on the key.

    Returns
    -------
    Path
        Path to the downloaded local file.

    Raises
    ------
    FileNotFoundError
        If the object key does not exist (HTTP 404).
    RuntimeError
        For other S3 related errors.
    """
    s3 = _get_s3()

    # Decide destination path
    if to_temp:
        suffix = Path(key).suffix
        # NamedTemporaryFile delete=False so caller controls lifecycle
        tmp = tempfile.NamedTemporaryFile(prefix="noaa_", suffix=suffix, delete=False)
        dest_path = Path(tmp.name)
        tmp.close()  # we will write via s3.download_file
    else:
        if destination is None:
            dest_path = Path(Path.cwd(), Path(key).name)
        else:
            destination = Path(destination)
            if destination.exists() and destination.is_dir():
                dest_path = destination / Path(key).name
            elif destination.suffix == "" and str(destination).endswith("/"):
                # Treat as directory path not yet created
                dest_path = destination / Path(key).name
            else:
                # Treat as explicit file path
                dest_path = destination

        # Ensure parent directory exists
        dest_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        s3.download_file(bucket, key, str(dest_path))
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        if error_code in {"404", "NoSuchKey", "NotFound"}:
            raise FileNotFoundError(f"Object not found: s3://{bucket}/{key}") from e
        raise RuntimeError(f"Failed to download s3://{bucket}/{key}: {e}") from e

    return dest_path
