from __future__ import annotations

import argparse
import os
from pathlib import Path
from tqdm import tqdm

import boto3
from botocore.exceptions import ClientError

DEFAULT_MINIO_ENDPOINT = "http://localhost:9000"
DEFAULT_MINIO_ACCESS_KEY = "minioadmin"
DEFAULT_MINIO_SECRET_KEY = "minioadmin"


def _build_minio_client():
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("MINIO_ENDPOINT", DEFAULT_MINIO_ENDPOINT),
        aws_access_key_id=os.getenv("MINIO_ROOT_USER", DEFAULT_MINIO_ACCESS_KEY),
        aws_secret_access_key=os.getenv(
            "MINIO_ROOT_PASSWORD", DEFAULT_MINIO_SECRET_KEY
        ),
        region_name="us-east-1",
    )


def _ensure_bucket_exists(client, bucket: str) -> None:
    try:
        client.head_bucket(Bucket=bucket)
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code")
        if error_code not in {"404", "NoSuchBucket", "NotFound"}:
            raise

        client.create_bucket(Bucket=bucket)


def upload_folder_to_minio(folder: Path, bucket: str) -> None:
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    if not folder.is_dir():
        raise ValueError(f"Expected a folder, got: {folder}")

    client = _build_minio_client()
    _ensure_bucket_exists(client, bucket)

    for file_path in tqdm(sorted(path for path in folder.rglob("*") if path.is_file())):
        object_key = file_path.relative_to(folder).as_posix()
        client.upload_file(str(file_path), bucket, object_key)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a local folder to MinIO")
    parser.add_argument("folder", help="Folder to upload recursively")
    parser.add_argument("bucket", help="Target MinIO bucket")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    upload_folder_to_minio(Path(args.folder), args.bucket)


if __name__ == "__main__":
    main()