from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import boto3
import pandas as pd
from omegaconf import DictConfig
from pandas import DataFrame
from PIL import Image, ImageOps
from botocore import UNSIGNED
from botocore.config import Config
from prefect import flow, task
from sqlalchemy import create_engine

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _get_minio_config(run_directives: DictConfig) -> DictConfig:
    minio_cfg = run_directives.get("minio_cfg") or run_directives.get("minio")
    if minio_cfg is None:
        raise ValueError("Missing 'minio_cfg' section in run directives.")

    endpoint = minio_cfg.get("endpoint")
    bucket = minio_cfg.get("bucket")

    if not endpoint:
        raise ValueError("Missing minio_cfg.endpoint in run directives.")

    if not bucket:
        raise ValueError("Missing minio_cfg.bucket in run directives.")

    return minio_cfg


def _get_minio_client(run_directives: DictConfig) -> tuple[Any, str, str]:
    minio_cfg = _get_minio_config(run_directives)
    endpoint = str(minio_cfg.get("endpoint"))
    bucket = str(minio_cfg.get("bucket"))
    prefix = str(minio_cfg.get("prefix", "")).strip("/")

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        config=Config(signature_version=UNSIGNED, s3={"addressing_style": "path"}),
        region_name="us-east-1",
    )

    return client, bucket, prefix


def _parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    parsed_uri = urlparse(s3_uri)
    if parsed_uri.scheme != "s3" or not parsed_uri.netloc:
        raise ValueError(f"Expected an s3:// URI, got: {s3_uri}")

    bucket = parsed_uri.netloc
    prefix = parsed_uri.path.lstrip("/").rstrip("/")
    return bucket, prefix


def _build_s3_uri(bucket: str, prefix: str, object_key: str) -> str:
    clean_prefix = prefix.strip("/")
    clean_key = object_key.lstrip("/")

    if clean_prefix:
        return f"s3://{bucket}/{clean_prefix}/{clean_key}"

    return f"s3://{bucket}/{clean_key}"


def _build_object_key(prefix: str, relative_path: Path) -> str:
    clean_prefix = prefix.strip("/")
    relative_key = relative_path.as_posix().lstrip("/")

    if clean_prefix:
        return f"{clean_prefix}/{relative_key}"

    return relative_key


def _resolve_input_cache_dir(run_directives: DictConfig) -> Path:
    output_root = Path(run_directives.get("output_path", "./artifacts/preprocessing"))
    input_cache_dir = output_root / "input_cache"
    input_cache_dir.mkdir(parents=True, exist_ok=True)
    return input_cache_dir


@task(name="list_input_images")
def _list_input_images(run_directives: DictConfig) -> list[dict[str, str]]:
    minio_client, bucket, prefix = _get_minio_client(run_directives)
    input_cache_dir = _resolve_input_cache_dir(run_directives)

    paginator = minio_client.get_paginator("list_objects_v2")
    image_records: list[dict[str, str]] = []

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix or ""):
        for object_summary in page.get("Contents", []):
            object_key = str(object_summary["Key"])
            if object_key.endswith("/"):
                continue

            object_suffix = Path(object_key).suffix.lower()
            if object_suffix not in IMAGE_EXTENSIONS:
                continue

            relative_key = Path(object_key)
            if prefix and object_key.startswith(f"{prefix}/"):
                relative_key = Path(object_key[len(prefix) + 1 :])

            local_path = input_cache_dir / relative_key
            local_path.parent.mkdir(parents=True, exist_ok=True)
            minio_client.download_file(bucket, object_key, str(local_path))
            image_records.append(
                {
                    "image_path": str(local_path),
                    "source_key": relative_key.as_posix(),
                }
            )

    return sorted(image_records, key=lambda record: record["source_key"])


@task(name="build_images_dataframe")
def _build_images_dataframe(image_records: list[dict[str, str]]) -> DataFrame:
    return pd.DataFrame(
        [
            {
                "image_id": str(uuid.uuid4()),
                "image_path": record["image_path"],
                "source_key": record["source_key"],
            }
            for record in image_records
        ]
    )


def _compute_file_sha256(file_path: Path) -> str:
    hash_obj = hashlib.sha256()
    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def _compute_file_sha256_from_value(value: Any) -> str:
    if not isinstance(value, (str, Path)):
        raise ValueError(f"Expected a file path, got {type(value).__name__}.")

    return _compute_file_sha256(Path(value))


@task(name="add_hash_metadata")
def _add_hash_metadata(images_df: DataFrame) -> DataFrame:
    images_df = images_df.copy()
    images_df["image_hash"] = images_df["image_path"].map(
        _compute_file_sha256_from_value
    )
    return images_df


def _resolve_output_paths(run_directives: DictConfig) -> tuple[Path, Path]:
    work_root = Path(run_directives.get("local_work_dir", "./artifacts/preprocessing"))
    base_output = work_root / "output"
    thumbnails_dir = work_root / "thumbnails"

    base_output.mkdir(parents=True, exist_ok=True)
    thumbnails_dir.mkdir(parents=True, exist_ok=True)
    return base_output, thumbnails_dir


def _correct_orientation_and_generate_thumbnail(
    image_path: Path,
    source_key: str,
    thumbnails_dir: Path,
    output_dir: Path,
    thumbnail_size: tuple[int, int],
    minio_client: Any,
    output_bucket: str,
    output_prefix: str,
    thumbnails_bucket: str,
    thumbnails_prefix: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "thumbnail_path": None,
        "width": None,
        "height": None,
        "preprocessing_error": None,
    }

    try:
        with Image.open(image_path) as image:
            oriented_image = ImageOps.exif_transpose(image)
            result["width"], result["height"] = oriented_image.size

            relative_path = Path(source_key)

            output_local_path = output_dir / relative_path
            output_local_path.parent.mkdir(parents=True, exist_ok=True)
            oriented_image.save(output_local_path)

            output_object_key = _build_object_key(output_prefix, relative_path)
            minio_client.upload_file(
                str(output_local_path), output_bucket, output_object_key
            )

            thumbnail = oriented_image.copy()
            thumbnail.thumbnail(thumbnail_size)
            thumbnail_relative_path = relative_path.with_name(
                f"{relative_path.stem}_thumb{relative_path.suffix.lower()}"
            )
            thumbnail_path = thumbnails_dir / thumbnail_relative_path
            thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
            thumbnail.save(thumbnail_path)

            thumbnail_object_key = _build_object_key(
                thumbnails_prefix, thumbnail_relative_path
            )
            minio_client.upload_file(
                str(thumbnail_path), thumbnails_bucket, thumbnail_object_key
            )
            result["thumbnail_path"] = _build_s3_uri(
                thumbnails_bucket, thumbnails_prefix, thumbnail_object_key
            )
    except Exception as exc:  # noqa: BLE001
        result["preprocessing_error"] = str(exc)

    return result


def _get_transformation_field(result: Any, field_name: str) -> Any:
    if not isinstance(result, dict):
        return None

    return result.get(field_name)


@task(name="apply_image_transformations")
def _apply_image_transformations(
    images_df: DataFrame, run_directives: DictConfig
) -> DataFrame:
    images_df = images_df.copy()
    minio_client, output_bucket, output_prefix = _get_minio_client(run_directives)
    output_uri = str(run_directives.get("output_path"))
    thumbnails_uri = str(run_directives.get("thumbnails_path"))
    thumbnails_bucket, thumbnails_prefix = _parse_s3_uri(thumbnails_uri)
    resolved_output_bucket, resolved_output_prefix = _parse_s3_uri(output_uri)
    output_dir, thumbnails_dir = _resolve_output_paths(run_directives)

    thumbnail_width = int(run_directives.get("thumbnail_width", 256))
    thumbnail_height = int(run_directives.get("thumbnail_height", 256))
    thumbnail_size = (thumbnail_width, thumbnail_height)

    if output_bucket != resolved_output_bucket or output_prefix != resolved_output_prefix:
        output_bucket = resolved_output_bucket
        output_prefix = resolved_output_prefix

    transform_results = [
        _correct_orientation_and_generate_thumbnail(
            image_path=Path(image_path),
            source_key=str(source_key),
            thumbnails_dir=thumbnails_dir,
            output_dir=output_dir,
            thumbnail_size=thumbnail_size,
            minio_client=minio_client,
            output_bucket=output_bucket,
            output_prefix=output_prefix,
            thumbnails_bucket=thumbnails_bucket,
            thumbnails_prefix=thumbnails_prefix,
        )
        for image_path, source_key in zip(
            images_df["image_path"], images_df["source_key"]
        )
    ]

    transform_results = pd.Series(transform_results, index=images_df.index)

    images_df["thumbnail_path"] = transform_results.map(
        lambda value: _get_transformation_field(value, "thumbnail_path")
    )
    images_df["width"] = transform_results.map(
        lambda value: _get_transformation_field(value, "width")
    )
    images_df["height"] = transform_results.map(
        lambda value: _get_transformation_field(value, "height")
    )
    images_df["preprocessing_error"] = transform_results.map(
        lambda value: _get_transformation_field(value, "preprocessing_error")
    )
    images_df["status"] = "pending_detection"
    return images_df


@task(name="save_dataframe_to_postgresql")
def _save_dataframe_to_postgresql(
    images_df: DataFrame, run_directives: DictConfig
) -> None:
    postgres_cfg = run_directives.get("postgres")
    if postgres_cfg is None:
        raise ValueError("Missing 'postgres' section in run directives.")

    connection_uri = postgres_cfg.get("connection_uri")
    table_name = postgres_cfg.get("table_name", "image_metadata")

    if not connection_uri:
        raise ValueError("Missing postgres.connection_uri in run directives.")

    engine = create_engine(connection_uri)
    images_df.to_sql(name=table_name, con=engine, if_exists="append", index=False)


@flow(name="preprocessing_flow")
def preprocessing_task(run_directives: DictConfig) -> DataFrame:
    image_paths = _list_input_images(run_directives)
    images_df = _build_images_dataframe(image_paths)
    images_df = _add_hash_metadata(images_df)
    images_df = _apply_image_transformations(images_df, run_directives)
    _save_dataframe_to_postgresql(images_df, run_directives)

    return images_df
