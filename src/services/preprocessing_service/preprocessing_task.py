from __future__ import annotations

import hashlib
from io import BytesIO
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
from prefect import flow, get_run_logger, task
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
    user = minio_cfg.get("user")
    password = minio_cfg.get("password")

    if user and password:
        client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=str(user),
            aws_secret_access_key=str(password),
            config=Config(s3={"addressing_style": "path"}),
            region_name="us-east-1",
        )
    else:
        client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            config=Config(
                signature_version=UNSIGNED, s3={"addressing_style": "path"}
            ),
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


def _get_image_bytes(minio_client: Any, bucket: str, object_key: str) -> bytes:
    response = minio_client.get_object(Bucket=bucket, Key=object_key)
    try:
        return response["Body"].read()
    finally:
        response["Body"].close()


@task(name="list_input_images")
def _list_input_images(run_directives: DictConfig) -> list[dict[str, str]]:
    minio_client, bucket, prefix = _get_minio_client(run_directives)
    logger = get_run_logger()
    logger.info(
        "Scanning MinIO bucket '%s' with prefix '%s' for input images.",
        bucket,
        prefix or "/",
    )

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

            image_records.append(
                {
                    "image_path": _build_s3_uri(bucket, "", object_key),
                    "object_key": object_key,
                    "source_key": relative_key.as_posix(),
                }
            )
            logger.info("Queued input image: %s", object_key)

    logger.info("Discovered %d image(s) to preprocess.", len(image_records))
    return sorted(image_records, key=lambda record: record["source_key"])


@task(name="build_images_dataframe")
def _build_images_dataframe(image_records: list[dict[str, str]]) -> DataFrame:
    return pd.DataFrame(
        [
            {
                "image_id": str(uuid.uuid4()),
                "image_path": record["image_path"],
                "object_key": record["object_key"],
                "source_key": record["source_key"],
            }
            for record in image_records
        ]
    )


def _compute_bytes_sha256(content: bytes) -> str:
    hash_obj = hashlib.sha256()
    hash_obj.update(content)
    return hash_obj.hexdigest()


def _compute_object_sha256(minio_client: Any, bucket: str, object_key: str) -> str:
    image_bytes = _get_image_bytes(minio_client, bucket, object_key)
    return _compute_bytes_sha256(image_bytes)



@task(name="add_hash_metadata")
def _add_hash_metadata(images_df: DataFrame, run_directives: DictConfig) -> DataFrame:
    images_df = images_df.copy()
    minio_client, bucket, _ = _get_minio_client(run_directives)
    logger = get_run_logger()
    total = len(images_df.index)

    hashes: list[str] = []
    for index, object_key in enumerate(images_df["object_key"], start=1):
        object_key_str = str(object_key)
        logger.info("[%d/%d] Computing hash for %s", index, total, object_key_str)
        hashes.append(_compute_object_sha256(minio_client, bucket, object_key_str))

    images_df["image_hash"] = hashes
    logger.info("Completed hashing for %d image(s).", total)
    return images_df


def _resolve_pillow_format(source_key: str) -> str:
    suffix = Path(source_key).suffix.lower()
    format_by_suffix = {
        ".jpg": "JPEG",
        ".jpeg": "JPEG",
        ".png": "PNG",
        ".bmp": "BMP",
        ".tif": "TIFF",
        ".tiff": "TIFF",
        ".webp": "WEBP",
    }

    return format_by_suffix.get(suffix, "JPEG")


def _correct_orientation_and_generate_thumbnail(
    image_bytes: bytes,
    source_key: str,
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
        with Image.open(BytesIO(image_bytes)) as image:
            oriented_image = ImageOps.exif_transpose(image)
            result["width"], result["height"] = oriented_image.size

            relative_path = Path(source_key)
            image_format = image.format or _resolve_pillow_format(source_key)

            output_buffer = BytesIO()
            oriented_image.save(output_buffer, format=image_format)
            output_buffer.seek(0)

            output_object_key = _build_object_key(output_prefix, relative_path)
            minio_client.upload_fileobj(output_buffer, output_bucket, output_object_key)

            thumbnail = oriented_image.copy()
            thumbnail.thumbnail(thumbnail_size)
            thumbnail_relative_path = relative_path.with_name(
                f"{relative_path.stem}_thumb{relative_path.suffix.lower()}"
            )
            thumbnail_buffer = BytesIO()
            thumbnail.save(thumbnail_buffer, format=image_format)
            thumbnail_buffer.seek(0)

            thumbnail_object_key = _build_object_key(
                thumbnails_prefix, thumbnail_relative_path
            )
            minio_client.upload_fileobj(
                thumbnail_buffer, thumbnails_bucket, thumbnail_object_key
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
    minio_client, input_bucket, _ = _get_minio_client(run_directives)
    logger = get_run_logger()
    output_uri = str(run_directives.get("output_path"))
    thumbnails_uri = str(run_directives.get("thumbnails_path"))
    thumbnails_bucket, thumbnails_prefix = _parse_s3_uri(thumbnails_uri)
    output_bucket, output_prefix = _parse_s3_uri(output_uri)

    thumbnail_width = int(run_directives.get("thumbnail_width", 256))
    thumbnail_height = int(run_directives.get("thumbnail_height", 256))
    thumbnail_size = (thumbnail_width, thumbnail_height)

    total = len(images_df.index)
    transform_results: list[dict[str, Any]] = []
    for index, (object_key, source_key) in enumerate(
        zip(images_df["object_key"], images_df["source_key"]), start=1
    ):
        object_key_str = str(object_key)
        source_key_str = str(source_key)
        logger.info(
            "[%d/%d] Processing image %s", index, total, object_key_str
        )

        result = _correct_orientation_and_generate_thumbnail(
            image_bytes=_get_image_bytes(minio_client, input_bucket, object_key_str),
            source_key=source_key_str,
            thumbnail_size=thumbnail_size,
            minio_client=minio_client,
            output_bucket=output_bucket,
            output_prefix=output_prefix,
            thumbnails_bucket=thumbnails_bucket,
            thumbnails_prefix=thumbnails_prefix,
        )

        if result.get("preprocessing_error"):
            logger.warning(
                "[%d/%d] Failed processing %s: %s",
                index,
                total,
                object_key_str,
                result.get("preprocessing_error"),
            )
        else:
            logger.info(
                "[%d/%d] Uploaded outputs for %s",
                index,
                total,
                object_key_str,
            )

        transform_results.append(result)

    logger.info("Completed transform/upload step for %d image(s).", total)

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
    postgres_cfg = run_directives.get("postgres_cfg")
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
    images_df = _add_hash_metadata(images_df, run_directives)
    images_df = _apply_image_transformations(images_df, run_directives)
    _save_dataframe_to_postgresql(images_df, run_directives)

    return images_df
