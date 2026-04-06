from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Any

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


def _resolve_input_cache_dir(run_directives: DictConfig) -> Path:
    output_root = Path(run_directives.get("output_path", "./artifacts/preprocessing"))
    input_cache_dir = output_root / "input_cache"
    input_cache_dir.mkdir(parents=True, exist_ok=True)
    return input_cache_dir


@task(name="list_input_images")
def _list_input_images(run_directives: DictConfig) -> list[Path]:
    minio_client, bucket, prefix = _get_minio_client(run_directives)
    input_cache_dir = _resolve_input_cache_dir(run_directives)

    paginator = minio_client.get_paginator("list_objects_v2")
    image_paths: list[Path] = []

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
            image_paths.append(local_path)

    return sorted(image_paths)


@task(name="build_images_dataframe")
def _build_images_dataframe(image_paths: list[Path]) -> DataFrame:
    return pd.DataFrame(
        [
            {
                "image_id": str(uuid.uuid4()),
                "image_path": str(path),
            }
            for path in image_paths
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
    base_output = Path(run_directives.get("output_path", "./artifacts/preprocessing"))
    thumbnails_dir = Path(
        run_directives.get("thumbnails_path", base_output / "thumbnails")
    )

    base_output.mkdir(parents=True, exist_ok=True)
    thumbnails_dir.mkdir(parents=True, exist_ok=True)
    return base_output, thumbnails_dir


def _correct_orientation_and_generate_thumbnail(
    image_path: Path,
    thumbnails_dir: Path,
    thumbnail_size: tuple[int, int],
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

            # Persist corrected orientation in place.
            oriented_image.save(image_path)

            thumbnail = oriented_image.copy()
            thumbnail.thumbnail(thumbnail_size)
            thumbnail_name = f"{image_path.stem}_thumb{image_path.suffix.lower()}"
            thumbnail_path = thumbnails_dir / thumbnail_name
            thumbnail.save(thumbnail_path)
            result["thumbnail_path"] = str(thumbnail_path)
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
    _, thumbnails_dir = _resolve_output_paths(run_directives)

    thumbnail_width = int(run_directives.get("thumbnail_width", 256))
    thumbnail_height = int(run_directives.get("thumbnail_height", 256))
    thumbnail_size = (thumbnail_width, thumbnail_height)

    transform_results = images_df["image_path"].map(
        lambda value: _correct_orientation_and_generate_thumbnail(
            image_path=Path(value),
            thumbnails_dir=thumbnails_dir,
            thumbnail_size=thumbnail_size,
        )
    )

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
