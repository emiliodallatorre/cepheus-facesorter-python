from __future__ import annotations

from omegaconf import DictConfig
from prefect import task
from sqlalchemy import create_engine, text


def _get_connection_uri(run_directives: DictConfig) -> str:
    postgres_cfg = run_directives.get("postgres_cfg") or run_directives.get("postgres")
    if postgres_cfg is None:
        raise ValueError("Missing 'postgres_cfg' (or 'postgres') section in run directives.")

    connection_uri = postgres_cfg.get("connection_uri")
    if not connection_uri:
        raise ValueError("Missing postgres connection_uri in run directives.")

    return str(connection_uri)


@task(name="healthcheck_database")
def healthcheck_database_task(run_directives: DictConfig) -> None:
    connection_uri = _get_connection_uri(run_directives)
    engine = create_engine(connection_uri, pool_pre_ping=True)

    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
