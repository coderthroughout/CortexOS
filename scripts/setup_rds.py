#!/usr/bin/env python3
"""
One-time RDS setup: create database cortexos (if needed), enable pgvector, run schema.
Uses CORTEX_DATABASE_URL from .env. For RDS, set URL to the *cortexos* database;
if that DB doesn't exist yet, set URL to postgres first and we create cortexos.
Example: postgresql://postgres:YOUR_PASSWORD@cortexos-db.xxx.us-east-1.rds.amazonaws.com:5432/postgres
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.parse import urlparse, urlunparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT


def main() -> int:
    url = os.environ.get("CORTEX_DATABASE_URL")
    if not url:
        print("Set CORTEX_DATABASE_URL in .env (e.g. postgresql://postgres:PASSWORD@cortexos-db.xxx.rds.amazonaws.com:5432/postgres)")
        return 1
    parsed = urlparse(url)
    if parsed.scheme != "postgresql":
        # support postgresql:// or postgres://
        if url.startswith("postgres://"):
            url = "postgresql://" + url[9:]
            parsed = urlparse(url)
    netloc = parsed.netloc
    path = parsed.path or "/postgres"
    db_name = path.lstrip("/") or "postgres"

    # 1) Connect to default DB (postgres) and create cortexos if missing
    base_url = urlunparse((parsed.scheme, netloc, "/postgres", parsed.params, parsed.query, parsed.fragment))
    if "?" in url and "sslmode" not in url:
        base_url += "&sslmode=prefer" if "?" in base_url else "?sslmode=prefer"
    elif "?" not in base_url:
        base_url += "?sslmode=prefer"

    print("Connecting to RDS (postgres)...")
    try:
        conn = psycopg2.connect(base_url)
    except Exception as e:
        print(f"Connection failed: {e}")
        print("Check: CORTEX_DATABASE_URL, security group (port 5432 from your IP), and master password.")
        return 1

    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", ("cortexos",))
    if cur.fetchone() is None:
        print("Creating database cortexos...")
        cur.execute(sql.SQL("CREATE DATABASE cortexos"))
    else:
        print("Database cortexos already exists.")
    cur.close()
    conn.close()

    # 2) Connect to cortexos and run extension + schema
    cortex_url = urlunparse((parsed.scheme, netloc, "/cortexos", parsed.params, parsed.query, parsed.fragment))
    if "?" in cortex_url:
        cortex_url += "&sslmode=prefer" if "sslmode" not in cortex_url else ""
    else:
        cortex_url += "?sslmode=prefer"

    print("Connecting to cortexos...")
    try:
        conn = psycopg2.connect(cortex_url)
    except Exception as e:
        print(f"Connection failed: {e}")
        return 1

    cur = conn.cursor()
    print("Enabling pgvector extension...")
    try:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()
    except Exception as e:
        print(f"pgvector extension failed: {e}")
        print("If RDS says extension not available, use PostgreSQL 15+ and enable in parameter group or use Aurora.")
        conn.rollback()
        cur.close()
        conn.close()
        return 1

    schema_path = Path(__file__).resolve().parent.parent / "cortex" / "memory" / "db_schema.sql"
    print(f"Running schema from {schema_path.name}...")
    sql_text = schema_path.read_text()
    # Remove line comments then split by semicolon (each statement ends with ;)
    lines = []
    for line in sql_text.splitlines():
        if "--" in line:
            line = line[: line.index("--")].strip()
        else:
            line = line.strip()
        if line:
            lines.append(line)
    full = " ".join(lines)
    statements = [s.strip() + ";" for s in full.split(";") if s.strip()]

    for stmt in statements:
        try:
            cur.execute(stmt)
            conn.commit()
        except Exception as e:
            conn.rollback()
            if "already exists" in str(e).lower():
                pass
            else:
                print(f"Warning: {e}")
    cur.close()
    conn.close()
    print("Done. RDS is ready. Use in .env:")
    print(f"  CORTEX_DATABASE_URL=postgresql://postgres:PASSWORD@{parsed.hostname}:5432/cortexos")
    return 0


if __name__ == "__main__":
    sys.exit(main())
