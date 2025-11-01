#!/usr/bin/env python3
"""Check PostgreSQL connectivity using asyncpg.
Usage:
  - Set DATABASE_URL environment variable, then run:
      python check_postgres.py
  - Or pass DSN as first argument:
      python check_postgres.py "postgresql://postgres:1512200011032003Dac@db.bwmqzqgnouisgshuprhh.supabase.co:5432/postgres"
"""
import os
import sys
import asyncio

try:
    import asyncpg
except Exception:
    print("asyncpg is not installed. Install: pip install asyncpg")
    sys.exit(2)

async def run_check(dsn: str):
    print(f"Trying to connect to Postgres DSN: {dsn}")
    try:
        conn = await asyncpg.connect(dsn, timeout=5)
        now = await conn.fetchval('SELECT now()')
        print(f"Connected. server time: {now}")
        await conn.close()
        return 0
    except Exception as e:
        print(f"Connection failed: {e}")
        return 1

if __name__ == '__main__':
    dsn = os.environ.get('DATABASE_URL', "postgresql://postgres:1512200011032003Dac@db.bwmqzqgnouisgshuprhh.supabase.co:5432/postgres")
    if len(sys.argv) > 1:
        dsn = sys.argv[1]
    if not dsn:
        print('No DATABASE_URL provided via env or argument.')
        sys.exit(2)
    code = asyncio.run(run_check(dsn))
    sys.exit(code)
