"""
Run: python inspect_db.py
Shows all tables in usage.db and their contents in a readable format.
"""
import sqlite3

conn = sqlite3.connect("usage.db")
conn.row_factory = sqlite3.Row  # lets us access columns by name

tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()

print(f"Found {len(tables)} table(s): {[t['name'] for t in tables]}\n")

for table in tables:
    table_name = table["name"]
    print(f"\n{'=' * 60}")
    print(f"TABLE: {table_name}")
    print("=" * 60)

    rows = conn.execute(f"SELECT * FROM {table_name}").fetchall()
    print(f"Row count: {len(rows)}")

    if not rows:
        print("(empty)")
        continue

    for row in rows:
        print(dict(row))

conn.close()