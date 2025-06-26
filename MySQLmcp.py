# mysql_mcp_server.py
import os
import mysql.connector
from mysql.connector import errorcode
from mcp.server.fastmcp import FastMCP
from typing import List, Dict, Any

mcp = FastMCP("MySQL MCP Server")

def get_connection(database: str = None):
    cfg = {
        "host": os.getenv("MYSQL_HOST", "localhost"),
        "user": os.getenv("MYSQL_USER", "root"),
        "password": os.getenv("MYSQL_PASSWORD", ""),
        "database": database or os.getenv("MYSQL_DATABASE")
    }
    return mysql.connector.connect(**cfg)

@mcp.tool()
def list_databases() -> List[str]:
    """Return list of accessible database names."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SHOW DATABASES")
    dbs = [row[0] for row in cursor]
    cursor.close()
    conn.close()
    return dbs

@mcp.tool()
def list_tables(database: str) -> List[str]:
    """Return list of tables in the specified database."""
    conn = get_connection(database)
    cursor = conn.cursor()
    cursor.execute("SHOW TABLES")
    tables = [row[0] for row in cursor]
    cursor.close()
    conn.close()
    return tables

@mcp.tool()
def describe_table(database: str, table: str) -> List[Dict[str, Any]]:
    """Return column info for a table: name, type, null, key, default."""
    conn = get_connection(database)
    cursor = conn.cursor()
    cursor.execute(f"DESCRIBE `{table}`")
    cols = [
        {
            "Field": row[0],
            "Type": row[1],
            "Null": row[2],
            "Key": row[3],
            "Default": row[4],
            "Extra": row[5],
        }
        for row in cursor
    ]
    cursor.close()
    conn.close()
    return cols

@mcp.tool()
def execute_query(database: str, query: str) -> List[Dict[str, Any]]:
    """Run a SELECT-only query in given database and return row results."""
    q = query.strip().lower()
    if not q.startswith("select"):
        raise ValueError("Only SELECT queries are allowed")
    conn = get_connection(database)
    cursor = conn.cursor(dictionary=True)
    cursor.execute(query)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows

if __name__ == "__main__":
    # starts MCP server (default stdio transport)
    mcp.run()
