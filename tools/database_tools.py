"""Database exploration and query execution tools for the SQL agent."""

from framework.agent import Tool
from framework.database import describe_table as _describe_table
from framework.database import execute_query
from framework.database import list_schemas as _list_schemas
from framework.database import list_tables as _list_tables

_MAX_RESULT_ROWS = 100


def list_schemas() -> str:
    schemas = _list_schemas()
    if not schemas:
        return "No schemas found."
    return "Available schemas:\n" + "\n".join(f"  - {s}" for s in schemas)


def list_tables(schema: str) -> str:
    tables = _list_tables(schema)
    if tables is None or not tables:
        return f"Schema '{schema}' not found or has no tables."
    return f"Tables in schema '{schema}':\n" + "\n".join(f"  - {t}" for t in tables)


def describe_table(schema: str, table: str) -> str:
    columns = _describe_table(schema, table)
    if not columns:
        return f"Table '{schema}.{table}' not found or has no columns."
    header = f"Columns in {schema}.{table}:\n"
    return header + "\n".join(f"  - {c}" for c in columns)


def run_query(query: str) -> str:
    result = execute_query(query)
    if not result.is_success:
        return f"Query error: {result.error_message}"
    df = result.dataframe
    assert df is not None
    total_rows = len(df)
    if total_rows == 0:
        return f"Query returned 0 rows.\nColumns: {df.columns}"
    truncated = df.head(_MAX_RESULT_ROWS) if total_rows > _MAX_RESULT_ROWS else df
    note = (
        f"\n[Showing {_MAX_RESULT_ROWS} of {total_rows} rows]"
        if total_rows > _MAX_RESULT_ROWS
        else f"\n[{total_rows} row(s)]"
    )
    return str(truncated) + note


LIST_SCHEMAS: Tool = Tool(
    name="list_schemas",
    description=(
        "List all available schemas (databases) in the DuckDB instance. "
        "Call this first to discover what data sources are available."
    ),
    parameters={"type": "object", "properties": {}, "required": []},
    function=list_schemas,
)

LIST_TABLES: Tool = Tool(
    name="list_tables",
    description="List all tables in a given schema.",
    parameters={
        "type": "object",
        "properties": {
            "schema": {
                "type": "string",
                "description": "The schema name (e.g. 'financial').",
            },
        },
        "required": ["schema"],
    },
    function=list_tables,
)

DESCRIBE_TABLE: Tool = Tool(
    name="describe_table",
    description=(
        "Show the column names and types for a specific table. "
        "Use this to understand the schema before writing a query."
    ),
    parameters={
        "type": "object",
        "properties": {
            "schema": {
                "type": "string",
                "description": "The schema name (e.g. 'financial').",
            },
            "table": {
                "type": "string",
                "description": "The table name (e.g. 'account').",
            },
        },
        "required": ["schema", "table"],
    },
    function=describe_table,
)

RUN_QUERY: Tool = Tool(
    name="run_query",
    description=(
        "Execute a SQL query and return the results. "
        "Use this to explore data and verify your query before submitting. "
        f"Results are capped at {_MAX_RESULT_ROWS} rows."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A valid DuckDB SQL query using schema-qualified table names.",
            },
        },
        "required": ["query"],
    },
    function=run_query,
)
