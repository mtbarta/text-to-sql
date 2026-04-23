"""Database tooling for executing SQL queries against the consolidated DuckDB database.

All CTU Relational databases are consolidated into a single DuckDB file (hecks.duckdb)
with each original database as its own schema. Queries use schema.table syntax.

Example:
    >>> result = execute_query("SELECT * FROM financial.account LIMIT 10")
    >>> if result.is_success:
    ...     print(result.dataframe)
"""

from dataclasses import dataclass
from pathlib import Path

import duckdb
import polars as pl
import sqlglot
from sqlglot.errors import ParseError

# Path to the consolidated database file
DATABASE_PATH = Path(__file__).parent.parent / "hecks.duckdb"

@dataclass
class QueryValidationResult:
    """Result of SQL query validation."""
    is_valid: bool
    error_message: str | None = None


def validate_query(query: str) -> QueryValidationResult:
    """Validate a SQL query string using SQLGlot.

    Args:
        query: (Potentially invalid) SQL query string.

    Returns:
        QueryValidationResult with is_valid=True if syntactically valid,
        False with error_message otherwise.
    """
    try:
        _ = sqlglot.parse_one(query, read="duckdb")
    except ParseError as e:
        return QueryValidationResult(is_valid=False, error_message=str(e))
    return QueryValidationResult(is_valid=True)


@dataclass
class QueryExecutionResult:
    """Result of executing a SQL query.

    Attributes:
        dataframe: The query results as a Polars DataFrame, or None if error.
        error_message: Error message if execution failed, None otherwise.
    """

    dataframe: pl.DataFrame | None
    error_message: str | None = None

    @property
    def is_success(self) -> bool:
        """Return True if the query executed successfully."""
        return self.error_message is None

    @property
    def is_empty(self) -> bool:
        """Return True if the query succeeded but returned no rows."""
        return self.is_success and self.dataframe is not None and self.dataframe.is_empty()


def execute_query(query: str) -> QueryExecutionResult:
    """Execute a SQL query against the consolidated database.

    Queries should use schema.table syntax (e.g., "SELECT * FROM financial.account").

    Args:
        query: SQL query string with schema-qualified table names.

    Returns:
        QueryExecutionResult containing either:
        - A Polars DataFrame with the query results (on success)
        - An error message describing what went wrong (on failure)

    Example:
        >>> result = execute_query("SELECT * FROM financial.account LIMIT 10")
        >>> if result.is_success:
        ...     print(result.dataframe)
        ... else:
        ...     print(f"Error: {result.error_message}")
    """
    conn = None
    try:
        conn = duckdb.connect(str(DATABASE_PATH), read_only=True)
        result = conn.execute(query)
        df = pl.DataFrame(result.fetch_arrow_table())
        return QueryExecutionResult(dataframe=df)
    except duckdb.Error as e:
        return QueryExecutionResult(dataframe=None, error_message=f"DuckDB error: {e}")
    except Exception as e:
        return QueryExecutionResult(dataframe=None, error_message=str(e))
    finally:
        if conn:
            conn.close()


# Helper functions for listing schemas and tables
# You can use these to make a tool if you like!

def list_schemas() -> list[str]:
    """List all available schemas (databases) in the consolidated database.

    Returns:
        Sorted list of schema names.
    """
    conn = None
    try:
        conn = duckdb.connect(str(DATABASE_PATH), read_only=True)
        result = conn.execute("""
            SELECT DISTINCT table_schema
            FROM information_schema.tables
            WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
            ORDER BY table_schema
        """).fetchall()
        return [row[0] for row in result]
    except Exception:
        return []
    finally:
        if conn:
            conn.close()


def list_tables(schema_name: str) -> list[str] | None:
    """List all tables in a schema.

    Args:
        schema_name: Name of the schema (e.g., "financial").

    Returns:
        List of table names, or None if schema not found.
    """
    conn = None
    try:
        conn = duckdb.connect(str(DATABASE_PATH), read_only=True)
        result = conn.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = ? AND table_type = 'BASE TABLE'
            ORDER BY table_name
            """,
            [schema_name],
        ).fetchall()
        return [row[0] for row in result]
    except Exception:
        return []
    finally:
        if conn:
            conn.close()

def describe_table(schema_name: str, table_name: str) -> list[str]:
    """Describe a table's columns with their types.

    Args:
        schema_name: Name of the schema (e.g., "financial").
        table_name: Name of the table (e.g., "account").

    Returns:
        List of column descriptions in "name (TYPE)" format, or empty list if not found.
    """
    conn = None
    try:
        conn = duckdb.connect(str(DATABASE_PATH), read_only=True)
        # Use information_schema for reliable column lookup
        # (DESCRIBE doesn't support parameterized identifiers)
        result = conn.execute(
            """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = ? AND table_name = ?
            ORDER BY ordinal_position
            """,
            [schema_name, table_name],
        ).fetchall()
        # Format as "column_name (TYPE, nullable)" for clarity
        columns: list[str] = []
        for col_name, data_type, is_nullable in result:
            nullable_str = ", nullable" if is_nullable == "YES" else ""
            columns.append(f"{col_name} ({data_type}{nullable_str})")
        return columns
    except Exception:
        return []
    finally:
        if conn:
            conn.close()
