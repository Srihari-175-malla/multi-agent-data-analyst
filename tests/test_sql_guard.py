"""Unit tests for the SQL guardrail."""
from src.guardrails.sql_guard import SQLGuard


def test_allows_simple_select():
    assert SQLGuard.check("SELECT * FROM {table}").allowed


def test_allows_with_cte():
    assert SQLGuard.check("WITH t AS (SELECT 1) SELECT * FROM t").allowed


def test_blocks_drop():
    result = SQLGuard.check("DROP TABLE {table}")
    assert not result.allowed


def test_blocks_insert():
    result = SQLGuard.check("INSERT INTO {table} VALUES (1)")
    assert not result.allowed


def test_blocks_multiple_statements():
    result = SQLGuard.check("SELECT 1; DROP TABLE {table}")
    assert not result.allowed


def test_blocks_update_disguised_in_select_like_string():
    result = SQLGuard.check("SELECT * FROM {table}; UPDATE {table} SET x=1")
    assert not result.allowed


def test_row_limit_enforced_when_missing():
    query = SQLGuard.enforce_row_limit("SELECT * FROM t", 100)
    assert "LIMIT 100" in query


def test_row_limit_not_duplicated_when_present():
    query = SQLGuard.enforce_row_limit("SELECT * FROM t LIMIT 5", 100)
    assert query.count("LIMIT") == 1
