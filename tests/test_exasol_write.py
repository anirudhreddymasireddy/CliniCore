"""write_exasol logic against a fake connection (no database needed)."""
import os, sys, pandas as pd, pytest
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline"))
from stage1_pipeline import write_exasol


class Res:
    def __init__(self, rows): self.rows = rows
    def fetchone(self): return self.rows[0]
    def fetchall(self): return self.rows


class Fake:
    def __init__(self, existing=None, bad=()):
        self.existing, self.bad, self.sql, self.imported = existing, set(bad), [], None
    def execute(self, q):
        self.sql.append(q)
        if "EXA_ALL_TABLES" in q: return Res([(1 if self.existing is not None else 0,)])
        if "EXA_ALL_COLUMNS" in q: return Res([(c,) for c in self.existing or []])
        if q.startswith("CREATE OR REPLACE"):
            for b in self.bad:
                if f'"{b}" ' in q:
                    raise RuntimeError(f"{b} not allowed as column name.")
        return Res([(0,)])
    def import_from_pandas(self, d, target): self.imported = (list(d.columns), target)


DF = pd.DataFrame({"stay_id": [1], "level": ["High"], "x": [1.0]})


def test_reserved_column_is_renamed():
    c = Fake(bad=["LEVEL"]); write_exasol(c, "S", "T", DF, replace=True)
    assert c.imported[0] == ["STAY_ID", "LEVEL_X", "X"]


def test_replace_recreates_even_if_table_exists():
    c = Fake(existing=["OLD"]); write_exasol(c, "S", "T", DF, replace=True)
    assert any(q.startswith("CREATE OR REPLACE") for q in c.sql) and c.imported[0] == ["STAY_ID", "LEVEL", "X"]


def test_append_with_different_layout_raises_clear_error():
    c = Fake(existing=["STAY_ID", "LEVEL"])
    with pytest.raises(RuntimeError, match="already exists with columns"):
        write_exasol(c, "S", "T", DF)


def test_append_to_matching_table_adds_rows_without_recreate():
    c = Fake(existing=["STAY_ID", "LEVEL", "X"]); write_exasol(c, "S", "T", DF)
    assert not any(q.startswith("CREATE OR REPLACE") for q in c.sql) and c.imported[1] == ("S", "T")


def test_append_finds_renamed_column():
    c = Fake(existing=["STAY_ID", "LEVEL_X", "X"]); write_exasol(c, "S", "T", DF)
    assert c.imported[0] == ["STAY_ID", "LEVEL_X", "X"]
