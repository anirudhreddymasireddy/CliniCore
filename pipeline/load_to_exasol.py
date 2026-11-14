"""Load every CSV / CSV.GZ under a folder into an Exasol schema (one table per file).
Usage:  python load_to_exasol.py <folder> <SCHEMA_NAME>
Connection: EXA_DSN (default 127.0.0.1:8563), EXA_USER, EXA_PASS  (see: exakit info)
"""
import os, sys, glob, ssl
import pandas as pd


def ddl(df, table, wide_text=False):
    """CREATE TABLE from a DataFrame. wide_text=True (append-only tables such as staff feedback): every text column gets room
    for later, longer entries (VARCHAR 2000) and all-empty columns become text, so a short first row never limits later rows."""
    cols = []
    for c in df.columns:
        s = df[c]
        if wide_text and (s.dtype == object or pd.api.types.is_string_dtype(s) or s.isna().all()):
            t = "VARCHAR(2000)"
        elif pd.api.types.is_integer_dtype(s):
            t = "BIGINT"
        elif pd.api.types.is_float_dtype(s):
            t = "DOUBLE"
        elif pd.api.types.is_bool_dtype(s):
            t = "BOOLEAN"
        else:
            n = int(s.dropna().astype(str).str.len().max() or 1)
            t = f"VARCHAR({min(max(50, n * 2), 20000)})"
        cols.append(f'"{str(c).upper()}" {t}')
    return f'CREATE OR REPLACE TABLE "{table.upper()}" (' + ", ".join(cols) + ")"


def main(folder, schema):
    import pyexasol                      # imported here so ddl() can be used (and tested) without the driver
    dsn = os.environ.get("EXA_DSN", "127.0.0.1:8563")
    local = dsn.split(":")[0] in ("127.0.0.1", "localhost") or os.environ.get("EXA_INSECURE") == "1"
    con = pyexasol.connect(dsn=dsn, user=os.environ.get("EXA_USER", "sys"), password=os.environ["EXA_PASS"], encryption=True,
                           websocket_sslopt={"cert_reqs": ssl.CERT_NONE if local else ssl.CERT_REQUIRED})
    con.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
    con.execute(f'OPEN SCHEMA "{schema}"')
    files = sorted(glob.glob(os.path.join(folder, "**", "*.csv*"), recursive=True))
    if not files:
        sys.exit(f"ERROR: no CSV files found under {folder}")
    for f in files:
        name = os.path.basename(f).split(".csv")[0]
        df = pd.read_csv(f, low_memory=False)
        df.columns = [str(c).upper() for c in df.columns]
        con.execute(ddl(df, name))
        con.import_from_pandas(df, name.upper())
        print(f"{name:22s} {len(df):>9,} rows")
    print("done")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
