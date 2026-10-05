"""Agent tools.

`TableAnalysisTool` computes exact answers over uploaded CSV / Excel tables
(sums, averages, counts, group-bys, top-N). The LLM never writes or runs code:
it only fills a small JSON spec which is validated and executed with pandas.
This avoids both arithmetic hallucinations and code-execution risks.
"""
from __future__ import annotations

import difflib

import pandas as pd

AGGS = {"sum", "mean", "min", "max", "median", "count", "count_distinct"}
OPERATIONS = AGGS | {"unique_values", "top_n", "bottom_n", "group_by", "filter_rows"}
FILTER_OPS = {"==", "!=", ">", "<", ">=", "<=", "contains"}
MAX_ROWS_OUT = 20


def _md(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:  # 'tabulate' not installed
        return df.to_string(index=False)


class ToolError(Exception):
    pass


class TableAnalysisTool:
    name = "table_analysis"

    def __init__(self, tables: dict[str, pd.DataFrame]):
        self.tables = tables

    # ---------------- schema for the planner prompt ----------------
    def describe(self, max_tables: int = 10) -> str:
        if not self.tables:
            return "(no tables uploaded)"
        lines = []
        for name, df in list(self.tables.items())[:max_tables]:
            cols = ", ".join(f"{c} [{'number' if pd.api.types.is_numeric_dtype(df[c]) else 'text'}]"
                             for c in df.columns[:30])
            lines.append(f"- '{name}' ({len(df)} rows): {cols}")
        return "\n".join(lines)

    # ---------------- helpers ----------------
    def _resolve_table(self, name: str | None) -> tuple[str, pd.DataFrame]:
        if not self.tables:
            raise ToolError("No tables are uploaded.")
        if not name:
            if len(self.tables) == 1:
                return next(iter(self.tables.items()))
            raise ToolError("Table name is required when several tables are uploaded.")
        if name in self.tables:
            return name, self.tables[name]
        lower = {k.lower(): k for k in self.tables}
        if name.lower() in lower:
            k = lower[name.lower()]
            return k, self.tables[k]
        partial = [k for k in self.tables if name.lower() in k.lower()]
        if len(partial) == 1:
            return partial[0], self.tables[partial[0]]
        raise ToolError(f"Unknown table '{name}'. Available: {', '.join(self.tables)}")

    @staticmethod
    def _resolve_column(df: pd.DataFrame, col: str | None, required: bool = True) -> str | None:
        if not col:
            if required:
                raise ToolError("A column name is required for this operation.")
            return None
        if col in df.columns:
            return col
        lower = {str(c).lower(): c for c in df.columns}
        if col.lower() in lower:
            return lower[col.lower()]
        close = difflib.get_close_matches(col.lower(), list(lower), n=1, cutoff=0.8)
        if close:
            return lower[close[0]]
        raise ToolError(f"Unknown column '{col}'. Available: {', '.join(map(str, df.columns))}")

    def _apply_filters(self, df: pd.DataFrame, filters: list[dict]) -> pd.DataFrame:
        for f in filters or []:
            col = self._resolve_column(df, f.get("column"))
            op, val = f.get("op", "=="), f.get("value")
            if op not in FILTER_OPS:
                raise ToolError(f"Unsupported filter operator '{op}'.")
            series = df[col]
            if f.get("value_column"):  # compare two columns, e.g. Stock < ReorderLevel
                other = self._resolve_column(df, f["value_column"])
                a, b = self._numeric(df, col), self._numeric(df, other)
                if op == "contains":
                    raise ToolError("'contains' cannot compare two columns.")
                mask = {"==": a == b, "!=": a != b, ">": a > b, "<": a < b, ">=": a >= b, "<=": a <= b}[op]
                df = df[mask]
                continue
            if op == "contains":
                mask = series.astype(str).str.contains(str(val), case=False, na=False, regex=False)
            elif pd.api.types.is_numeric_dtype(series) and op != "contains":
                try:
                    v = float(val)
                except (TypeError, ValueError):
                    raise ToolError(f"Column '{col}' is numeric but the filter value '{val}' is not.")
                mask = {"==": series == v, "!=": series != v, ">": series > v,
                        "<": series < v, ">=": series >= v, "<=": series <= v}[op]
            else:
                s, v = series.astype(str).str.strip().str.lower(), str(val).strip().lower()
                if op in ("==", "!="):
                    mask = (s == v) if op == "==" else (s != v)
                else:
                    try:  # dates or other comparable text
                        dt = pd.to_datetime(series, errors="coerce")
                        dv = pd.to_datetime(val)
                        mask = {">": dt > dv, "<": dt < dv, ">=": dt >= dv, "<=": dt <= dv}[op]
                    except Exception:
                        raise ToolError(f"Cannot compare text column '{col}' with '{op}'.")
            df = df[mask]
        return df

    @staticmethod
    def _numeric(df: pd.DataFrame, col: str) -> pd.Series:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() == 0:
            raise ToolError(f"Column '{col}' has no numeric values.")
        return s

    @staticmethod
    def _fmt(v) -> str:
        import numbers

        if isinstance(v, numbers.Integral):
            return f"{int(v):,}"
        if isinstance(v, numbers.Real):
            return f"{float(v):,.2f}"
        return str(v)

    # ---------------- main entry ----------------
    def run(self, spec: dict) -> str:
        op = str(spec.get("operation", "")).lower()
        if op not in OPERATIONS:
            raise ToolError(f"Unsupported operation '{op}'.")
        tname, df = self._resolve_table(spec.get("table"))
        filters = list(spec.get("filters") or [])
        # LLMs sometimes put a single filter at the top level instead of inside "filters"
        if spec.get("op") and spec.get("column") and ("value" in spec or spec.get("value_column")):
            top = {"column": spec["column"], "op": spec["op"]}
            top.update({k: spec[k] for k in ("value", "value_column") if spec.get(k) is not None})
            if top not in filters:
                filters.append(top)
        df = self._apply_filters(df, filters)
        filt_desc = (" where " + " and ".join(f"{f.get('column')} {f.get('op', '==')} {f.get('value_column') or f.get('value')}" for f in filters)
                     if filters else "")
        head = f"Table '{tname}'{filt_desc}: {len(df)} matching rows."

        if op == "count":
            return f"{head}\nCount = {len(df)}"
        if op == "filter_rows":
            return f"{head}\n{_md(df.head(MAX_ROWS_OUT))}"

        col = self._resolve_column(df, spec.get("column"))
        if op == "count_distinct":
            return f"{head}\nDistinct values of {col} = {df[col].nunique()}"
        if op == "unique_values":
            vals = df[col].dropna().astype(str).unique()[:50]
            return f"{head}\nUnique values of {col}: {', '.join(vals)}"
        if op in ("top_n", "bottom_n"):
            n = max(1, min(int(spec.get("n") or 5), MAX_ROWS_OUT))
            d = df.assign(_k=self._numeric(df, col))
            d = d.nlargest(n, "_k") if op == "top_n" else d.nsmallest(n, "_k")
            return f"{head}\n{'Top' if op == 'top_n' else 'Bottom'} {n} by {col}:\n{_md(d.drop(columns='_k'))}"
        if op == "group_by":
            gcol = self._resolve_column(df, spec.get("group_by"))
            agg = str(spec.get("agg", "sum")).lower()
            if agg not in AGGS - {"count_distinct"}:
                raise ToolError(f"Unsupported aggregation '{agg}'.")
            if agg == "count":
                res = df.groupby(gcol)[col].count()
            else:
                res = df.assign(**{col: self._numeric(df, col)}).groupby(gcol)[col].agg(agg)
            res = res.sort_values(ascending=False).head(MAX_ROWS_OUT)
            return f"{head}\n{agg} of {col} by {gcol}:\n" + _md(res.reset_index())

        # single aggregate
        s = self._numeric(df, col)
        value = getattr(s, op)()
        return f"{head}\n{op} of {col} = {self._fmt(value)}"
