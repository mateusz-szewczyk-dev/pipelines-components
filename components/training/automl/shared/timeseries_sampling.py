"""Bounded, timestamp-based tail sampling for AutoML time series CSVs."""

from __future__ import annotations

import heapq
import pickle
import sys
import tempfile

SAMPLING_METHOD = "last_values_per_series"
SYNTHETIC_ID_COLUMN = "__synthetic_item_id"


def _row_memory_bytes(row: tuple) -> int:
    """Conservatively account for row values and heap/dictionary storage.

    Include a string representation allowance for mixed columns normalized before
    Parquet export. The budget covers retained buffers, not just pandas cells.
    """
    return 192 + sys.getsizeof(row) + sum(max(sys.getsizeof(value), sys.getsizeof(str(value))) for value in row)


def _timestamp_value(value):
    """Serialize normalized datetimes as ISO strings and fractional years as numbers."""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value.item() if hasattr(value, "item") else value


def dataframe_series_profile(data, id_column: str, timestamp_column: str) -> dict:
    """Return JSON-compatible row counts and timestamp ranges keyed by series ID."""
    return {
        str(item_id): {
            "rows": len(series),
            "timestamp_min": _timestamp_value(series[timestamp_column].min()),
            "timestamp_max": _timestamp_value(series[timestamp_column].max()),
        }
        for item_id, series in data.groupby(id_column, sort=False)
        if len(series)
    }


def _normalize_chunk(chunk, id_column, timestamp_column, target, synthetic_id, require_two_columns):
    """Validate and normalize a chunk while preserving file order and duplicate keys."""
    import pandas as pd

    required = {timestamp_column, target}
    if not synthetic_id:
        required.add(id_column)
    missing = required - set(chunk.columns)
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}. Available columns: {list(chunk.columns)}")
    if synthetic_id:
        if SYNTHETIC_ID_COLUMN in chunk.columns:
            raise ValueError(f"Dataset already contains reserved synthetic ID column {SYNTHETIC_ID_COLUMN!r}.")
        if require_two_columns and len(chunk.columns) != 2:
            raise ValueError(
                "When id_column is not provided, the dataset must have exactly 2 columns "
                f"(timestamp + target), but found {len(chunk.columns)} columns. Provide id_column."
            )
        chunk[SYNTHETIC_ID_COLUMN] = "item_0"

    out = chunk.replace([float("inf"), float("-inf")], float("nan"))
    if out[id_column].isna().any():
        raise ValueError(f"Column {id_column!r} contains null values. Fix the input data; do not drop rows here.")
    # A fixed string ID dtype also preserves leading zeros and prevents a series
    # from changing identity as pandas infers different dtypes across chunks.
    out[id_column] = out[id_column].astype(str)
    numeric = pd.to_numeric(out[timestamp_column], errors="coerce")
    non_null = out[timestamp_column][out[timestamp_column].notna()]
    is_numeric = len(non_null) > 0 and pd.to_numeric(non_null, errors="coerce").notna().all()
    if is_numeric:
        values = numeric[numeric.notna()]
        if not (1800 <= values.min() <= values.max() <= 2200):
            raise ValueError(
                f"Column {timestamp_column!r} contains numeric values outside the fractional year range "
                "(1800-2200). If these are Unix timestamps, convert them to ISO date strings upstream."
            )
        out[timestamp_column] = numeric
        timestamp_kind = "fractional_year"
    else:
        out[timestamp_column] = pd.to_datetime(
            out[timestamp_column], errors="coerce", utc=True, format="mixed"
        ).dt.tz_localize(None)
        timestamp_kind = "datetime"
    bad = int(out[timestamp_column].isna().sum())
    if bad:
        raise ValueError(
            f"Column {timestamp_column!r} has {bad} value(s) that could not be parsed as datetimes. "
            "Fix the input data. Dropping those rows would create irregular series."
        )
    return out, timestamp_kind


def _series_quotas(stats: dict, budget: int) -> dict:
    """Water-fill a common history length, retaining short series in full."""
    # Reserve space for the result's RangeIndex. Python row costs conservatively
    # bound both the buffers and the eventual pandas/string-normalized output.
    available = budget - 132
    if sum(item["row_bytes"] for item in stats.values()) > available:
        raise ValueError("Sampling budget cannot retain at least one observation per time series; increase preset.")
    low, high = 1, max(item["source_rows_seen"] for item in stats.values())
    while low < high:
        candidate = (low + high + 1) // 2
        size = sum(min(candidate, item["source_rows_seen"]) * item["row_bytes"] for item in stats.values())
        if size <= available:
            low = candidate
        else:
            high = candidate - 1
    return {item_id: min(low, item["source_rows_seen"]) for item_id, item in stats.items()}


def sample_timeseries_csv(
    stream,
    *,
    id_column: str,
    timestamp_column: str,
    target: str,
    max_size_bytes: int,
    chunk_size: int = 10000,
    spool_directory=None,
    synthetic_id: bool = False,
    require_two_columns: bool = False,
):
    """Scan a CSV to EOF, then keep each series' newest unique timestamps.

    The first pass validates every row and spools normalized tuples to a private
    temporary file. The second pass maintains bounded timestamp heaps using
    deterministic quotas calculated from source statistics. No second S3 download
    is needed. Any incomplete read is fatal. Temporary files are always closed.

    Quotas use conservative maximum row costs per series; actual buffer and result
    size are checked as well. Metadata is bounded separately by the same budget.
    RAM is O(sample budget + metadata + one CSV chunk), disk is O(source size).
    Conflicting duplicate keys keep the last occurrence in file order.
    """
    import pandas as pd

    if max_size_bytes <= 0 or chunk_size <= 0:
        raise ValueError("Sampling budget and chunk_size must be positive.")
    report = {
        "sampling_method": SAMPLING_METHOD,
        "input_complete": False,
        "source_rows_seen": 0,
        "sample_cap_bytes": max_size_bytes,
        "cap_reached": False,
        "truncated": False,
        "per_series": [],
    }
    stats = {}
    metadata_bytes = 0
    columns = None
    timestamp_kind = None
    with tempfile.TemporaryFile(dir=spool_directory) as spool:
        csv_kwargs = {"chunksize": chunk_size}
        if not synthetic_id:
            csv_kwargs["dtype"] = {id_column: "string"}
        for chunk in pd.read_csv(stream, **csv_kwargs):
            if len(chunk) == 0:
                continue
            chunk, kind = _normalize_chunk(
                chunk, id_column, timestamp_column, target, synthetic_id, require_two_columns
            )
            if timestamp_kind is not None and kind != timestamp_kind:
                raise ValueError("Timestamp column mixes datetimes and fractional years across CSV chunks.")
            timestamp_kind = kind
            columns = list(chunk.columns)
            id_index, ts_index = columns.index(id_column), columns.index(timestamp_column)
            for row in chunk.itertuples(index=False, name=None):
                item_id, timestamp = row[id_index], row[ts_index]
                if item_id not in stats:
                    metadata_bytes += sys.getsizeof(item_id) + 1024
                    if metadata_bytes > max_size_bytes:
                        raise ValueError("Series metadata exceeds the sampling budget; increase preset.")
                    stats[item_id] = {
                        "series_id": item_id,
                        "source_rows_seen": 0,
                        "source_timestamp_min": timestamp,
                        "source_timestamp_max": timestamp,
                        "out_of_order": False,
                        "last_timestamp": timestamp,
                        "row_bytes": 0,
                    }
                item = stats[item_id]
                item["source_rows_seen"] += 1
                item["source_timestamp_min"] = min(item["source_timestamp_min"], timestamp)
                item["source_timestamp_max"] = max(item["source_timestamp_max"], timestamp)
                item["out_of_order"] |= timestamp < item["last_timestamp"]
                item["last_timestamp"] = timestamp
                item["row_bytes"] = max(item["row_bytes"], _row_memory_bytes(row))
                report["source_rows_seen"] += 1
                # Only tuples generated in this process are unpickled on replay.
                pickle.dump(row, spool, protocol=pickle.HIGHEST_PROTOCOL)
        report["input_complete"] = True
        if not stats:
            return pd.DataFrame(), report
        quotas = _series_quotas(stats, max_size_bytes)
        heaps = {item_id: [] for item_id in stats}
        retained = {item_id: {} for item_id in stats}
        spool.seek(0)
        for _ in range(report["source_rows_seen"]):
            row = pickle.load(spool)
            item_id, timestamp = row[id_index], row[ts_index]
            rows, heap = retained[item_id], heaps[item_id]
            if timestamp in rows:
                rows[timestamp] = row
            elif len(heap) < quotas[item_id]:
                heapq.heappush(heap, timestamp)
                rows[timestamp] = row
            else:
                report["cap_reached"] = True
                if timestamp > heap[0]:
                    del rows[heapq.heapreplace(heap, timestamp)]
                    rows[timestamp] = row
        buffer_bytes = sum(sum(_row_memory_bytes(row) for row in rows.values()) for rows in retained.values())
        if buffer_bytes > max_size_bytes:
            raise ValueError("Retained time series buffers exceed the sampling budget.")
        # Materialize only the retained observations. Sorting these heaps is bounded
        # by the sample size; the source itself is never sorted in RAM.
        data = pd.DataFrame(
            [rows[timestamp] for item_id, rows in sorted(retained.items()) for timestamp in sorted(rows)],
            columns=columns,
        )
        from .parquet_utils import stringify_mixed_object_columns

        stringify_mixed_object_columns(data)
        result_bytes = int(data.memory_usage(deep=True).sum())
        if result_bytes > max_size_bytes:
            raise ValueError("Retained pandas data exceeds the sampling budget; increase preset.")
        retained_profile = dataframe_series_profile(data, id_column, timestamp_column)
        for item_id, item in sorted(stats.items()):
            report["per_series"].append(
                {
                    "series_id": item_id,
                    "source_rows_seen": item["source_rows_seen"],
                    "source_timestamp_min": _timestamp_value(item["source_timestamp_min"]),
                    "source_timestamp_max": _timestamp_value(item["source_timestamp_max"]),
                    "out_of_order": item["out_of_order"],
                    "row_limit": quotas[item_id],
                    "retained": retained_profile[item_id],
                }
            )
        report.update(
            timestamp_kind=timestamp_kind,
            sampled_rows=len(data),
            sampled_in_memory_bytes=result_bytes,
            retained_buffer_bytes=buffer_bytes,
            series_count=len(stats),
            truncated=report["cap_reached"],
            sampling_applied=report["cap_reached"],
        )
        return data, report
