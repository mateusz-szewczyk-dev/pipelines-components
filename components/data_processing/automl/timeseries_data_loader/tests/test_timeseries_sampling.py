"""Regression and real-pandas checks for timestamp-based per-series sampling."""

import io
import json
import os
import sys
from pathlib import Path
from unittest import mock

import pytest

from ..component import timeseries_data_loader
from .mocked_pandas import MockedDataFrame
from .test_component_unit import (
    _date_from_day_offset,
    _make_test_artifact,
    _mock_boto3_and_pandas,
    _mock_boto3_module,
    _read_csv_rows,
    mocked_env_variables,
)


def _panel_csv(length=300, order="sorted"):
    rows = [f"{item},{_date_from_day_offset(day)},{day},{day}" for item in ("A", "B") for day in range(length)]
    if order == "reversed":
        rows.reverse()
    elif order == "shuffled":
        import random

        random.Random(17).shuffle(rows)
    return "item_id,timestamp,target,feature\n" + "\n".join(rows) + "\n"


@pytest.mark.parametrize("order", ["sorted", "reversed", "shuffled"])
@mock.patch.dict(os.environ, mocked_env_variables, clear=True)
def test_large_panel_retains_latest_for_every_series(tmp_path, order):
    """A capped component retains recent A and B rows and profiles the actual splits."""
    test_artifact = _make_test_artifact(tmp_path)
    with (
        mock.patch.object(MockedDataFrame, "BYTES_PER_ROW", 500_000),
        _mock_boto3_and_pandas(get_object_return={"Body": io.BytesIO(_panel_csv(order=order).encode())}),
    ):
        result = timeseries_data_loader.python_func(
            file_key="panel.csv",
            bucket_name="b",
            workspace_path=str(tmp_path),
            target="target",
            timestamp_column="timestamp",
            id_column="item_id",
            sampled_test_dataset=test_artifact,
        )
    splits = {
        "selection_train": _read_csv_rows(result.models_selection_train_data_path),
        "extra_train": _read_csv_rows(result.extra_train_data_path),
        "test": _read_csv_rows(test_artifact.path),
    }
    status = json.loads((tmp_path / "component_status" / "component_status.json").read_text())
    profile = status["metadata"]["sampling_profile"]
    assert result.sample_config["source_rows_seen"] == 600
    assert result.sample_config["sampled_rows"] == 208
    assert "sampling_profile" not in timeseries_data_loader.component_spec.outputs
    for series in profile["datasets"]["retained"]:
        item = series["series_id"]
        retained = [row for rows in splits.values() for row in rows if row["item_id"] == item]
        assert sorted(int(row["target"]) for row in retained) == list(range(196, 300))
        assert series["rows"] == 104
        assert series["timestamp_min"] == _date_from_day_offset(196)
        assert series["timestamp_max"] == _date_from_day_offset(299)
        assert series["source_rows_seen"] == 300
        assert series["source_timestamp_min"] == _date_from_day_offset(0)
        assert series["source_timestamp_max"] == _date_from_day_offset(299)
        for name, rows in splits.items():
            own_rows = [row for row in rows if row["item_id"] == item]
            stats = next(entry for entry in profile["datasets"][name] if entry["series_id"] == item)
            assert stats["rows"] == len(own_rows)
            assert stats["timestamp_min"] == min(row["timestamp"] for row in own_rows)
            assert stats["timestamp_max"] == max(row["timestamp"] for row in own_rows)


@mock.patch.dict(os.environ, mocked_env_variables, clear=True)
def test_user_test_sampling_keeps_both_series_tails(tmp_path, caplog):
    """External evaluation data uses its own cap and retains every series' latest dates."""

    def get_object(**kwargs):
        is_test = kwargs["Key"] == "test.csv"
        MockedDataFrame.BYTES_PER_ROW = 4_000_000 if is_test else 100
        return {"Body": io.BytesIO(_panel_csv(30 if is_test else 100).encode())}

    test_artifact = _make_test_artifact(tmp_path)
    with (
        mock.patch.object(MockedDataFrame, "BYTES_PER_ROW", 100),
        _mock_boto3_and_pandas(get_object_side_effect=get_object),
    ):
        timeseries_data_loader.python_func(
            file_key="train.csv",
            bucket_name="b",
            workspace_path=str(tmp_path),
            target="target",
            timestamp_column="timestamp",
            id_column="item_id",
            sampled_test_dataset=test_artifact,
            test_data_bucket_name="b",
            test_data_file_key="test.csv",
        )
    rows = _read_csv_rows(test_artifact.path)
    for item in ("A", "B"):
        assert sorted(int(row["target"]) for row in rows if row["item_id"] == item) == list(range(24, 30))
    status = json.loads((tmp_path / "component_status" / "component_status.json").read_text())
    test_profile = status["metadata"]["sampling_profile"]["datasets"]["test"]
    assert len(test_profile) == 2
    assert all(series["source_rows_seen"] == 30 for series in test_profile)
    assert all(series["source_timestamp_min"] == _date_from_day_offset(0) for series in test_profile)
    split_stage = next(stage for stage in status["stages"] if stage["id"] == "split_and_export")
    assert split_stage["metrics"]["truncated"] is True
    assert "newest timestamps per series" in caplog.text
    assert "leading-row prefix" not in caplog.text


@pytest.fixture
def real_pandas(monkeypatch):
    """Use real pandas, including the shared writer after earlier mocked component tests."""
    pd = pytest.importorskip("pandas")
    from kfp_components.components.training.automl.shared import parquet_utils

    monkeypatch.setattr(parquet_utils, "pd", pd)
    return pd


@pytest.mark.parametrize("order", ["sorted", "reversed", "shuffled"])
@pytest.mark.parametrize("chunk_size", [1, 17, 10000])
@mock.patch.dict(os.environ, mocked_env_variables, clear=True)
def test_real_sampler_matches_full_frame_oracle(real_pandas, tmp_path, order, chunk_size):
    """Real pandas/Parquet agree with full group-tail sorting across orders and chunk sizes."""
    pytest.importorskip("pyarrow")
    from pandas.io.parquet import get_engine

    get_engine("pyarrow")
    pd = real_pandas
    read_csv, itertuples = pd.read_csv, pd.DataFrame.itertuples

    class SizedTuple(tuple):
        def __sizeof__(self):
            return 500_000

    def sized_rows(frame, **kwargs):
        return (SizedTuple(row) for row in itertuples(frame, **kwargs))

    def read_chunks(stream, **kwargs):
        kwargs["chunksize"] = chunk_size
        return read_csv(stream, **kwargs)

    # Latest duplicates arrive in a different chunk; keep the last file occurrence.
    body = _panel_csv(order=order) + f"A,{_date_from_day_offset(299)},999,9\n"
    test_artifact = _make_test_artifact(tmp_path)
    with (
        _mock_boto3_module(get_object_return={"Body": io.BytesIO(body.encode())}),
        mock.patch.object(pd, "read_csv", side_effect=read_chunks),
        mock.patch.object(pd.DataFrame, "itertuples", sized_rows),
    ):
        result = timeseries_data_loader.python_func(
            file_key="panel.csv",
            bucket_name="b",
            workspace_path=str(tmp_path),
            target="target",
            timestamp_column="timestamp",
            id_column="item_id",
            sampled_test_dataset=test_artifact,
        )
    actual = (
        pd.concat(
            [
                pd.read_parquet(result.models_selection_train_data_path),
                pd.read_parquet(result.extra_train_data_path),
                pd.read_parquet(test_artifact.path),
            ]
        )
        .sort_values(["item_id", "timestamp"])
        .reset_index(drop=True)
    )
    expected = read_csv(io.StringIO(body))
    expected["timestamp"] = pd.to_datetime(expected["timestamp"])
    expected = (
        expected.drop_duplicates(["item_id", "timestamp"], keep="last")
        .sort_values(["item_id", "timestamp"])
        .groupby("item_id")
        .tail(104)
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(actual, expected)
    assert Path(result.models_selection_train_data_path).read_bytes()[:4] == b"PAR1"


@mock.patch.dict(os.environ, mocked_env_variables, clear=True)
def test_invalid_tail_is_not_hidden_by_budget(tmp_path):
    """A malformed timestamp after the old head cutoff fails instead of being skipped."""
    with (
        mock.patch.object(MockedDataFrame, "BYTES_PER_ROW", 500_000),
        _mock_boto3_and_pandas(
            get_object_return={
                "Body": io.BytesIO((_panel_csv() + "A,invalid,1,1\n").encode()),
            }
        ),
        pytest.raises(ValueError, match="could not be parsed"),
    ):
        timeseries_data_loader.python_func(
            file_key="panel.csv",
            bucket_name="b",
            workspace_path=str(tmp_path),
            target="target",
            timestamp_column="timestamp",
            id_column="item_id",
            sampled_test_dataset=_make_test_artifact(tmp_path),
        )


@mock.patch.dict(os.environ, mocked_env_variables, clear=True)
def test_numeric_ids_keep_file_order_across_chunks(real_pandas, tmp_path):
    """Mixed numeric ID tokens cannot change the identity of integer series IDs."""
    rows = [f"1,{_date_from_day_offset(day)},{day},0" for day in range(150)]
    rows[0] = f"1,{_date_from_day_offset(149)},111,0"
    rows[60] = f"2.5,{_date_from_day_offset(60)},60,0"
    rows.append(f"1,{_date_from_day_offset(149)},999,0")
    actual = _run_real_component(real_pandas, tmp_path, "item_id,timestamp,target,feature\n" + "\n".join(rows), 50)
    own = actual[actual["item_id"] == "1"]
    assert len(own) == 148
    assert own.loc[own["timestamp"] == real_pandas.Timestamp(_date_from_day_offset(149)), "target"].tolist() == [999]
    assert set(actual["item_id"]) == {"1", "2.5"}


@mock.patch.dict(os.environ, mocked_env_variables, clear=True)
def test_duplicate_cost_is_independent_of_chunk_boundary(real_pandas, tmp_path):
    """An overwritten large value contributes to the budget in every chunk layout."""
    rows = [f"A,{_date_from_day_offset(0)},111," + "x" * 500_000]
    rows.append(f"A,{_date_from_day_offset(0)},999,small")
    rows.extend(f"A,{_date_from_day_offset(day)},{day},small" for day in range(1, 400))
    body = "item_id,timestamp,target,feature\n" + "\n".join(rows)
    results = []
    for chunk_size in (1, 10000):
        workspace = tmp_path / str(chunk_size)
        workspace.mkdir()
        results.append(_run_real_component(real_pandas, workspace, body, chunk_size))
    real_pandas.testing.assert_frame_equal(*results)
    assert 100 <= len(results[0]) < 400


@mock.patch.dict(os.environ, mocked_env_variables, clear=True)
def test_full_profile_preserves_all_series_and_actual_unique_counts(tmp_path):
    """A bounded status summary points to full, post-dedup counts and source ranges."""
    rows = [f"S{item:03d},{_date_from_day_offset(day)},{day},0" for item in range(60) for day in (0, 1, 2, 2)]
    artifact = _make_test_artifact(tmp_path)
    with _mock_boto3_and_pandas(
        get_object_return={"Body": io.BytesIO(("item_id,timestamp,target,feature\n" + "\n".join(rows)).encode())}
    ):
        result = timeseries_data_loader.python_func(
            file_key="panel.csv",
            bucket_name="b",
            workspace_path=str(tmp_path),
            target="target",
            timestamp_column="timestamp",
            id_column="item_id",
            sampled_test_dataset=artifact,
        )
    status_dir = tmp_path / "component_status"
    summary = json.loads((status_dir / "component_status.json").read_text())["metadata"]["sampling_profile"]
    full = json.loads((status_dir / summary["profile_file"]).read_text())
    assert result.sample_config["sampled_rows"] == 180
    assert full["source_rows_seen"] == 240
    assert full["input_complete"] is True
    for name, entries in full["datasets"].items():
        assert len(entries) == summary["series_counts"][name] == 60
        assert len(summary["datasets"][name]) == 50
        assert summary["profiles_truncated"][name] is True
    for series in full["datasets"]["retained"]:
        assert series["rows"] == 3
        assert series["source_rows_seen"] == 4
        assert series["source_timestamp_min"] == _date_from_day_offset(0)
        assert series["source_timestamp_max"] == _date_from_day_offset(2)


def _oversized_value_size(value, *args):
    """Simulate an oversized CSV value without allocating a 100 MiB string."""
    if isinstance(value, str) and value == "oversized":
        return 200 * 1024 * 1024
    return _original_getsizeof(value, *args)


_original_getsizeof = sys.getsizeof


@pytest.mark.parametrize("order", ["sorted", "reversed", "shuffled"])
@pytest.mark.parametrize("outlier_day", [0, 100])
@mock.patch.dict(os.environ, mocked_env_variables, clear=True)
def test_oversized_history_keeps_contiguous_latest_tail(tmp_path, order, outlier_day):
    """An unretainable old row acts as a boundary, with no gaps in the exported tail."""
    rows = [
        f"A,{_date_from_day_offset(day)},{day},{'oversized' if day == outlier_day else 'small'}" for day in range(301)
    ]
    if order == "reversed":
        rows.reverse()
    elif order == "shuffled":
        import random

        random.Random(17).shuffle(rows)
    artifact = _make_test_artifact(tmp_path)
    with (
        mock.patch.object(sys, "getsizeof", side_effect=_oversized_value_size),
        _mock_boto3_and_pandas(
            get_object_return={"Body": io.BytesIO(("item_id,timestamp,target,feature\n" + "\n".join(rows)).encode())}
        ),
    ):
        result = timeseries_data_loader.python_func(
            file_key="panel.csv",
            bucket_name="b",
            workspace_path=str(tmp_path),
            target="target",
            timestamp_column="timestamp",
            id_column="item_id",
            sampled_test_dataset=artifact,
        )
    retained = [
        row
        for path in (result.models_selection_train_data_path, result.extra_train_data_path, artifact.path)
        for row in _read_csv_rows(path)
    ]
    assert sorted(int(row["target"]) for row in retained) == list(range(outlier_day + 1, 301))
    status = json.loads((tmp_path / "component_status" / "component_status.json").read_text())
    series = status["metadata"]["sampling_profile"]["datasets"]["retained"][0]
    assert series["source_rows_seen"] == 301
    assert series["oversized_rows_seen"] == 1
    assert series["source_timestamp_min"] == _date_from_day_offset(0)
    assert series["timestamp_min"] == _date_from_day_offset(outlier_day + 1)


@mock.patch.dict(os.environ, mocked_env_variables, clear=True)
def test_oversized_latest_observation_fails_instead_of_returning_old_history(tmp_path):
    """The most recent observation cannot be silently omitted for a series."""
    body = _panel_csv() + f"A,{_date_from_day_offset(300)},300,oversized\n"
    with (
        mock.patch.object(sys, "getsizeof", side_effect=_oversized_value_size),
        _mock_boto3_and_pandas(get_object_return={"Body": io.BytesIO(body.encode())}),
        pytest.raises(ValueError, match="cannot retain the latest observation"),
    ):
        timeseries_data_loader.python_func(
            file_key="panel.csv",
            bucket_name="b",
            workspace_path=str(tmp_path),
            target="target",
            timestamp_column="timestamp",
            id_column="item_id",
            sampled_test_dataset=_make_test_artifact(tmp_path),
        )


@mock.patch.dict(os.environ, mocked_env_variables, clear=True)
def test_last_duplicate_can_replace_an_oversized_marker(tmp_path):
    """An early oversized duplicate does not remove valid, later corrections or history."""
    body = "item_id,timestamp,target,feature\nA," + _date_from_day_offset(150) + ",999,oversized\n"
    body += "\n".join(f"A,{_date_from_day_offset(day)},{day},small" for day in range(300))
    artifact = _make_test_artifact(tmp_path)
    with (
        mock.patch.object(sys, "getsizeof", side_effect=_oversized_value_size),
        _mock_boto3_and_pandas(get_object_return={"Body": io.BytesIO(body.encode())}),
    ):
        result = timeseries_data_loader.python_func(
            file_key="panel.csv",
            bucket_name="b",
            workspace_path=str(tmp_path),
            target="target",
            timestamp_column="timestamp",
            id_column="item_id",
            sampled_test_dataset=artifact,
        )
    rows = [
        row
        for path in (result.models_selection_train_data_path, result.extra_train_data_path, artifact.path)
        for row in _read_csv_rows(path)
    ]
    assert sorted(int(row["target"]) for row in rows) == list(range(300))


@mock.patch.dict(os.environ, mocked_env_variables, clear=True)
def test_real_duplicate_values_keep_last_file_occurrence(real_pandas, tmp_path):
    """Sorting equal timestamps must not change the values of the last file occurrence."""
    import random

    rows = [f"A,{_date_from_day_offset(i % 150)},{i},0" for i in range(600)]
    random.Random(17).shuffle(rows)
    body = "item_id,timestamp,target,feature\n" + "\n".join(rows)
    actual = _run_real_component(real_pandas, tmp_path, body, 17)
    expected = real_pandas.read_csv(io.StringIO(body), dtype={"item_id": "string"})
    expected["timestamp"] = real_pandas.to_datetime(expected["timestamp"])
    expected = expected.drop_duplicates(["item_id", "timestamp"], keep="last").sort_values("timestamp")
    assert actual["target"].tolist() == expected["target"].tolist()


def _run_real_component(pd, tmp_path, body, chunk_size):
    pytest.importorskip("pyarrow")
    read_csv = pd.read_csv

    def read_chunks(stream, **kwargs):
        kwargs["chunksize"] = chunk_size
        return read_csv(stream, **kwargs)

    artifact = _make_test_artifact(tmp_path)
    with (
        _mock_boto3_module(get_object_return={"Body": io.BytesIO(body.encode())}),
        mock.patch.object(pd, "read_csv", side_effect=read_chunks),
    ):
        result = timeseries_data_loader.python_func(
            file_key="panel.csv",
            bucket_name="b",
            workspace_path=str(tmp_path),
            target="target",
            timestamp_column="timestamp",
            id_column="item_id",
            sampled_test_dataset=artifact,
        )
    return (
        pd.concat(
            [
                pd.read_parquet(path)
                for path in (result.models_selection_train_data_path, result.extra_train_data_path, artifact.path)
            ]
        )
        .sort_values(["item_id", "timestamp"])
        .reset_index(drop=True)
    )


@pytest.mark.parametrize("years", [[2000, 2001], [2000.5, 2001.5]])
def test_numeric_timestamp_report_is_json_serializable(real_pandas, years):
    """Integer and fractional year bounds serialize as native JSON numbers."""
    from kfp_components.components.training.automl.shared.timeseries_sampling import sample_timeseries_csv

    body = "item_id,timestamp,target\n" + "\n".join(f"A,{year},1" for year in years)
    _, report = sample_timeseries_csv(
        io.StringIO(body), id_column="item_id", timestamp_column="timestamp", target="target", max_size_bytes=10000
    )
    decoded = json.loads(json.dumps(report))
    assert decoded["per_series"][0]["retained"]["timestamp_min"] == years[0]
    assert decoded["per_series"][0]["retained"]["timestamp_max"] == years[-1]
