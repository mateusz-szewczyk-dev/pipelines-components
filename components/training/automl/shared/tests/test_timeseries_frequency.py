"""Exercise the fallback on real pandas timestamps without mocking frequency logic."""

import pandas as pd
import pytest

from ..timeseries_frequency import infer_frequency_from_gaps


def panel(*indexes):
    """Build AutoGluon's item/timestamp index without depending on AutoGluon."""
    return pd.concat(
        [
            pd.DataFrame(
                {"target": range(len(index))},
                index=pd.MultiIndex.from_product([[str(item)], index], names=["item_id", "timestamp"]),
            )
            for item, index in enumerate(indexes)
        ]
    )


@pytest.mark.parametrize("freq", ["D", "h", "15min", "100ns", "7D", "MS", "ME", "3MS", "3ME", "12MS"])
def test_infers_fixed_and_calendar_grids_with_gaps(freq):
    """Gaps preserve the data's cadence, values, and timestamps when resampled."""
    grid = pd.date_range("2020-01-01", periods=18, freq=freq)
    data = panel(grid.delete([2, 5]), grid.delete([3, 7])).sample(frac=1, random_state=42)
    original = data.copy(deep=True)
    inferred = infer_frequency_from_gaps(data)
    assert inferred == pd.tseries.frequencies.to_offset(freq).freqstr
    for _, series in data.groupby(level="item_id"):
        values = series.droplevel("item_id").sort_index()
        resampled = values.resample(inferred).mean()
        pd.testing.assert_frame_equal(resampled.loc[values.index], values.astype(float), check_freq=False)
        pd.testing.assert_index_equal(resampled.index, grid.rename("timestamp"))
    pd.testing.assert_frame_equal(data, original)


def test_kaggle_daily_mapping():
    """Three series sharing the supplied 2433 -> 2436 gap resolve to daily data."""
    dates = pd.to_datetime(["1976-08-30", "1976-09-02", "1976-09-03", "1976-09-04", "1976-09-05"])
    assert infer_frequency_from_gaps(panel(dates, dates, dates)) == "D"


def test_short_series_are_validated_against_supported_grid():
    """A short history cannot vote but can share a grid supported by other items."""
    grid = pd.date_range("2020-01-01", periods=12, freq="h")
    assert infer_frequency_from_gaps(panel(grid.delete(2), grid[[0, 3]], grid[[0]])) == "h"


@pytest.mark.parametrize(
    "indexes",
    [
        [pd.date_range("2020-01-01", periods=12, freq="h"), pd.date_range("2020-01-01", periods=12, freq="D")],
        [pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-04"])],  # no dominant step
        [pd.to_datetime(["2020-01-01", "2020-01-03", "2020-01-05", "2020-01-08"])],  # non-multiple gap
        [pd.date_range("2020-01-01", periods=4, freq="D"), pd.to_datetime(["2020-01-02 12:00"])],
        [pd.date_range("2020-01-01 00:15", periods=6, freq="h").delete(2)],  # resampling would move rows
        [pd.date_range("2020-01-01", periods=12, freq="B").delete(3)],  # daily vs business days
        [pd.date_range("2020-01-01", periods=2, freq="h")],  # insufficient evidence
        [pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02"])],
        [pd.to_datetime(["2020-01-01", "2020-01-02", None])],
    ],
)
def test_ambiguous_or_incompatible_data_requires_explicit_frequency(indexes):
    """Reject mixed cadences, ambiguity and any conversion that would move data."""
    with pytest.raises(ValueError, match="Set freq explicitly"):
        infer_frequency_from_gaps(panel(*indexes))


def test_large_expansion_is_rejected_before_constructing_a_grid(monkeypatch):
    """A large gap cannot turn a tiny input into an unbounded resampling job."""
    dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-04", "2026-01-01"])

    def unexpected_grid(*args, **kwargs):
        raise AssertionError("Expansion must be checked before constructing a grid")

    monkeypatch.setattr(pd, "date_range", unexpected_grid)
    with pytest.raises(ValueError, match="expand a series by more than 10x"):
        infer_frequency_from_gaps(panel(dates))
