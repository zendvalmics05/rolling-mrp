"""Unit tests for Phase 6.1: External Forecast API Client, Caching, and Fallback."""

import os
import shutil
import pytest

from src.integration.forecast_api import (
    ForecastApiClient,
    ForecastResponse,
    PeriodForecastItem,
)


@pytest.fixture
def temp_cache_dir(tmp_path):
    cache_path = tmp_path / "forecast_cache"
    cache_path.mkdir(parents=True, exist_ok=True)
    yield str(cache_path)
    shutil.rmtree(str(cache_path), ignore_errors=True)


def test_forecast_api_fetch_and_schema(temp_cache_dir):
    """Verify standard 12-week forecast retrieval and JSON schema conformity."""
    client = ForecastApiClient(cache_dir=temp_cache_dir)
    items = ["FG-01", "FG-02"]

    response = client.fetch_rolling_forecast(
        item_ids=items,
        start_period=1,
        horizon_length=12,
    )

    assert response.status == "SUCCESS"
    assert response.source == "api"
    assert response.effective_period_start == 1
    assert response.horizon_length == 12
    # 2 items * 12 periods = 24 forecast records
    assert len(response.forecasts) == 24

    for f in response.forecasts:
        assert f.item_id in items
        assert 1 <= f.period <= 12
        assert f.quantity > 0.0
        # Confidence interval consistency
        assert f.confidence_interval_lower <= f.quantity <= f.confidence_interval_upper


def test_forecast_caching_and_hit(temp_cache_dir):
    """Verify that second call retrieves from cache without calling the provider."""
    call_count = 0

    def custom_mock_provider(item_ids, start, horizon):
        nonlocal call_count
        call_count += 1
        return [
            PeriodForecastItem(
                item_id=item_ids[0],
                period=start + h,
                quantity=100.0,
                confidence_interval_lower=90.0,
                confidence_interval_upper=110.0,
            )
            for h in range(horizon)
        ]

    client = ForecastApiClient(
        cache_dir=temp_cache_dir,
        mock_provider=custom_mock_provider,
        cache_ttl_seconds=3600,
    )

    # Call 1: should call provider
    resp1 = client.fetch_rolling_forecast(["FG-01"], start_period=1, horizon_length=12)
    assert resp1.source == "api"
    assert call_count == 1

    # Call 2: should hit in-memory / disk cache
    resp2 = client.fetch_rolling_forecast(["FG-01"], start_period=1, horizon_length=12)
    assert resp2.source == "cache"
    assert call_count == 1  # Provider not called again

    # Verify disk persistence
    files = os.listdir(temp_cache_dir)
    assert len(files) == 1
    assert files[0].endswith(".json")


def test_forecast_api_failover_fallback(temp_cache_dir):
    """Verify automatic failover fallback when API call fails."""
    client = ForecastApiClient(cache_dir=temp_cache_dir)

    # Call with simulated failure and no prior cache -> generates baseline fallback
    resp_fallback = client.fetch_rolling_forecast(
        item_ids=["FG-01"],
        start_period=1,
        horizon_length=12,
        simulate_api_failure=True,
    )

    assert resp_fallback.status == "FALLBACK_GENERATED"
    assert resp_fallback.source == "fallback_simulation"
    assert len(resp_fallback.forecasts) == 12

    # Now succeed once to populate cache
    resp_success = client.fetch_rolling_forecast(
        item_ids=["FG-01"],
        start_period=1,
        horizon_length=12,
        force_refresh=True,
    )
    assert resp_success.status == "SUCCESS"

    # Now simulate failure with populated cache -> returns cached fallback
    client.memory_cache.clear()  # Clear memory to force disk check
    resp_cached_fallback = client.fetch_rolling_forecast(
        item_ids=["FG-01"],
        start_period=1,
        horizon_length=12,
        force_refresh=True,
        simulate_api_failure=True,
    )

    assert resp_cached_fallback.status == "FALLBACK_CACHED"
    assert resp_cached_fallback.source == "cache"
    assert len(resp_cached_fallback.forecasts) == 12


def test_mrp_gross_requirements_conversion(temp_cache_dir):
    """Verify conversion of ForecastResponse into MRP engine gross requirements dict."""
    client = ForecastApiClient(cache_dir=temp_cache_dir)
    resp = client.fetch_rolling_forecast(["FG-01", "FG-02"], start_period=1, horizon_length=12)

    mrp_gross = resp.to_mrp_gross_requirements()
    assert "FG-01" in mrp_gross
    assert "FG-02" in mrp_gross
    assert len(mrp_gross["FG-01"]) == 12
    assert len(mrp_gross["FG-02"]) == 12
    assert all(isinstance(q, (int, float)) for q in mrp_gross["FG-01"])

