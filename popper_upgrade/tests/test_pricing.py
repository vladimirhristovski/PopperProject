import pytest

from popper_upgrade.config import EUR_USD_RATE, EUR_USD_RATE_DATE
from popper_upgrade.serving.pricing import check_pricing, cost_usd, eur_to_usd, usage_parts, usd_to_eur
from popper_upgrade.serving.usage import UsageCounter
from popper_upgrade.tests.test_models import _registry_models

SOL = {"input": 2, "cache_read": 0.2, "output": 10}
SONNET = {"input": 2, "cache_read": 0.1, "cache_write": 2.5, "output": 10}


def test_rate_is_configured_with_its_date():
    assert EUR_USD_RATE == 1.1193
    assert EUR_USD_RATE_DATE == "2026-10-23"


def test_eur_conversion_round_trips_at_the_configured_rate():
    assert eur_to_usd(20) == pytest.approx(22.386)
    assert usd_to_eur(22.386) == pytest.approx(20)
    assert eur_to_usd(10, rate=2.0) == 20
    assert usd_to_eur(10, rate=2.0) == 5


def test_anthropic_style_usage_splits_input_cache_read_and_write():
    parts = usage_parts(
        {
            "prompt_tokens": 1300,
            "completion_tokens": 50,
            "cache_creation_input_tokens": 200,
            "cache_read_input_tokens": 1000,
        }
    )

    assert parts == {"input": 100, "cache_read": 1000, "cache_write": 200, "output": 50}
    expected = (100 * 2 + 1000 * 0.1 + 200 * 2.5 + 50 * 10) / 1e6
    assert cost_usd(parts, SONNET) == pytest.approx(expected)


def test_openai_style_usage_reads_cached_tokens_from_the_details():
    parts = usage_parts(
        {"prompt_tokens": 1000, "completion_tokens": 100, "prompt_tokens_details": {"cached_tokens": 800}}
    )

    assert parts == {"input": 200, "cache_read": 800, "cache_write": 0, "output": 100}
    assert cost_usd(parts, SOL) == pytest.approx((200 * 2 + 800 * 0.2 + 100 * 10) / 1e6)


def test_missing_cache_prices_fall_back_to_the_input_price():
    parts = {"input": 0, "cache_read": 1000, "cache_write": 1000, "output": 0}

    assert cost_usd(parts, {"input": 2, "output": 10}) == pytest.approx(4000 / 1e6)


def test_no_pricing_or_usage_gives_no_cost():
    assert cost_usd({"input": 1, "cache_read": 0, "cache_write": 0, "output": 1}, None) is None
    assert usage_parts(None) is None
    assert usage_parts({}) is None
    assert usage_parts({"total_tokens": 3}) is None


@pytest.mark.parametrize(
    ("pricing", "message"),
    [
        ([1, 2], "object"),
        ({"input": 1}, "needs output"),
        ({"input": 1, "output": 1, "extra": 2}, "unknown pricing fields"),
        ({"input": -1, "output": 1}, "non-negative"),
        ({"input": "2", "output": 1}, "non-negative"),
    ],
)
def test_pricing_is_validated(pricing, message):
    with pytest.raises(ValueError, match=message):
        check_pricing(pricing)


def test_registry_prices_are_valid_and_match_the_locked_models():
    models = _registry_models()

    assert check_pricing(models["gpt-6-sol"]["pricing"]) == SOL
    assert check_pricing(models["claude-sonnet-5-5"]["pricing"]) == SONNET


def test_counter_accumulates_cost_per_run_and_in_total():
    counter = UsageCounter(SONNET)

    counter.add({"prompt_tokens": 1000, "completion_tokens": 100, "cache_read_input_tokens": 500})
    counter.add({"prompt_tokens": 2000, "completion_tokens": 100, "cache_creation_input_tokens": 1000})
    first = (500 * 2 + 500 * 0.1 + 100 * 10) / 1e6
    second = (1000 * 2 + 1000 * 2.5 + 100 * 10) / 1e6

    assert counter.cost_usd() == pytest.approx(first + second)
    assert counter.cache_snapshot() == (500, 1000)
    counter.reset()
    assert counter.cost_usd() is None
    assert counter.cache_snapshot() == (None, None)
    counter.add({"prompt_tokens": 1000, "completion_tokens": 0})
    assert counter.cost_usd() == pytest.approx(2000 / 1e6)
    assert counter.total_cost_usd() == pytest.approx(first + second + 2000 / 1e6)


def test_counter_without_pricing_has_no_cost():
    counter = UsageCounter()

    counter.add({"prompt_tokens": 1000, "completion_tokens": 100})

    assert counter.cost_usd() is None
    assert counter.total_cost_usd() == 0.0
