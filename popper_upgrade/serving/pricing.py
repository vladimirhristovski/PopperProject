import json

from popper_upgrade.config import EUR_USD_RATE, WORST_CASE_BYTES_PER_TOKEN, WORST_CASE_OUTPUT_TOKENS

PER_MILLION = 1_000_000
PRICE_FIELDS = ("input", "cache_read", "cache_write", "output")
REQUIRED_PRICES = ("input", "output")


def check_pricing(pricing):
    if not isinstance(pricing, dict):
        raise ValueError("pricing must be an object of USD prices per 1M tokens")
    unknown = set(pricing) - set(PRICE_FIELDS)
    if unknown:
        raise ValueError(f"unknown pricing fields: {sorted(unknown)}")
    for field in REQUIRED_PRICES:
        if field not in pricing:
            raise ValueError(f"pricing needs {field}")
    for field, value in pricing.items():
        if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
            raise ValueError(f"pricing {field} must be a non-negative number")
    return pricing


def usage_parts(usage):
    if not isinstance(usage, dict):
        return None
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if not isinstance(prompt, int) and not isinstance(completion, int):
        return None
    details = usage.get("prompt_tokens_details")
    cached = details.get("cached_tokens") if isinstance(details, dict) else None
    read = usage.get("cache_read_input_tokens", cached)
    write = usage.get("cache_creation_input_tokens")
    read = read if isinstance(read, int) else 0
    write = write if isinstance(write, int) else 0
    prompt = prompt if isinstance(prompt, int) else 0
    return {
        "input": max(prompt - read - write, 0),
        "cache_read": read,
        "cache_write": write,
        "output": completion if isinstance(completion, int) else 0,
    }


def cost_usd(parts, pricing):
    if not pricing or parts is None:
        return None
    total = 0.0
    for field, tokens in parts.items():
        price = pricing.get(field)
        if price is None:
            price = pricing["input"] if field in ("cache_read", "cache_write") else 0.0
        total += tokens * price / PER_MILLION
    return total


OUTPUT_CAP_KEYS = ("max_tokens", "max_completion_tokens", "max_output_tokens")


def output_cap(body):
    try:
        data = json.loads(body)
    except (TypeError, ValueError):
        return WORST_CASE_OUTPUT_TOKENS
    if isinstance(data, dict):
        for key in OUTPUT_CAP_KEYS:
            value = data.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
    return WORST_CASE_OUTPUT_TOKENS


def worst_case_cost(body, pricing):
    prompt = len(body or b"") // WORST_CASE_BYTES_PER_TOKEN + 1
    parts = {"input": prompt, "cache_read": 0, "cache_write": 0, "output": output_cap(body)}
    return cost_usd(parts, pricing)


def usd_to_eur(usd, rate=EUR_USD_RATE):
    return usd / rate


def eur_to_usd(eur, rate=EUR_USD_RATE):
    return eur * rate
