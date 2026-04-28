"""Subscription current_period_start/end resolution across Stripe API versions."""
from types import SimpleNamespace

from shared.payments.stripe import _subscription_period_dates


def _sub(top_start=None, top_end=None, item_start=None, item_end=None, with_items=True):
    """Build a fake Stripe Subscription object using SimpleNamespace.

    The real stripe.StripeObject responds to attribute access; SimpleNamespace
    is a sufficient stand-in for the helper, which uses getattr().
    """
    item = SimpleNamespace(current_period_start=item_start, current_period_end=item_end)
    items = SimpleNamespace(data=[item]) if with_items else None
    return SimpleNamespace(
        current_period_start=top_start,
        current_period_end=top_end,
        items=items,
    )


def test_resolves_from_top_level_when_present():
    sub = _sub(top_start=1000, top_end=2000, item_start=999, item_end=1999)
    assert _subscription_period_dates(sub) == (1000, 2000)


def test_falls_back_to_items_when_top_level_is_none():
    """The 2025-10-29.clover behavior: top-level fields are None,
    items[0] holds the actual values."""
    sub = _sub(top_start=None, top_end=None, item_start=1000, item_end=2000)
    assert _subscription_period_dates(sub) == (1000, 2000)


def test_partial_top_level_fills_missing_from_items():
    sub = _sub(top_start=1000, top_end=None, item_start=None, item_end=2000)
    assert _subscription_period_dates(sub) == (1000, 2000)


def test_returns_none_when_neither_location_has_value():
    sub = _sub(top_start=None, top_end=None, item_start=None, item_end=None)
    assert _subscription_period_dates(sub) == (None, None)


def test_returns_none_when_subscription_has_no_items_attribute():
    sub = _sub(top_start=None, top_end=None, with_items=False)
    assert _subscription_period_dates(sub) == (None, None)


def test_handles_empty_items_data_list():
    sub = SimpleNamespace(
        current_period_start=None,
        current_period_end=None,
        items=SimpleNamespace(data=[]),
    )
    assert _subscription_period_dates(sub) == (None, None)


def test_resolves_correctly_when_subscription_is_a_dict_subclass():
    """The real stripe.Subscription inherits from dict and bracket access
    is required to get the JSON 'items' field — attribute access returns
    the dict's built-in items() method instead. Mimic that here so the
    helper is exercised against the same shape it sees in production.
    """
    # A dict subclass: attribute access for dict-method names returns the
    # method (the bug). Bracket access returns the JSON value (the workaround).
    sub = {
        'current_period_start': None,
        'current_period_end': None,
        'items': {
            'data': [
                {'current_period_start': 1234567890, 'current_period_end': 1237159890}
            ]
        },
    }
    assert _subscription_period_dates(sub) == (1234567890, 1237159890)
