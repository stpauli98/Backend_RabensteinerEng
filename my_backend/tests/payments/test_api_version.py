"""Pin the Stripe API version explicitly."""
import stripe


def test_stripe_api_version_is_pinned():
    # Force a fresh import of the module under test so the side-effect at
    # module load runs against this test's interpreter state.
    import importlib
    import shared.payments.stripe as payments
    importlib.reload(payments)

    assert stripe.api_version == '2025-10-29.clover', (
        f"Expected stripe.api_version pinned to '2025-10-29.clover', "
        f"got {stripe.api_version!r}"
    )


def test_stripe_logger_is_suppressed_to_warning():
    """The 'stripe' SDK logger emits an INFO line for every API call (we saw
    this in local Docker logs while testing). Ensure we suppress to WARNING
    in line with the existing pattern for other chatty third-party loggers.
    """
    import logging
    # Triggers app_factory module load, which sets the third-party levels.
    from core.app_factory import create_app  # noqa: F401

    assert logging.getLogger('stripe').level == logging.WARNING
