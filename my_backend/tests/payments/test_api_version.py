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
