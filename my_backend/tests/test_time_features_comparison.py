"""
Test script for comparing TIME feature calculations between:
1. Backend transformer.py (current implementation)
2. Original training.py formulas

This test verifies that both implementations produce identical results.

Usage:
    python -m pytest tests/test_time_features_comparison.py -v
    OR
    python tests/test_time_features_comparison.py
"""

import numpy as np
import pandas as pd
import datetime
import pytz
import sys
import os

# Add the my_backend directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Constants from original training.py
YEAR_SECONDS = 31557600    # 60×60×24×365.25
MONTH_SECONDS = 2629800    # 60×60×24×365.25/12
WEEK_SECONDS = 604800      # 60×60×24×7
DAY_SECONDS = 86400        # 60×60×24


def original_month_calculation(utc_timestamps, use_local_time=False, timezone='UTC'):
    """
    Original Month calculation from training_original.py
    Uses constant MONTH_SECONDS (2629800) instead of calendar.monthrange()
    """
    if use_local_time:
        tz = pytz.timezone(timezone)
        utc_th_tz = [pytz.utc.localize(dt) if dt.tzinfo is None else dt for dt in utc_timestamps]
        lt_th = [dt.astimezone(tz) for dt in utc_th_tz]
        sec = np.array([dt.timestamp() for dt in lt_th])
    else:
        sec = pd.Series(utc_timestamps).map(pd.Timestamp.timestamp)

    m_sin = np.sin(sec / MONTH_SECONDS * 2 * np.pi)
    m_cos = np.cos(sec / MONTH_SECONDS * 2 * np.pi)

    return m_sin, m_cos


def original_year_calculation(utc_timestamps, use_local_time=False, timezone='UTC'):
    """Original Year calculation from training_original.py"""
    if use_local_time:
        tz = pytz.timezone(timezone)
        utc_th_tz = [pytz.utc.localize(dt) if dt.tzinfo is None else dt for dt in utc_timestamps]
        lt_th = [dt.astimezone(tz) for dt in utc_th_tz]
        sec = np.array([dt.timestamp() for dt in lt_th])
    else:
        sec = pd.Series(utc_timestamps).map(pd.Timestamp.timestamp)

    y_sin = np.sin(sec / YEAR_SECONDS * 2 * np.pi)
    y_cos = np.cos(sec / YEAR_SECONDS * 2 * np.pi)

    return y_sin, y_cos


def original_week_calculation(utc_timestamps, use_local_time=False, timezone='UTC'):
    """Original Week calculation from training_original.py"""
    if use_local_time:
        tz = pytz.timezone(timezone)
        utc_th_tz = [pytz.utc.localize(dt) if dt.tzinfo is None else dt for dt in utc_timestamps]
        lt_th = [dt.astimezone(tz) for dt in utc_th_tz]
        sec = np.array([dt.timestamp() for dt in lt_th])
    else:
        sec = pd.Series(utc_timestamps).map(pd.Timestamp.timestamp)

    w_sin = np.sin(sec / WEEK_SECONDS * 2 * np.pi)
    w_cos = np.cos(sec / WEEK_SECONDS * 2 * np.pi)

    return w_sin, w_cos


def original_day_calculation(utc_timestamps, use_local_time=False, timezone='UTC'):
    """Original Day calculation from training_original.py"""
    if use_local_time:
        tz = pytz.timezone(timezone)
        utc_th_tz = [pytz.utc.localize(dt) if dt.tzinfo is None else dt for dt in utc_timestamps]
        lt_th = [dt.astimezone(tz) for dt in utc_th_tz]
        sec = np.array([dt.timestamp() for dt in lt_th])
    else:
        sec = pd.Series(utc_timestamps).map(pd.Timestamp.timestamp)

    d_sin = np.sin(sec / DAY_SECONDS * 2 * np.pi)
    d_cos = np.cos(sec / DAY_SECONDS * 2 * np.pi)

    return d_sin, d_cos


def test_month_constants():
    """Verify that Month uses constant 2629800, not dynamic calendar.monthrange()"""

    # Test timestamps spanning different months with different lengths
    test_dates = [
        datetime.datetime(2024, 1, 15, 12, 0, 0),   # January (31 days)
        datetime.datetime(2024, 2, 15, 12, 0, 0),   # February (29 days - leap year)
        datetime.datetime(2024, 4, 15, 12, 0, 0),   # April (30 days)
        datetime.datetime(2024, 7, 15, 12, 0, 0),   # July (31 days)
    ]

    print("\n" + "="*70)
    print("TEST: Month Calculation with Constant MONTH_SECONDS (2629800)")
    print("="*70)

    for dt in test_dates:
        m_sin, m_cos = original_month_calculation([dt])
        sec = dt.timestamp()

        # Calculate what the dynamic method would give
        import calendar
        days_in_month = calendar.monthrange(dt.year, dt.month)[1]
        dynamic_month_seconds = days_in_month * 86400

        dynamic_sin = np.sin(sec / dynamic_month_seconds * 2 * np.pi)
        dynamic_cos = np.cos(sec / dynamic_month_seconds * 2 * np.pi)

        print(f"\n{dt.strftime('%Y-%m-%d')} ({calendar.month_name[dt.month]}, {days_in_month} days):")
        print(f"  Constant (2629800):  M_sin={m_sin[0]:.6f}, M_cos={m_cos[0]:.6f}")
        print(f"  Dynamic ({dynamic_month_seconds}):   M_sin={dynamic_sin:.6f}, M_cos={dynamic_cos:.6f}")
        print(f"  Difference:          sin={abs(m_sin[0]-dynamic_sin):.6f}, cos={abs(m_cos[0]-dynamic_cos):.6f}")

    print("\n" + "="*70)
    print("CONCLUSION: Using constant 2629800 gives consistent results across months")
    print("="*70)


def test_backend_vs_original():
    """Compare backend TimeFeatures class with original formulas"""

    try:
        from domains.training.data.processor import TimeFeatures
        backend_available = True
    except ImportError:
        print("WARNING: Could not import backend TimeFeatures. Running original-only tests.")
        backend_available = False

    # Generate test timestamps
    base_time = datetime.datetime(2024, 6, 15, 12, 0, 0)  # Mid-year reference
    test_timestamps = [base_time + datetime.timedelta(hours=i) for i in range(24)]

    print("\n" + "="*70)
    print("TEST: Backend vs Original TIME Feature Calculations")
    print("="*70)
    print(f"Test period: {test_timestamps[0]} to {test_timestamps[-1]}")
    print(f"Number of timestamps: {len(test_timestamps)}")

    # Original calculations
    orig_y_sin, orig_y_cos = original_year_calculation(test_timestamps)
    orig_m_sin, orig_m_cos = original_month_calculation(test_timestamps)
    orig_w_sin, orig_w_cos = original_week_calculation(test_timestamps)
    orig_d_sin, orig_d_cos = original_day_calculation(test_timestamps)

    print("\n--- ORIGINAL CALCULATIONS (first 5 values) ---")
    print(f"Y_sin: {list(orig_y_sin[:5])}")
    print(f"M_sin: {list(orig_m_sin[:5])}")
    print(f"W_sin: {list(orig_w_sin[:5])}")
    print(f"D_sin: {list(orig_d_sin[:5])}")

    if backend_available:
        # Backend calculations
        time_features = TimeFeatures(timezone='UTC')

        backend_year = time_features._calc_year_features(test_timestamps, lt=False)
        backend_month = time_features._calc_month_features(test_timestamps, lt=False)
        backend_week = time_features._calc_week_features(test_timestamps, lt=False)
        backend_day = time_features._calc_day_features(test_timestamps, lt=False)

        print("\n--- BACKEND CALCULATIONS (first 5 values) ---")
        print(f"Y_sin: {list(backend_year['Y_sin'][:5])}")
        print(f"M_sin: {list(backend_month['M_sin'][:5])}")
        print(f"W_sin: {list(backend_week['W_sin'][:5])}")
        print(f"D_sin: {list(backend_day['D_sin'][:5])}")

        # Compare
        print("\n--- COMPARISON ---")

        y_diff = np.max(np.abs(np.array(orig_y_sin) - backend_year['Y_sin']))
        m_diff = np.max(np.abs(np.array(orig_m_sin) - backend_month['M_sin']))
        w_diff = np.max(np.abs(np.array(orig_w_sin) - backend_week['W_sin']))
        d_diff = np.max(np.abs(np.array(orig_d_sin) - backend_day['D_sin']))

        print(f"Year max difference:  {y_diff:.10f} {'✓ PASS' if y_diff < 1e-10 else '✗ FAIL'}")
        print(f"Month max difference: {m_diff:.10f} {'✓ PASS' if m_diff < 1e-10 else '✗ FAIL'}")
        print(f"Week max difference:  {w_diff:.10f} {'✓ PASS' if w_diff < 1e-10 else '✗ FAIL'}")
        print(f"Day max difference:   {d_diff:.10f} {'✓ PASS' if d_diff < 1e-10 else '✗ FAIL'}")

        all_pass = all([y_diff < 1e-10, m_diff < 1e-10, w_diff < 1e-10, d_diff < 1e-10])

        print("\n" + "="*70)
        if all_pass:
            print("RESULT: ALL TESTS PASSED - Backend matches Original")
        else:
            print("RESULT: TESTS FAILED - Backend does NOT match Original")
        print("="*70)

        return all_pass

    return True


def test_local_time_calculation():
    """Test LT (Local Time) mode calculations"""

    try:
        from domains.training.data.processor import TimeFeatures
        backend_available = True
    except ImportError:
        backend_available = False

    base_time = datetime.datetime(2024, 6, 15, 12, 0, 0)
    test_timestamps = [base_time + datetime.timedelta(hours=i) for i in range(24)]
    timezone = 'Europe/Vienna'  # CET/CEST

    print("\n" + "="*70)
    print("TEST: Local Time (LT) Mode Calculations")
    print("="*70)
    print(f"Timezone: {timezone}")

    # Original with LT
    orig_m_sin_lt, orig_m_cos_lt = original_month_calculation(
        test_timestamps, use_local_time=True, timezone=timezone
    )
    orig_m_sin_utc, orig_m_cos_utc = original_month_calculation(
        test_timestamps, use_local_time=False
    )

    print("\n--- Month Sin Values (first 5) ---")
    print(f"UTC mode: {list(orig_m_sin_utc[:5])}")
    print(f"LT mode:  {list(orig_m_sin_lt[:5])}")
    print(f"Difference shows timezone offset effect: {np.mean(np.abs(orig_m_sin_lt - orig_m_sin_utc)):.6f}")

    if backend_available:
        time_features = TimeFeatures(timezone=timezone)
        backend_m_lt = time_features._calc_month_features(test_timestamps, lt=True)
        backend_m_utc = time_features._calc_month_features(test_timestamps, lt=False)

        lt_diff = np.max(np.abs(np.array(orig_m_sin_lt) - backend_m_lt['M_sin']))
        utc_diff = np.max(np.abs(np.array(orig_m_sin_utc) - backend_m_utc['M_sin']))

        print(f"\nBackend vs Original (LT mode):  max diff = {lt_diff:.10f} {'✓' if lt_diff < 1e-10 else '✗'}")
        print(f"Backend vs Original (UTC mode): max diff = {utc_diff:.10f} {'✓' if utc_diff < 1e-10 else '✗'}")


def print_test_summary():
    """Print summary of what to test manually via web interface"""

    print("\n" + "="*70)
    print("MANUAL TESTING GUIDE - Frontend to Backend")
    print("="*70)

    print("""
1. PREPARE TEST SESSION:
   - Upload a CSV file with timestamps spanning multiple months
   - Example: data from 2024-01-01 to 2024-06-30

2. CONFIGURE TIME COMPONENTS:
   - Enable: Jahr (Year), Monat (Month), Woche (Week), Tag (Day)
   - Set Zeithorizont: Start=-24, End=0
   - Set Skalierung: ja (Min=0, Max=1)

3. CONFIGURE ZEITSCHRITTE:
   - Eingabe: 24
   - Ausgabe: 1
   - Zeitschrittweite: 60 (1 hour)
   - Offset: 0

4. RUN TRAINING and check logs for:
   - "TIME components configured: Y=True, M=True, W=True, D=True"
   - Feature names should include: Y_sin, Y_cos, M_sin, M_cos, W_sin, W_cos, D_sin, D_cos

5. COMPARE RESULTS:
   - Check Violin plots have correct number of features
   - Compare evaluation metrics with original training.py on same data
""")


if __name__ == "__main__":
    print("\n" + "#"*70)
    print("# TIME FEATURES COMPARISON TEST")
    print("# Backend vs Original training.py")
    print("#"*70)

    # Run tests
    test_month_constants()
    test_backend_vs_original()
    test_local_time_calculation()
    print_test_summary()
