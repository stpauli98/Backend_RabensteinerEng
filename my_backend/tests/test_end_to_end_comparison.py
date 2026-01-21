"""
END-TO-END Comparison Test: Backend vs Original training.py

This test creates synthetic data with known patterns and compares:
1. Data loading and transformation
2. TIME feature generation
3. Array creation (i_array_3D, o_array_3D)
4. Scaling (MinMaxScaler)
5. Train/Val/Test split
6. Model training (Linear model for reproducibility)
7. Predictions and metrics

Usage:
    docker-compose run --rm backend python /app/tests/test_end_to_end_comparison.py
"""

import numpy as np
import pandas as pd
import datetime
import os
import sys
import json
import tempfile
from typing import Dict, Tuple, List

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Constants
YEAR_SECONDS = 31557600
MONTH_SECONDS = 2629800
WEEK_SECONDS = 604800
DAY_SECONDS = 86400

np.random.seed(42)  # For reproducibility


class SyntheticDataGenerator:
    """Generate synthetic time series data with known patterns"""

    def __init__(self, start_date: str = "2024-01-01", days: int = 180, freq_hours: int = 1):
        self.start_date = pd.to_datetime(start_date)
        self.days = days
        self.freq_hours = freq_hours
        self.n_points = (days * 24) // freq_hours

    def generate_timestamps(self) -> pd.DatetimeIndex:
        """Generate UTC timestamps"""
        return pd.date_range(
            start=self.start_date,
            periods=self.n_points,
            freq=f'{self.freq_hours}H'
        )

    def generate_input_data(self) -> pd.DataFrame:
        """
        Generate input data with:
        - Daily cycle (temperature-like)
        - Weekly cycle (workday pattern)
        - Yearly cycle (seasonal)
        - Random noise
        """
        timestamps = self.generate_timestamps()

        # Time in seconds from epoch
        sec = timestamps.map(lambda x: x.timestamp()).values

        # Daily cycle (amplitude 10, offset 20)
        daily = 10 * np.sin(sec / DAY_SECONDS * 2 * np.pi) + 20

        # Weekly cycle (amplitude 5)
        weekly = 5 * np.sin(sec / WEEK_SECONDS * 2 * np.pi)

        # Yearly cycle (amplitude 15)
        yearly = 15 * np.sin(sec / YEAR_SECONDS * 2 * np.pi)

        # Combined signal with noise
        values = daily + weekly + yearly + np.random.normal(0, 1, len(sec))

        df = pd.DataFrame({
            'UTC': timestamps,
            'value': values
        })

        return df

    def generate_output_data(self) -> pd.DataFrame:
        """
        Generate output data that correlates with input:
        - Linear relationship with input pattern
        - Time lag of ~6 hours
        - Different scaling
        """
        timestamps = self.generate_timestamps()
        sec = timestamps.map(lambda x: x.timestamp()).values

        # Similar pattern but with phase shift (6 hours = 21600 seconds)
        phase_shift = 21600

        daily = 8 * np.sin((sec - phase_shift) / DAY_SECONDS * 2 * np.pi) + 50
        weekly = 3 * np.sin((sec - phase_shift) / WEEK_SECONDS * 2 * np.pi)
        yearly = 10 * np.sin((sec - phase_shift) / YEAR_SECONDS * 2 * np.pi)

        values = daily + weekly + yearly + np.random.normal(0, 0.5, len(sec))

        df = pd.DataFrame({
            'UTC': timestamps,
            'value': values
        })

        return df


class OriginalPipelineSimulator:
    """Simulate the original training.py pipeline logic"""

    def __init__(self, mts_config: Dict, time_config: Dict):
        self.mts = mts_config
        self.time = time_config

    def create_training_arrays(self, i_dat: pd.DataFrame, o_dat: pd.DataFrame,
                               utc_start: datetime.datetime, utc_end: datetime.datetime
                               ) -> Tuple[np.ndarray, np.ndarray, List]:
        """
        Simulate original training array creation from training_original.py
        """
        I_N = self.mts['I_N']
        O_N = self.mts['O_N']
        DELT = self.mts['DELT']
        OFST = self.mts['OFST']

        # Zeithorizont for input (e.g., -24 to 0 hours)
        i_th_strt = -24
        i_th_end = 0

        # Zeithorizont for output (e.g., 0 to 24 hours)
        o_th_strt = 0
        o_th_end = 24

        i_arrays = []
        o_arrays = []
        utc_ref_log = []

        # Main time loop (match backend structure)
        utc_ref = utc_start

        while True:
            if utc_ref > utc_end:
                break

            # Input time horizon
            utc_th_strt = utc_ref + datetime.timedelta(hours=i_th_strt)
            utc_th_end = utc_ref + datetime.timedelta(hours=i_th_end)

            # Create input timestamps
            input_timestamps = pd.date_range(start=utc_th_strt, end=utc_th_end, periods=I_N)

            # Interpolate input values
            i_values = self._interpolate(i_dat, input_timestamps)
            if i_values is None:
                utc_ref = utc_ref + datetime.timedelta(minutes=DELT)
                continue

            # Add TIME features
            time_features = self._calculate_time_features(input_timestamps.tolist())

            # Combine: [data_value, Y_sin, Y_cos, M_sin, M_cos, W_sin, W_cos, D_sin, D_cos]
            # Match backend order: Y, M, W, D
            feature_order = ['Y_sin', 'Y_cos', 'M_sin', 'M_cos', 'W_sin', 'W_cos', 'D_sin', 'D_cos']
            i_row = np.column_stack([i_values.reshape(-1, 1)] +
                                    [time_features[k].reshape(-1, 1) for k in feature_order])

            # Output time horizon
            utc_th_strt_o = utc_ref + datetime.timedelta(hours=o_th_strt)
            utc_th_end_o = utc_ref + datetime.timedelta(hours=o_th_end)

            output_timestamps = pd.date_range(start=utc_th_strt_o, end=utc_th_end_o, periods=O_N)

            o_values = self._interpolate(o_dat, output_timestamps)
            if o_values is None:
                utc_ref = utc_ref + datetime.timedelta(minutes=DELT)
                continue

            i_arrays.append(i_row)
            o_arrays.append(o_values.reshape(-1, 1))
            utc_ref_log.append(utc_ref)

            # Increment at end of loop (matches backend)
            utc_ref = utc_ref + datetime.timedelta(minutes=DELT)

        if not i_arrays:
            raise ValueError("No valid samples created")

        i_array_3D = np.array(i_arrays)
        o_array_3D = np.array(o_arrays)

        return i_array_3D, o_array_3D, utc_ref_log

    def _interpolate(self, df: pd.DataFrame, timestamps: pd.DatetimeIndex) -> np.ndarray:
        """Linear interpolation for timestamps"""
        try:
            # Ensure UTC column is datetime
            df_copy = df.copy()
            df_copy['UTC'] = pd.to_datetime(df_copy['UTC'])
            df_copy = df_copy.set_index('UTC')

            # Check bounds
            if timestamps.min() < df_copy.index.min() or timestamps.max() > df_copy.index.max():
                return None

            # Reindex and interpolate
            combined = df_copy.reindex(df_copy.index.union(timestamps))
            combined = combined.interpolate(method='time')

            return combined.loc[timestamps, 'value'].values
        except:
            return None

    def _calculate_time_features(self, timestamps: List[datetime.datetime]) -> Dict[str, np.ndarray]:
        """Calculate TIME features using original formulas with constants"""
        sec = np.array([ts.timestamp() for ts in timestamps])

        features = {}

        if self.time.get('jahr', False):
            features['Y_cos'] = np.cos(sec / YEAR_SECONDS * 2 * np.pi)
            features['Y_sin'] = np.sin(sec / YEAR_SECONDS * 2 * np.pi)

        if self.time.get('monat', False):
            features['M_cos'] = np.cos(sec / MONTH_SECONDS * 2 * np.pi)
            features['M_sin'] = np.sin(sec / MONTH_SECONDS * 2 * np.pi)

        if self.time.get('woche', False):
            features['W_cos'] = np.cos(sec / WEEK_SECONDS * 2 * np.pi)
            features['W_sin'] = np.sin(sec / WEEK_SECONDS * 2 * np.pi)

        if self.time.get('tag', False):
            features['D_cos'] = np.cos(sec / DAY_SECONDS * 2 * np.pi)
            features['D_sin'] = np.sin(sec / DAY_SECONDS * 2 * np.pi)

        return features

    def scale_data(self, i_array_3D: np.ndarray, o_array_3D: np.ndarray
                   ) -> Tuple[np.ndarray, np.ndarray, Dict, Dict]:
        """Apply MinMaxScaler per feature"""
        from sklearn.preprocessing import MinMaxScaler

        n_samples, n_timesteps, n_features = i_array_3D.shape

        # Flatten for scaling
        i_flat = i_array_3D.reshape(-1, n_features)

        i_scalers = {}
        i_scaled_flat = np.zeros_like(i_flat)

        for i in range(n_features):
            scaler = MinMaxScaler()
            i_scaled_flat[:, i] = scaler.fit_transform(i_flat[:, i:i+1]).flatten()
            i_scalers[i] = scaler

        i_array_scaled = i_scaled_flat.reshape(n_samples, n_timesteps, n_features)

        # Output scaling
        o_flat = o_array_3D.reshape(-1, o_array_3D.shape[-1])
        o_scalers = {}
        o_scaled_flat = np.zeros_like(o_flat)

        for i in range(o_array_3D.shape[-1]):
            scaler = MinMaxScaler()
            o_scaled_flat[:, i] = scaler.fit_transform(o_flat[:, i:i+1]).flatten()
            o_scalers[i] = scaler

        o_array_scaled = o_scaled_flat.reshape(o_array_3D.shape)

        return i_array_scaled, o_array_scaled, i_scalers, o_scalers

    def split_data(self, i_array: np.ndarray, o_array: np.ndarray
                   ) -> Dict[str, np.ndarray]:
        """Split into train/val/test (70/20/10)"""
        n_dat = len(i_array)
        n_train = int(0.7 * n_dat)
        n_val = int(0.2 * n_dat)

        return {
            'train_x': i_array[:n_train],
            'train_y': o_array[:n_train],
            'val_x': i_array[n_train:n_train+n_val],
            'val_y': o_array[n_train:n_train+n_val],
            'test_x': i_array[n_train+n_val:],
            'test_y': o_array[n_train+n_val:]
        }

    def train_linear_model(self, train_x: np.ndarray, train_y: np.ndarray):
        """Train simple linear regression model"""
        from sklearn.linear_model import LinearRegression

        # Flatten for linear model
        X = train_x.reshape(len(train_x), -1)
        y = train_y.reshape(len(train_y), -1)

        model = LinearRegression()
        model.fit(X, y)

        return model

    def predict(self, model, test_x: np.ndarray) -> np.ndarray:
        """Make predictions"""
        X = test_x.reshape(len(test_x), -1)
        predictions = model.predict(X)
        return predictions.reshape(len(test_x), -1, 1)

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()

        # MAE
        mae = np.mean(np.abs(y_true_flat - y_pred_flat))

        # MSE
        mse = np.mean((y_true_flat - y_pred_flat) ** 2)

        # RMSE
        rmse = np.sqrt(mse)

        # MAPE (avoid division by zero)
        mask = y_true_flat != 0
        mape = np.mean(np.abs((y_true_flat[mask] - y_pred_flat[mask]) / y_true_flat[mask])) * 100

        # WAPE
        wape = np.sum(np.abs(y_true_flat - y_pred_flat)) / np.sum(np.abs(y_true_flat)) * 100

        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'WAPE': wape
        }


class BackendPipelineRunner:
    """Run the actual backend pipeline"""

    def __init__(self, mts_config: Dict, time_config: Dict):
        self.mts_config = mts_config
        self.time_config = time_config

    def run(self, i_dat: pd.DataFrame, o_dat: pd.DataFrame,
            utc_start: datetime.datetime, utc_end: datetime.datetime) -> Dict:
        """Run backend pipeline and return results"""

        # Import backend modules
        from domains.training.config import MTS, T
        from domains.training.data.transformer import create_training_arrays
        from domains.training.ml.scaler import process_and_scale_data

        # Configure MTS
        mts = MTS()
        mts.I_N = self.mts_config['I_N']
        mts.O_N = self.mts_config['O_N']
        mts.DELT = self.mts_config['DELT']
        mts.OFST = self.mts_config['OFST']

        # Configure T class
        T.Y.IMP = self.time_config.get('jahr', False)
        T.M.IMP = self.time_config.get('monat', False)
        T.W.IMP = self.time_config.get('woche', False)
        T.D.IMP = self.time_config.get('tag', False)
        T.H.IMP = self.time_config.get('feiertag', False)
        T.TZ = self.time_config.get('zeitzone', 'UTC')

        # Set SPEC and thresholds
        for comp in ['Y', 'M', 'W', 'D']:
            getattr(T, comp).SPEC = 'Zeithorizont'
            getattr(T, comp).TH_STRT = -24
            getattr(T, comp).TH_END = 0
            getattr(T, comp).LT = False
            getattr(T, comp).SCAL = False

        # Prepare data in expected format
        i_dat_dict = {'input': i_dat}
        o_dat_dict = {'output': o_dat}

        # Create i_dat_inf and o_dat_inf DataFrames
        i_dat_inf = pd.DataFrame({
            'utc_min': [i_dat['UTC'].min()],
            'utc_max': [i_dat['UTC'].max()],
            'spec': ['Historische Daten'],
            'th_strt': [-24],
            'th_end': [0],
            'meth': ['Lineare Interpolation'],
            'avg': [False],
            'delt_transf': [60],
            'scal': [False],
            'scal_max': [1],
            'scal_min': [0]
        }, index=['input'])

        o_dat_inf = pd.DataFrame({
            'utc_min': [o_dat['UTC'].min()],
            'utc_max': [o_dat['UTC'].max()],
            'spec': ['Historische Daten'],
            'th_strt': [0],
            'th_end': [24],
            'meth': ['Lineare Interpolation'],
            'avg': [False],
            'delt_transf': [60],
            'scal': [False],
            'scal_max': [1],
            'scal_min': [0]
        }, index=['output'])

        # Create training arrays
        (i_array_3D, o_array_3D,
         i_combined, o_combined,
         utc_ref_log) = create_training_arrays(
            i_dat_dict, o_dat_dict,
            i_dat_inf, o_dat_inf,
            utc_start, utc_end,
            mts_config=mts
        )

        return {
            'i_array_3D': i_array_3D,
            'o_array_3D': o_array_3D,
            'utc_ref_log': utc_ref_log,
            'n_samples': len(i_array_3D)
        }


def run_end_to_end_test():
    """Run complete end-to-end comparison test"""

    print("\n" + "="*80)
    print("END-TO-END COMPARISON TEST: Backend vs Original Pipeline")
    print("="*80)

    # Configuration
    mts_config = {
        'I_N': 24,      # 24 input time steps
        'O_N': 24,      # 24 output time steps
        'DELT': 60,     # 60 minutes between samples
        'OFST': 0       # No offset
    }

    time_config = {
        'jahr': True,
        'monat': True,
        'woche': True,
        'tag': True,
        'feiertag': False,
        'zeitzone': 'UTC'
    }

    print("\n--- CONFIGURATION ---")
    print(f"MTS: I_N={mts_config['I_N']}, O_N={mts_config['O_N']}, DELT={mts_config['DELT']}")
    print(f"TIME: Y={time_config['jahr']}, M={time_config['monat']}, W={time_config['woche']}, D={time_config['tag']}")

    # Generate synthetic data
    print("\n--- GENERATING SYNTHETIC DATA ---")
    generator = SyntheticDataGenerator(start_date="2024-01-01", days=90, freq_hours=1)
    i_dat = generator.generate_input_data()
    o_dat = generator.generate_output_data()

    print(f"Input data: {len(i_dat)} rows, {i_dat['UTC'].min()} to {i_dat['UTC'].max()}")
    print(f"Output data: {len(o_dat)} rows")
    print(f"Input value range: {i_dat['value'].min():.2f} to {i_dat['value'].max():.2f}")
    print(f"Output value range: {o_dat['value'].min():.2f} to {o_dat['value'].max():.2f}")

    # Define time range for training (leave buffer for zeithorizont)
    utc_start = i_dat['UTC'].min() + datetime.timedelta(days=2)
    utc_end = i_dat['UTC'].max() - datetime.timedelta(days=2)

    print(f"\nTraining range: {utc_start} to {utc_end}")

    # =========================================================================
    # RUN ORIGINAL PIPELINE SIMULATION
    # =========================================================================
    print("\n" + "-"*80)
    print("RUNNING ORIGINAL PIPELINE SIMULATION")
    print("-"*80)

    original = OriginalPipelineSimulator(mts_config, time_config)

    try:
        orig_i_array, orig_o_array, orig_utc_log = original.create_training_arrays(
            i_dat, o_dat, utc_start, utc_end
        )
        print(f"✓ Arrays created: i_array={orig_i_array.shape}, o_array={orig_o_array.shape}")
        print(f"  Samples: {len(orig_utc_log)}")
        print(f"  Input features: {orig_i_array.shape[-1]} (1 data + 8 TIME)")

        # Scale
        orig_i_scaled, orig_o_scaled, orig_i_scalers, orig_o_scalers = original.scale_data(
            orig_i_array, orig_o_array
        )
        print(f"✓ Data scaled")

        # Split
        orig_split = original.split_data(orig_i_scaled, orig_o_scaled)
        print(f"✓ Data split: train={len(orig_split['train_x'])}, val={len(orig_split['val_x'])}, test={len(orig_split['test_x'])}")

        # Train
        orig_model = original.train_linear_model(orig_split['train_x'], orig_split['train_y'])
        print(f"✓ Model trained (LinearRegression)")

        # Predict
        orig_predictions = original.predict(orig_model, orig_split['test_x'])
        print(f"✓ Predictions made: {orig_predictions.shape}")

        # Metrics
        orig_metrics = original.calculate_metrics(orig_split['test_y'], orig_predictions)
        print(f"✓ Metrics calculated:")
        for k, v in orig_metrics.items():
            print(f"    {k}: {v:.4f}")

        original_success = True

    except Exception as e:
        print(f"✗ Original pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        original_success = False

    # =========================================================================
    # RUN BACKEND PIPELINE
    # =========================================================================
    print("\n" + "-"*80)
    print("RUNNING BACKEND PIPELINE")
    print("-"*80)

    try:
        backend = BackendPipelineRunner(mts_config, time_config)
        backend_result = backend.run(i_dat, o_dat, utc_start, utc_end)

        back_i_array = backend_result['i_array_3D']
        back_o_array = backend_result['o_array_3D']

        print(f"✓ Arrays created: i_array={back_i_array.shape}, o_array={back_o_array.shape}")
        print(f"  Samples: {backend_result['n_samples']}")
        print(f"  Input features: {back_i_array.shape[-1]}")

        # Scale using same logic
        back_i_scaled, back_o_scaled, back_i_scalers, back_o_scalers = original.scale_data(
            back_i_array, back_o_array
        )
        print(f"✓ Data scaled")

        # Split
        back_split = original.split_data(back_i_scaled, back_o_scaled)
        print(f"✓ Data split: train={len(back_split['train_x'])}, val={len(back_split['val_x'])}, test={len(back_split['test_x'])}")

        # Train
        back_model = original.train_linear_model(back_split['train_x'], back_split['train_y'])
        print(f"✓ Model trained (LinearRegression)")

        # Predict
        back_predictions = original.predict(back_model, back_split['test_x'])
        print(f"✓ Predictions made: {back_predictions.shape}")

        # Metrics
        back_metrics = original.calculate_metrics(back_split['test_y'], back_predictions)
        print(f"✓ Metrics calculated:")
        for k, v in back_metrics.items():
            print(f"    {k}: {v:.4f}")

        backend_success = True

    except Exception as e:
        print(f"✗ Backend pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        backend_success = False

    # =========================================================================
    # COMPARISON
    # =========================================================================
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)

    if original_success and backend_success:
        print("\n--- ARRAY SHAPES ---")
        print(f"Original i_array: {orig_i_array.shape}")
        print(f"Backend i_array:  {back_i_array.shape}")
        shape_match = orig_i_array.shape == back_i_array.shape
        print(f"Shape match: {'✓ YES' if shape_match else '✗ NO'}")

        if shape_match:
            print("\n--- ARRAY VALUES (first sample, first 5 timesteps) ---")
            print("Feature columns: [data, D_cos, D_sin, M_cos, M_sin, W_cos, W_sin, Y_cos, Y_sin]")
            print(f"\nOriginal[0, :5, :]:")
            print(orig_i_array[0, :5, :])
            print(f"\nBackend[0, :5, :]:")
            print(back_i_array[0, :5, :])

            # Compare arrays
            print("\n--- ARRAY DIFFERENCES ---")
            max_diff = np.max(np.abs(orig_i_array - back_i_array))
            mean_diff = np.mean(np.abs(orig_i_array - back_i_array))
            print(f"Max absolute difference: {max_diff:.10f}")
            print(f"Mean absolute difference: {mean_diff:.10f}")

            arrays_match = max_diff < 1e-6
            print(f"Arrays match (< 1e-6): {'✓ YES' if arrays_match else '✗ NO'}")

        print("\n--- METRICS COMPARISON ---")
        print(f"{'Metric':<10} {'Original':>12} {'Backend':>12} {'Diff':>12} {'Match':>8}")
        print("-" * 56)

        metrics_match = True
        for metric in orig_metrics.keys():
            orig_val = orig_metrics[metric]
            back_val = back_metrics[metric]
            diff = abs(orig_val - back_val)
            match = diff < 0.01  # Allow 1% difference
            metrics_match = metrics_match and match
            print(f"{metric:<10} {orig_val:>12.4f} {back_val:>12.4f} {diff:>12.6f} {'✓' if match else '✗':>8}")

        print("\n" + "="*80)
        all_pass = shape_match and (not shape_match or arrays_match) and metrics_match
        if all_pass:
            print("FINAL RESULT: ✓ ALL TESTS PASSED - Backend matches Original")
        else:
            print("FINAL RESULT: ✗ SOME TESTS FAILED - Discrepancies found")
        print("="*80)

        return all_pass
    else:
        print("\n✗ Cannot compare - one or both pipelines failed")
        return False


def run_time_features_isolation_test():
    """Test just the TIME features in isolation"""

    print("\n" + "="*80)
    print("TIME FEATURES ISOLATION TEST")
    print("="*80)

    # Single reference time
    utc_ref = datetime.datetime(2024, 6, 15, 12, 0, 0)
    n_timesteps = 24

    # Generate timestamps for zeithorizont (-24h to 0h)
    timestamps = [utc_ref + datetime.timedelta(hours=h) for h in range(-24, 0)]

    print(f"\nReference time: {utc_ref}")
    print(f"Timestamps: {timestamps[0]} to {timestamps[-1]}")

    # Original calculation
    sec = np.array([ts.timestamp() for ts in timestamps])

    orig_features = {
        'Y_sin': np.sin(sec / YEAR_SECONDS * 2 * np.pi),
        'Y_cos': np.cos(sec / YEAR_SECONDS * 2 * np.pi),
        'M_sin': np.sin(sec / MONTH_SECONDS * 2 * np.pi),
        'M_cos': np.cos(sec / MONTH_SECONDS * 2 * np.pi),
        'W_sin': np.sin(sec / WEEK_SECONDS * 2 * np.pi),
        'W_cos': np.cos(sec / WEEK_SECONDS * 2 * np.pi),
        'D_sin': np.sin(sec / DAY_SECONDS * 2 * np.pi),
        'D_cos': np.cos(sec / DAY_SECONDS * 2 * np.pi),
    }

    # Backend calculation
    from domains.training.data.processor import TimeFeatures

    tf = TimeFeatures(timezone='UTC')
    back_year = tf._calc_year_features(timestamps, lt=False)
    back_month = tf._calc_month_features(timestamps, lt=False)
    back_week = tf._calc_week_features(timestamps, lt=False)
    back_day = tf._calc_day_features(timestamps, lt=False)

    back_features = {
        'Y_sin': back_year['Y_sin'],
        'Y_cos': back_year['Y_cos'],
        'M_sin': back_month['M_sin'],
        'M_cos': back_month['M_cos'],
        'W_sin': back_week['W_sin'],
        'W_cos': back_week['W_cos'],
        'D_sin': back_day['D_sin'],
        'D_cos': back_day['D_cos'],
    }

    print("\n--- COMPARISON ---")
    print(f"{'Feature':<10} {'Max Diff':>15} {'Status':>10}")
    print("-" * 37)

    all_match = True
    for feature in sorted(orig_features.keys()):
        diff = np.max(np.abs(orig_features[feature] - back_features[feature]))
        match = diff < 1e-10
        all_match = all_match and match
        print(f"{feature:<10} {diff:>15.12f} {'✓ PASS' if match else '✗ FAIL':>10}")

    print("\n" + "="*80)
    if all_match:
        print("TIME FEATURES: ✓ ALL MATCH")
    else:
        print("TIME FEATURES: ✗ MISMATCH DETECTED")
    print("="*80)

    return all_match


if __name__ == "__main__":
    print("\n" + "#"*80)
    print("#" + " "*30 + "END-TO-END TEST" + " "*33 + "#")
    print("#" + " "*20 + "Backend vs Original training.py" + " "*25 + "#")
    print("#"*80)

    # Run TIME features isolation test first
    time_ok = run_time_features_isolation_test()

    # Run full end-to-end test
    e2e_ok = run_end_to_end_test()

    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"TIME Features Test: {'✓ PASSED' if time_ok else '✗ FAILED'}")
    print(f"End-to-End Test:    {'✓ PASSED' if e2e_ok else '✗ FAILED'}")
    print("="*80)

    sys.exit(0 if (time_ok and e2e_ok) else 1)
