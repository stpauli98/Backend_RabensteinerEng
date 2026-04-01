"""
Forecast Service — executes the forecast pipeline.

Adapted from loading.py. Replaces hardcoded Excel/CSV sources with
dynamic config received from the frontend via the forecast endpoint.
"""

import os
import copy
import datetime
import logging
import numpy as np
import pandas as pd
import requests
from io import StringIO
from typing import Dict, Any

from domains.training.services.prediction_service import PredictionService

logger = logging.getLogger(__name__)


def wf_GeoSphere(lat: float, lon: float, fcst_var: str) -> pd.DataFrame:
    from shared.database.operations import get_supabase_client
    supabase = get_supabase_client(use_service_role=True)

    result = supabase.table('api_parameters') \
        .select('parameter, lbl') \
        .eq('api_source', 'GeoSphere') \
        .eq('name_user', fcst_var) \
        .limit(1) \
        .execute()

    if not result.data:
        raise ValueError(f"GeoSphere parameter not found for: {fcst_var}")

    param = result.data[0]['parameter']
    lbl = result.data[0]['lbl']

    request_url = ("https://dataset.api.hub.geosphere.at/v1/timeseries/"
                   "forecast/nwp-v1-1h-2500m")

    params = {
        "parameters": [param],
        "lat_lon": [f"{lat:.6f},{lon:.6f}"],
        "forecast_offset": 0,
        "output_format": "csv",
    }

    r = requests.get(request_url, params=params, timeout=60)
    if not r.ok:
        try:
            detail = r.json()
        except Exception:
            detail = r.text[:800]
        raise RuntimeError(f"GeoSphere API HTTP {r.status_code}: {detail}")

    df = pd.read_csv(StringIO(r.text))
    df["time"] = pd.to_datetime(df["time"])
    df = df.rename(columns={"time": "UTC"})
    df["UTC"] = df["UTC"].dt.tz_localize(None)
    df = df[["UTC", lbl]]

    return df


def wf_OpenWeather(lat: float, lon: float, fcst_var: str) -> pd.DataFrame:
    from shared.database.operations import get_supabase_client
    supabase = get_supabase_client(use_service_role=True)

    api_key = os.environ.get("OPENWEATHER_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENWEATHER_API_KEY not set in environment")

    result = supabase.table('api_parameters') \
        .select('parameter, lbl') \
        .eq('api_source', 'OpenWeather') \
        .eq('name_user', fcst_var) \
        .limit(1) \
        .execute()

    if not result.data:
        raise ValueError(f"OpenWeather parameter not found for: {fcst_var}")

    param = result.data[0]['parameter']
    lbl = result.data[0]['lbl']

    request_url = (f"https://api.openweathermap.org/data/2.5/forecast?"
                   f"lat={lat}&lon={lon}&appid={api_key}&units=metric")

    response = requests.get(request_url, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(f"OpenWeather API HTTP {response.status_code}")

    data = response.json()

    value_list, utc_list = [], []
    for item in data["list"]:
        utc_list.append(
            pd.to_datetime(item["dt_txt"], format="%Y-%m-%d %H:%M:%S")
            .replace(tzinfo=datetime.timezone.utc)
        )
        if param in ["deg", "gust", "speed"]:
            value_list.append(item["wind"][param])
        else:
            value_list.append(item["main"][param])

    df = pd.DataFrame({"UTC": utc_list, lbl: value_list})
    return df


def intrpl_timeline(utc_timeline: pd.Series, df: pd.DataFrame) -> pd.DataFrame:
    time_col = df.columns[0]
    value_col = df.columns[1]

    out = df[[time_col, value_col]].copy()
    out = out.set_index(time_col)

    target_index = pd.Index(pd.to_datetime(utc_timeline), name=time_col)
    out = out.reindex(out.index.union(target_index)).sort_index()
    out[value_col] = out[value_col].interpolate(method="time", limit_area="inside")
    out = out[~out.index.duplicated(keep="first")]
    out = out.loc[target_index].reset_index()

    return out


def _download_user_csv(storage_path: str) -> pd.DataFrame:
    """Download a User feature CSV from Supabase Storage."""
    from shared.database.operations import get_supabase_client
    supabase = get_supabase_client(use_service_role=True)

    try:
        data = supabase.storage.from_('csv-files').download(storage_path)
        content = data.decode('utf-8')
        df = pd.read_csv(StringIO(content), sep=';', parse_dates=[0])
        return df
    except Exception as e:
        raise FileNotFoundError(f"Could not download CSV from storage: {storage_path}. Error: {e}")


def run_forecast(
    session_id: str,
    user_id: str,
    config: Dict[str, Any],
    user_csvs: Dict[str, pd.DataFrame] = None
) -> Dict[str, Any]:
    input_features = config['input_features']
    output_features = config['output_features']
    time_info = config['time_info']
    zeitschritte = config['zeitschritte']

    N_IN = int(zeitschritte['eingabe'])
    N_OUT = int(zeitschritte['ausgabe'])
    TZ = time_info.get('zeitzone', 'UTC')

    input_features = sorted(input_features, key=lambda f: f['feature_index'])

    # 1. Collect raw data
    data_in = {}
    for feat in input_features:
        src = feat['data_source']
        name = feat['bezeichnung']

        if src == 'User':
            if user_csvs and name in user_csvs:
                data_in[name] = user_csvs[name]
            else:
                storage_path = feat.get('storage_path')
                if not storage_path:
                    raise ValueError(f"No data for User feature: {name}. Provide user_data in request.")
                data_in[name] = _download_user_csv(storage_path)
        elif src == 'Extern':
            api = feat['api_source']
            var = feat['fcst_var']
            lat = float(feat['latitude'])
            lon = float(feat['longitude'])

            if api == 'GeoSphere':
                data_in[name] = wf_GeoSphere(lat, lon, var)
            elif api == 'OpenWeather':
                data_in[name] = wf_OpenWeather(lat, lon, var)
            else:
                raise ValueError(f"Unknown api_source: {api}")

    # 2. Compute UTC reference time
    utc_now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)

    # Allow frontend to override utc_ref (useful for testing with old data)
    custom_utc_ref = config.get('utc_ref_override')
    if custom_utc_ref:
        utc_ref = pd.to_datetime(custom_utc_ref)
        logger.info(f"Using custom utc_ref from frontend: {utc_ref}")
    else:
        utc_refs = []
        for feat in input_features:
            if feat['data_source'] in ['User', 'Extern']:
                name = feat['bezeichnung']
                if name in data_in:
                    dt = feat.get('data_type', 'time horizon')
                    if dt == 'time horizon' and float(feat.get('horizon_start_h', 0)) <= 0 and float(feat.get('horizon_end_h', 0)) <= 0:
                        utc_refs.append(
                            data_in[name].iloc[:, 0].max()
                            - datetime.timedelta(hours=float(feat.get('horizon_end_h', 0)))
                        )
                    elif dt == 'actual value':
                        utc_refs.append(data_in[name].iloc[:, 0].max())

        utc_ref = min(utc_refs) if utc_refs else utc_now
        if utc_ref > utc_now:
            utc_ref = utc_now
        logger.info(f"Computed utc_ref from data: {utc_ref}")

    # 3. Build interpolated input arrays
    data_in_final = {}
    period_seconds = {
        "year": 31557600, "month": 31557600 / 12,
        "week": 604800, "day": 86400,
    }

    for feat in input_features:
        name = feat['bezeichnung']
        dt = feat.get('data_type', 'time horizon')
        src = feat['data_source']

        if dt == 'time horizon':
            utc_strt = utc_ref + datetime.timedelta(hours=float(feat['horizon_start_h']))
            utc_end = utc_ref + datetime.timedelta(hours=float(feat['horizon_end_h']))
        else:
            utc_strt = utc_ref
            utc_end = utc_ref

        utc_timeline = pd.Series(
            pd.date_range(start=utc_strt, end=utc_end, periods=N_IN), name="t"
        )

        if src in ['User', 'Extern']:
            data_in_final[name] = intrpl_timeline(utc_timeline, data_in[name])
            if data_in_final[name].iloc[:, 1].isna().any():
                raise ValueError(f"NaN in {name} — CSV time range doesn't cover required horizon")

    # 4. Generate time features (sin/cos encoding)
    time_categories = ['jahr', 'monat', 'woche', 'tag']
    time_feature_names = []
    period_map = {'jahr': 'year', 'monat': 'month', 'woche': 'week', 'tag': 'day'}

    for cat in time_categories:
        if time_info.get(cat, False):
            cat_data = time_info.get('category_data', {}).get(cat, {})

            cat_dt = cat_data.get('datenform', 'Zeithorizont')
            if cat_dt == 'Zeithorizont':
                h_start = float(cat_data.get('zeithorizontStart', 0))
                h_end = float(cat_data.get('zeithorizontEnd', 0))
                t_strt = utc_ref + datetime.timedelta(hours=h_start)
                t_end = utc_ref + datetime.timedelta(hours=h_end)
            else:
                t_strt = utc_ref
                t_end = utc_ref

            utc_tl = pd.Series(
                pd.date_range(start=t_strt, end=t_end, periods=N_IN), name="t"
            )

            detailed = cat_data.get('detaillierteBerechnung', False)
            if detailed:
                local_tl = (
                    pd.to_datetime(utc_tl)
                    .dt.tz_localize("UTC")
                    .dt.tz_convert(TZ)
                    .dt.tz_localize(None)
                )
            else:
                local_tl = utc_tl

            timestamps = pd.Series(local_tl).map(pd.Timestamp.timestamp)
            period = period_seconds[period_map[cat]]
            angle = timestamps * (2 * np.pi / period)

            for func_name, func in [('sin', np.sin), ('cos', np.cos)]:
                feat_name = f"{period_map[cat]}|{func_name}"
                value = pd.Series(func(angle), name=feat_name)
                data_in_final[feat_name] = pd.DataFrame({
                    "UTC": utc_tl, feat_name: value
                })
                time_feature_names.append(feat_name)

    # 5. Assemble input array in correct order
    value_list = []
    for feat in input_features:
        value_list.append(data_in_final[feat['bezeichnung']].iloc[:, 1])
    for tf_name in time_feature_names:
        value_list.append(data_in_final[tf_name].iloc[:, 1])

    arr = np.array([s.values for s in value_list]).T

    # 6. Load model and scalers, scale input
    pred_service = PredictionService(session_id, user_id)

    # Auto-detect model file from storage
    from utils.model_storage import list_session_models
    session_models = list_session_models(session_id)
    h5_models = [m for m in session_models if m.get('format') == 'h5']
    if not h5_models:
        raise FileNotFoundError(f"No .h5 model found in storage for session {session_id}")
    model_filename = h5_models[0]['filename']
    model = pred_service.load_model(model_filename)

    scalers = pred_service.load_scalers()
    input_scalers = scalers.get('input', {})
    output_scalers = scalers.get('output', {})

    arr_scaled = copy.deepcopy(arr)
    for i in range(arr_scaled.shape[1]):
        if i in input_scalers and input_scalers[i] is not None:
            arr_scaled[:, i] = input_scalers[i].transform(
                arr_scaled[:, i].reshape(-1, 1)
            ).flatten()

    input_x = arr_scaled.reshape((1, arr_scaled.shape[0], arr_scaled.shape[1]))

    # 7. Predict and inverse-scale
    raw_predictions = model.predict(input_x, verbose=0)

    forecasts = {}
    for idx, out_feat in enumerate(output_features):
        yhat = raw_predictions[idx] if len(output_features) > 1 else raw_predictions[0]

        h_start = float(out_feat['horizon_start_h'])
        h_end = float(out_feat['horizon_end_h'])
        utc_strt = utc_ref + datetime.timedelta(hours=h_start)
        utc_end = utc_ref + datetime.timedelta(hours=h_end)

        utc_list = pd.date_range(start=utc_strt, end=utc_end, periods=N_OUT)

        if idx in output_scalers and output_scalers[idx] is not None:
            values = output_scalers[idx].inverse_transform(
                yhat.reshape(-1, 1)
            ).flatten()
        else:
            values = yhat.flatten()

        forecasts[out_feat['bezeichnung']] = [
            {"UTC": str(utc_list[j]), "value": round(float(values[j]), 4)}
            for j in range(len(values))
        ]

    return {
        "forecasts": forecasts,
        "utc_ref": str(utc_ref),
        "model_used": model_filename,
        "n_predictions": N_OUT,
        "horizon": {
            "start": str(utc_list[0]) if len(utc_list) > 0 else None,
            "end": str(utc_list[-1]) if len(utc_list) > 0 else None,
            "resolution_min": float(zeitschritte.get('zeitschrittweite', 15))
        }
    }
