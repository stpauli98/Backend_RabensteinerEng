"""
Data transformer module that implements the EXACT data transformation loop
from training_original.py lines 1068-1760
"""

import datetime
import math
import numpy as np
import pandas as pd
import pytz
import calendar
import copy
import logging
from typing import Dict, List, Tuple, Optional
from domains.training.config import MTS, T, HOL
from domains.training.data.loader import utc_idx_pre, utc_idx_post

logger = logging.getLogger(__name__)
# Log level now controlled by LOG_LEVEL environment variable in app_factory.py

# Time constants - MUST match original training.py exactly
YEAR_SECONDS = 31557600    # 60×60×24×365.25 seconds in a year
MONTH_SECONDS = 2629800    # 60×60×24×365.25/12 seconds in a month  
WEEK_SECONDS = 604800      # 60×60×24×7 seconds in a week
DAY_SECONDS = 86400        # 60×60×24 seconds in a day


def create_training_arrays(i_dat: Dict, o_dat: Dict, i_dat_inf: pd.DataFrame,
                          o_dat_inf: pd.DataFrame, utc_strt: datetime.datetime,
                          utc_end: datetime.datetime,
                          socketio=None,
                          session_id: str = None,
                          mts_config: 'MTS' = None) -> Tuple:
    """
    Create training arrays exactly as in training_original.py lines 1068-1760
    This function implements the main time loop that builds i_arrays and o_arrays

    Args:
        i_dat: Input data dictionary
        o_dat: Output data dictionary
        i_dat_inf: Input data info DataFrame
        o_dat_inf: Output data info DataFrame
        utc_strt: Start UTC timestamp
        utc_end: End UTC timestamp
        socketio: Optional SocketIO instance for progress updates
        session_id: Optional session ID for progress tracking
        mts_config: Optional configured MTS instance (if None, uses default MTS())

    Returns:
        Tuple of (i_array_3D, o_array_3D, i_combined_array, o_combined_array, utc_ref_log)
    """
    
    # Use provided mts_config or create default MTS instance
    mts = mts_config if mts_config is not None else MTS()

    # EXACT COPY from training_original.py lines 1062-1069
    # Berechnung der Referenzzeit
    utc_ref = utc_strt.replace(minute=0, second=0, microsecond=0) \
        - datetime.timedelta(hours=1) \
        + datetime.timedelta(minutes=mts.OFST)

    while utc_ref < utc_strt:
        utc_ref += datetime.timedelta(minutes=mts.DELT)
    
    error = False
    i_arrays = []
    o_arrays = []
    utc_ref_log = []
    utc_strt = utc_ref
    
    # Koristi set() za O(1) lookup umjesto O(n) liste
    hol_d = {d.date() for d in HOL.get(T.H.CNTRY, [])} if T.H.IMP else set()
    
    iteration_count = 0
    total_iterations = int((utc_end - utc_ref).total_seconds() / 60 / mts.DELT)

    import time
    loop_start_time = time.time()

    while True:

        if utc_ref > utc_end:
            break

        iteration_count += 1
        if iteration_count % 2000 == 0:
            elapsed = time.time() - loop_start_time
            progress = (iteration_count / total_iterations * 100) if total_iterations > 0 else 0

            # ETA kalkulacija
            if iteration_count > 0 and elapsed > 0:
                rate = iteration_count / elapsed  # iterations per second
                remaining = total_iterations - iteration_count
                eta_seconds = remaining / rate if rate > 0 else 0
            else:
                eta_seconds = 0

            # Emit progress via Socket.IO
            if socketio and session_id:
                try:
                    room = f"training_{session_id}"
                    socketio.emit('training_progress', {
                        'session_id': session_id,
                        'status': 'data_transformation',
                        'message': f'Transforming data: {iteration_count}/{total_iterations}',
                        'progress_percent': round(progress, 1),
                        'phase': 'data_transformation',
                        'current_iteration': iteration_count,
                        'total_iterations': total_iterations,
                        'elapsed_seconds': round(elapsed, 1),
                        'eta_seconds': round(eta_seconds, 0),
                        'estimated_completion': round(elapsed + eta_seconds, 0)
                    }, room=room)
                except Exception as e:
                    logger.warning(f"Failed to emit data transformation progress: {e}")


        prog_1 = (utc_ref - utc_strt) / (utc_end - utc_strt) * 100
        
        df_int_i = pd.DataFrame()
        df_int_o = pd.DataFrame()
        
        for i, (key, df) in enumerate(i_dat.items()):
            
            if i_dat_inf.loc[key, "spec"] == "Historische Daten":
                
                utc_th_strt = utc_ref + datetime.timedelta(hours=float(i_dat_inf.loc[key, "th_strt"]))
                utc_th_end = utc_ref + datetime.timedelta(hours=float(i_dat_inf.loc[key, "th_end"]))
                
                if i_dat_inf.loc[key, "avg"] == True:
                    idx1 = utc_idx_post(i_dat[key], utc_th_strt)
                    idx2 = utc_idx_pre(i_dat[key], utc_th_end)
                    val = (i_dat[key].iloc[idx1:idx2, 1]).mean()
                    
                    if math.isnan(float(val)):
                        error = True
                        break
                    else:
                        df_int_i[key] = [val] * mts.I_N
                
                else:
                    val_list = []
                    
                    # ORIGINAL LOGIC: Use freq parameter with fallback to manual generation
                    try:
                        utc_th = pd.date_range(
                            start=utc_th_strt,
                            end=utc_th_end,
                            freq=f'{i_dat_inf.loc[key, "delt_transf"]}min'
                        ).to_list()
                        # Ensure exactly mts.I_N elements (date_range with freq can create N+1)
                        utc_th = utc_th[:mts.I_N]
                    except:
                        # Fallback: manual time series generation
                        delt = pd.to_timedelta(i_dat_inf.loc[key, "delt_transf"], unit="min")
                        utc_th = []
                        utc = utc_th_strt
                        for i1 in range(mts.I_N):
                            utc_th.append(utc)
                            utc += delt
                    
                    if len(utc_th) > 0:
                        data_utc_min = i_dat[key].iloc[0, 0]
                        data_utc_max = i_dat[key].iloc[-1, 0]
                    
                    if i_dat_inf.loc[key, "meth"] == "Lineare Interpolation":
                        
                        for i1 in range(len(utc_th)):
                            idx1 = utc_idx_pre(i_dat[key], utc_th[i1])
                            idx2 = utc_idx_post(i_dat[key], utc_th[i1])

                            if idx1 is None or idx2 is None:
                                if iteration_count <= 5:
                                    logger.warning(f"      ❌ Interpolation failed at iteration {iteration_count}")
                                    logger.warning(f"         utc_ref: {utc_ref}")
                                    logger.warning(f"         File: {key}")
                                    logger.warning(f"         Requested timestamp: {utc_th[i1]}")
                                    logger.warning(f"         Data range: {i_dat[key].iloc[0, 0]} to {i_dat[key].iloc[-1, 0]}")
                                    logger.warning(f"         Zeithorizont window: {utc_th_strt} to {utc_th_end}")
                                    logger.warning(f"         idx1={idx1}, idx2={idx2}")
                                error = True
                                break
                            
                            if idx1 == idx2:
                                val = i_dat[key].iloc[idx1, 1]
                            else:
                                utc1 = i_dat[key].iloc[idx1, 0]
                                utc2 = i_dat[key].iloc[idx2, 0]
                                val1 = i_dat[key].iloc[idx1, 1]
                                val2 = i_dat[key].iloc[idx2, 1]
                                
                                val = (utc_th[i1] - utc1) / (utc2 - utc1) * (val2 - val1) + val1
                            
                            if math.isnan(float(val)):
                                error = True
                                break
                            
                            val_list.append(val)
                        
                        if not error:
                            df_int_i[key] = val_list
                        else:
                            pass
                    
                    elif i_dat_inf.loc[key, "meth"] == "Mittelwertbildung":
                        logger.warning("MUSS NOCH PROGRAMMIERT WERDEN!")
                    
                    elif i_dat_inf.loc[key, "meth"] == "Nächster Wert":
                        logger.warning("MUSS NOCH PROGRAMMIERT WERDEN!")
            
            elif i_dat_inf.loc[key, "spec"] == "Historische Prognosen":
                logger.warning("MUSS NOCH PROGRAMMIERT WERDEN!")
        
        for i, (key, df) in enumerate(o_dat.items()):
            
            if o_dat_inf.loc[key, "spec"] == "Historische Daten":
                
                utc_th_strt = utc_ref + datetime.timedelta(hours=float(o_dat_inf.loc[key, "th_strt"]))
                utc_th_end = utc_ref + datetime.timedelta(hours=float(o_dat_inf.loc[key, "th_end"]))
                
                if o_dat_inf.loc[key, "avg"] == True:
                    idx1 = utc_idx_post(o_dat[key], utc_th_strt)
                    idx2 = utc_idx_pre(o_dat[key], utc_th_end)
                    val = (o_dat[key].iloc[idx1:idx2, 1]).mean()
                    
                    if math.isnan(float(val)):
                        error = True
                        break
                    else:
                        df_int_o[key] = [val] * mts.O_N
                
                else:
                    val_list = []
                    
                    # ORIGINAL LOGIC: Use freq parameter with fallback to manual generation
                    try:
                        utc_th = pd.date_range(
                            start=utc_th_strt,
                            end=utc_th_end,
                            freq=f'{o_dat_inf.loc[key, "delt_transf"]}min'
                        ).to_list()
                        # Ensure exactly mts.O_N elements (date_range with freq can create N+1)
                        utc_th = utc_th[:mts.O_N]
                    except:
                        # Fallback: manual time series generation
                        delt = pd.to_timedelta(o_dat_inf.loc[key, "delt_transf"], unit="min")
                        utc_th = []
                        utc = utc_th_strt
                        for i1 in range(mts.O_N):
                            utc_th.append(utc)
                            utc += delt
                    
                    if o_dat_inf.loc[key, "meth"] == "Lineare Interpolation":
                        
                        for i1 in range(len(utc_th)):
                            idx1 = utc_idx_pre(o_dat[key], utc_th[i1])
                            idx2 = utc_idx_post(o_dat[key], utc_th[i1])
                            
                            if idx1 is None or idx2 is None:
                                if iteration_count <= 5:
                                    logger.warning(f"      ❌ Interpolation failed at iteration {iteration_count}")
                                    logger.warning(f"         utc_ref: {utc_ref}")
                                    logger.warning(f"         File: {key} (OUTPUT)")
                                    logger.warning(f"         Requested timestamp: {utc_th[i1]}")
                                    logger.warning(f"         Data range: {o_dat[key].iloc[0, 0]} to {o_dat[key].iloc[-1, 0]}")
                                    logger.warning(f"         Zeithorizont window: {utc_th_strt} to {utc_th_end}")
                                    logger.warning(f"         idx1={idx1}, idx2={idx2}")
                                error = True
                                break
                            
                            if idx1 == idx2:
                                val = o_dat[key].iloc[idx1, 1]
                            else:
                                utc1 = o_dat[key].iloc[idx1, 0]
                                utc2 = o_dat[key].iloc[idx2, 0]
                                val1 = o_dat[key].iloc[idx1, 1]
                                val2 = o_dat[key].iloc[idx2, 1]
                                val = (utc_th[i1] - utc1) / (utc2 - utc1) * (val2 - val1) + val1
                            
                            if math.isnan(float(val)):
                                error = True
                                break
                            
                            val_list.append(val)
                        
                        if not error:
                            df_int_o[key] = val_list
                    
                    elif o_dat_inf.loc[key, "meth"] == "Mittelwertbildung":
                        logger.warning("MUSS NOCH PROGRAMMIERT WERDEN!")
                    
                    elif o_dat_inf.loc[key, "meth"] == "Nächster Wert":
                        logger.warning("MUSS NOCH PROGRAMMIERT WERDEN!")
            
            elif o_dat_inf.loc[key, "spec"] == "Historische Prognosen":
                logger.warning("MUSS NOCH PROGRAMMIERT WERDEN!")
        
        if error == False:
            
            if T.Y.IMP:
                if T.Y.SPEC == "Zeithorizont":
                    utc_th_strt = utc_ref + datetime.timedelta(hours=T.Y.TH_STRT)
                    utc_th_end = utc_ref + datetime.timedelta(hours=T.Y.TH_END)
                    
                    # Use periods parameter to guarantee exactly mts.I_N elements
                    utc_th = pd.date_range(
                        start=utc_th_strt,
                        end=utc_th_end,
                        periods=mts.I_N
                    ).to_list()

                    if T.Y.LT == False:
                        sec = pd.Series(utc_th).map(pd.Timestamp.timestamp)
                        df_int_i["Y_sin"] = np.sin(sec / 31557600 * 2 * np.pi)
                        df_int_i["Y_cos"] = np.cos(sec / 31557600 * 2 * np.pi)
                    
                    else:
                        utc_th = [pytz.utc.localize(dt) for dt in utc_th]
                        lt_th = [dt.astimezone(pytz.timezone(T.TZ)) for dt in utc_th]

                        sec = np.array([(dt.timetuple().tm_yday - 1) * 86400 +
                                       dt.hour * 3600 +
                                       dt.minute * 60 +
                                       dt.second for dt in lt_th])
                        
                        y = np.array([x.year for x in lt_th])
                        is_leap = np.vectorize(calendar.isleap)(y)
                        sec_y = np.where(is_leap, 31622400, 31536000)
                        
                        df_int_i["Y_sin"] = np.sin(sec / sec_y * 2 * np.pi)
                        df_int_i["Y_cos"] = np.cos(sec / sec_y * 2 * np.pi)
                
                elif T.Y.SPEC == "Aktuelle Zeit":
                    if T.Y.LT == False:
                        sec = utc_ref.timestamp()
                        df_int_i["Y_sin"] = [np.sin(sec / 31557600 * 2 * np.pi)] * mts.I_N
                        df_int_i["Y_cos"] = [np.cos(sec / 31557600 * 2 * np.pi)] * mts.I_N
                    else:
                        lt = pytz.utc.localize(utc_ref).astimezone(pytz.timezone(T.TZ))
                        sec = (lt.timetuple().tm_yday - 1) * 86400 + lt.hour * 3600 + lt.minute * 60 + lt.second

                        if calendar.isleap(lt.year):
                            sec_y = 31622400
                        else:
                            sec_y = 31536000
                        
                        df_int_i["Y_sin"] = [np.sin(sec / sec_y * 2 * np.pi)] * mts.I_N
                        df_int_i["Y_cos"] = [np.cos(sec / sec_y * 2 * np.pi)] * mts.I_N
            
            if T.M.IMP:
                if T.M.SPEC == "Zeithorizont":
                    utc_th_strt = utc_ref + datetime.timedelta(hours=T.M.TH_STRT)
                    utc_th_end = utc_ref + datetime.timedelta(hours=T.M.TH_END)
                    
                    # Use periods parameter to guarantee exactly mts.I_N elements
                    utc_th = pd.date_range(
                        start=utc_th_strt,
                        end=utc_th_end,
                        periods=mts.I_N
                    ).to_list()

                    if T.M.LT == False:
                        # MATCHES ORIGINAL: Use Unix timestamp / constant MONTH_SECONDS
                        sec = pd.Series(utc_th).map(pd.Timestamp.timestamp)
                        
                        df_int_i["M_sin"] = np.sin(sec / MONTH_SECONDS * 2 * np.pi)
                        df_int_i["M_cos"] = np.cos(sec / MONTH_SECONDS * 2 * np.pi)
                    else:
                        # LT mode: Convert to local time, then use timestamp / MONTH_SECONDS
                        utc_th_tz = [pytz.utc.localize(dt) for dt in utc_th]
                        lt_th = [dt.astimezone(pytz.timezone(T.TZ)) for dt in utc_th_tz]
                        
                        # MATCHES ORIGINAL: Use Unix timestamp of local time / constant
                        sec = np.array([dt.timestamp() for dt in lt_th])
                        
                        df_int_i["M_sin"] = np.sin(sec / MONTH_SECONDS * 2 * np.pi)
                        df_int_i["M_cos"] = np.cos(sec / MONTH_SECONDS * 2 * np.pi)
                
                elif T.M.SPEC == "Aktuelle Zeit":
                    if T.M.LT == False:
                        # MATCHES ORIGINAL: Use Unix timestamp / constant MONTH_SECONDS
                        sec = utc_ref.timestamp() if hasattr(utc_ref, 'timestamp') else pd.Timestamp(utc_ref).timestamp()
                        
                        df_int_i["M_sin"] = [np.sin(sec / MONTH_SECONDS * 2 * np.pi)] * mts.I_N
                        df_int_i["M_cos"] = [np.cos(sec / MONTH_SECONDS * 2 * np.pi)] * mts.I_N
                    else:
                        lt = pytz.utc.localize(utc_ref).astimezone(pytz.timezone(T.TZ))
                        # MATCHES ORIGINAL: Use Unix timestamp of local time / constant
                        sec = lt.timestamp()
                        
                        df_int_i["M_sin"] = [np.sin(sec / MONTH_SECONDS * 2 * np.pi)] * mts.I_N
                        df_int_i["M_cos"] = [np.cos(sec / MONTH_SECONDS * 2 * np.pi)] * mts.I_N
            
            if T.W.IMP:
                if T.W.SPEC == "Zeithorizont":
                    utc_th_strt = utc_ref + datetime.timedelta(hours=T.W.TH_STRT)
                    utc_th_end = utc_ref + datetime.timedelta(hours=T.W.TH_END)
                    
                    # Use periods parameter to guarantee exactly mts.I_N elements
                    utc_th = pd.date_range(
                        start=utc_th_strt,
                        end=utc_th_end,
                        periods=mts.I_N
                    ).to_list()

                    if T.W.LT == False:
                        # MATCHES ORIGINAL: Use Unix timestamp / WEEK_SECONDS
                        sec = pd.Series(utc_th).map(pd.Timestamp.timestamp)
                        df_int_i["W_sin"] = np.sin(sec / WEEK_SECONDS * 2 * np.pi)
                        df_int_i["W_cos"] = np.cos(sec / WEEK_SECONDS * 2 * np.pi)
                    else:
                        # LT mode: Convert to local time, use Unix timestamp
                        utc_th_tz = [pytz.utc.localize(dt) for dt in utc_th]
                        lt_th = [dt.astimezone(pytz.timezone(T.TZ)) for dt in utc_th_tz]
                        # MATCHES ORIGINAL: Use Unix timestamp of local time
                        sec = np.array([dt.timestamp() for dt in lt_th])
                        df_int_i["W_sin"] = np.sin(sec / WEEK_SECONDS * 2 * np.pi)
                        df_int_i["W_cos"] = np.cos(sec / WEEK_SECONDS * 2 * np.pi)
                
                elif T.W.SPEC == "Aktuelle Zeit":
                    if T.W.LT == False:
                        # MATCHES ORIGINAL: Use Unix timestamp / WEEK_SECONDS
                        sec = utc_ref.timestamp() if hasattr(utc_ref, 'timestamp') else pd.Timestamp(utc_ref).timestamp()
                        df_int_i["W_sin"] = [np.sin(sec / WEEK_SECONDS * 2 * np.pi)] * mts.I_N
                        df_int_i["W_cos"] = [np.cos(sec / WEEK_SECONDS * 2 * np.pi)] * mts.I_N
                    else:
                        lt = pytz.utc.localize(utc_ref).astimezone(pytz.timezone(T.TZ))
                        # MATCHES ORIGINAL: Use Unix timestamp of local time
                        sec = lt.timestamp()
                        df_int_i["W_sin"] = [np.sin(sec / WEEK_SECONDS * 2 * np.pi)] * mts.I_N
                        df_int_i["W_cos"] = [np.cos(sec / WEEK_SECONDS * 2 * np.pi)] * mts.I_N
            
            if T.D.IMP:
                if T.D.SPEC == "Zeithorizont":
                    utc_th_strt = utc_ref + datetime.timedelta(hours=T.D.TH_STRT)
                    utc_th_end = utc_ref + datetime.timedelta(hours=T.D.TH_END)
                    
                    # Use periods parameter to guarantee exactly mts.I_N elements
                    utc_th = pd.date_range(
                        start=utc_th_strt,
                        end=utc_th_end,
                        periods=mts.I_N
                    ).to_list()

                    if T.D.LT == False:
                        # MATCHES ORIGINAL: Use Unix timestamp / DAY_SECONDS
                        sec = pd.Series(utc_th).map(pd.Timestamp.timestamp)
                        df_int_i["D_sin"] = np.sin(sec / DAY_SECONDS * 2 * np.pi)
                        df_int_i["D_cos"] = np.cos(sec / DAY_SECONDS * 2 * np.pi)
                    else:
                        # LT mode: Convert to local time, use Unix timestamp
                        utc_th_tz = [pytz.utc.localize(dt) for dt in utc_th]
                        lt_th = [dt.astimezone(pytz.timezone(T.TZ)) for dt in utc_th_tz]
                        # MATCHES ORIGINAL: Use Unix timestamp of local time
                        sec = np.array([dt.timestamp() for dt in lt_th])
                        df_int_i["D_sin"] = np.sin(sec / DAY_SECONDS * 2 * np.pi)
                        df_int_i["D_cos"] = np.cos(sec / DAY_SECONDS * 2 * np.pi)
                
                elif T.D.SPEC == "Aktuelle Zeit":
                    if T.D.LT == False:
                        # MATCHES ORIGINAL: Use Unix timestamp / DAY_SECONDS
                        sec = utc_ref.timestamp() if hasattr(utc_ref, 'timestamp') else pd.Timestamp(utc_ref).timestamp()
                        df_int_i["D_sin"] = [np.sin(sec / DAY_SECONDS * 2 * np.pi)] * mts.I_N
                        df_int_i["D_cos"] = [np.cos(sec / DAY_SECONDS * 2 * np.pi)] * mts.I_N
                    else:
                        lt = pytz.utc.localize(utc_ref).astimezone(pytz.timezone(T.TZ))
                        # MATCHES ORIGINAL: Use Unix timestamp of local time
                        sec = lt.timestamp()
                        df_int_i["D_sin"] = [np.sin(sec / DAY_SECONDS * 2 * np.pi)] * mts.I_N
                        df_int_i["D_cos"] = [np.cos(sec / DAY_SECONDS * 2 * np.pi)] * mts.I_N
            
            if T.H.IMP:
                if T.H.SPEC == "Zeithorizont":
                    # Zeithorizont implementation - check each timestep for holidays
                    utc_th_strt = utc_ref + datetime.timedelta(hours=T.H.TH_STRT)
                    utc_th_end = utc_ref + datetime.timedelta(hours=T.H.TH_END)

                    # Use periods parameter to guarantee exactly mts.I_N elements
                    utc_th = pd.date_range(
                        start=utc_th_strt,
                        end=utc_th_end,
                        periods=mts.I_N
                    ).to_list()

                    if T.H.LT == False:
                        # Compare UTC dates against holidays
                        df_int_i["H"] = np.array([1 if dt.date() in hol_d else 0 for dt in utc_th])
                    else:
                        # Convert to local time first
                        utc_th_loc = [pytz.utc.localize(dt) if dt.tzinfo is None else dt for dt in utc_th]
                        lt_th = [dt.astimezone(pytz.timezone(T.TZ)) for dt in utc_th_loc]
                        # Compare local dates against holidays
                        df_int_i["H"] = np.array([1 if dt.date() in hol_d else 0 for dt in lt_th])

                elif T.H.SPEC == "Aktuelle Zeit":
                    if T.H.LT == False:
                        df_int_i["H"] = [1 if utc_ref.date() in hol_d else 0] * mts.I_N
                    else:
                        lt = pytz.utc.localize(utc_ref).astimezone(pytz.timezone(T.TZ))
                        df_int_i["H"] = [1 if lt.date() in hol_d else 0] * mts.I_N
        
        
        # MATCHES ORIGINAL: Only append if no error occurred during this iteration
        # This ensures partial data from failed interpolations isn't included
        if error == False:
            i_arrays.append(df_int_i.values)
            o_arrays.append(df_int_o.values)
            utc_ref_log.append(utc_ref)
        else:
            # CRITICAL: Reset error for next iteration (matches original line 1786)
            # Without this, once error=True, ALL subsequent iterations would be skipped
            error = False

        # ALWAYS advance utc_ref (matches original line 1788)
        # This is OUTSIDE the if/else, so it runs regardless of error state
        utc_ref = utc_ref + datetime.timedelta(minutes=mts.DELT)
    
    if len(i_arrays) > 0 and len(o_arrays) > 0:
        i_array_3D = np.array(i_arrays)
        o_array_3D = np.array(o_arrays)

        n_dat = i_array_3D.shape[0]
        n_features_in = i_array_3D.shape[2] if len(i_array_3D.shape) > 2 else 0
        n_features_out = o_array_3D.shape[2] if len(o_array_3D.shape) > 2 else 0

        logger.info(f"Transformer complete: {n_dat} samples, {n_features_in} input features, {n_features_out} output features")

        i_combined_array = np.vstack(i_arrays)
        o_combined_array = np.vstack(o_arrays)
    else:
        logger.warning("No valid datasets created during interpolation")
        i_array_3D = np.array([])
        o_array_3D = np.array([])
        i_combined_array = np.array([])
        o_combined_array = np.array([])
        n_dat = 0
    
    del i_arrays, o_arrays
    
    
    return i_array_3D, o_array_3D, i_combined_array, o_combined_array, utc_ref_log
