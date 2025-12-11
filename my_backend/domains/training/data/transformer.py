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


def create_training_arrays(i_dat: Dict, o_dat: Dict, i_dat_inf: pd.DataFrame,
                          o_dat_inf: pd.DataFrame, utc_strt: datetime.datetime,
                          utc_end: datetime.datetime,
                          socketio=None,
                          session_id: str = None) -> Tuple:
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
        
    Returns:
        Tuple of (i_array_3D, o_array_3D, i_combined_array, o_combined_array, utc_ref_log)
    """
    
    mts = MTS()
    
    utc_ref = utc_strt - datetime.timedelta(minutes=mts.DELT)
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

    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"      Starting main time loop: {total_iterations} iterations expected")
    logger.info(f"      Time range: {utc_strt} to {utc_end}")
    logger.info(f"      Input files: {list(i_dat.keys())}")
    logger.info(f"      Output files: {list(o_dat.keys())}")

    # Log data ranges
    for key in i_dat.keys():
        logger.info(f"      Input '{key}': data from {i_dat[key].iloc[0, 0]} to {i_dat[key].iloc[-1, 0]}")
    for key in o_dat.keys():
        logger.info(f"      Output '{key}': data from {o_dat[key].iloc[0, 0]} to {o_dat[key].iloc[-1, 0]}")

    # Log zeithorizont windows
    for key in i_dat_inf.index:
        logger.info(f"      Input '{key}': zeithorizont {i_dat_inf.loc[key, 'th_strt']} to {i_dat_inf.loc[key, 'th_end']}")
    for key in o_dat_inf.index:
        logger.info(f"      Output '{key}': zeithorizont {o_dat_inf.loc[key, 'th_strt']} to {o_dat_inf.loc[key, 'th_end']}")

    import time
    loop_start_time = time.time()

    while True:

        if utc_ref > utc_end:
            break

        iteration_count += 1
        if iteration_count % 500 == 0:
            elapsed = time.time() - loop_start_time
            progress = (iteration_count / total_iterations * 100) if total_iterations > 0 else 0

            # ETA kalkulacija
            if iteration_count > 0 and elapsed > 0:
                rate = iteration_count / elapsed  # iterations per second
                remaining = total_iterations - iteration_count
                eta_seconds = remaining / rate if rate > 0 else 0
            else:
                eta_seconds = 0

            logger.info(f"      Progress: {iteration_count}/{total_iterations} ({progress:.1f}%) - {elapsed:.1f}s elapsed - ETA: {eta_seconds:.0f}s")

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
                    
                    
                    try:
                        utc_th = pd.date_range(
                            start=utc_th_strt,
                            end=utc_th_end,
                            freq=f'{i_dat_inf.loc[key, "delt_transf"]}min'
                        ).to_list()
                    except Exception as e:
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
                    
                    try:
                        utc_th = pd.date_range(
                            start=utc_th_strt,
                            end=utc_th_end,
                            freq=f'{o_dat_inf.loc[key, "delt_transf"]}min'
                        ).to_list()
                    except:
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
                    
                    try:
                        utc_th = pd.date_range(
                            start=utc_th_strt,
                            end=utc_th_end,
                            freq=f'{T.Y.DELT}min'
                        ).to_list()
                    except:
                        delt = pd.to_timedelta(T.Y.DELT, unit="min")
                        utc_th = []
                        utc = utc_th_strt
                        for i1 in range(mts.I_N):
                            utc_th.append(utc)
                            utc += delt
                    
                    if T.Y.LT == False:
                        sec = pd.Series(utc_th).map(pd.Timestamp.timestamp)
                        df_int_i["y_sin"] = np.sin(sec / 31557600 * 2 * np.pi)
                        df_int_i["y_cos"] = np.cos(sec / 31557600 * 2 * np.pi)
                    
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
                        
                        df_int_i["y_sin"] = np.sin(sec / sec_y * 2 * np.pi)
                        df_int_i["y_cos"] = np.cos(sec / sec_y * 2 * np.pi)
                
                elif T.Y.SPEC == "Aktuelle Zeit":
                    if T.Y.LT == False:
                        sec = utc_ref.timestamp()
                        df_int_i["y_sin"] = [np.sin(sec / 31557600 * 2 * np.pi)] * mts.I_N
                        df_int_i["y_cos"] = [np.cos(sec / 31557600 * 2 * np.pi)] * mts.I_N
                    else:
                        lt = pytz.utc.localize(utc_ref).astimezone(pytz.timezone(T.TZ))
                        sec = (lt.timetuple().tm_yday - 1) * 86400 + lt.hour * 3600 + lt.minute * 60 + lt.second

                        if calendar.isleap(lt.year):
                            sec_y = 31622400
                        else:
                            sec_y = 31536000
                        
                        df_int_i["y_sin"] = [np.sin(sec / sec_y * 2 * np.pi)] * mts.I_N
                        df_int_i["y_cos"] = [np.cos(sec / sec_y * 2 * np.pi)] * mts.I_N
            
            if T.M.IMP:
                if T.M.SPEC == "Zeithorizont":
                    utc_th_strt = utc_ref + datetime.timedelta(hours=T.M.TH_STRT)
                    utc_th_end = utc_ref + datetime.timedelta(hours=T.M.TH_END)
                    
                    try:
                        utc_th = pd.date_range(
                            start=utc_th_strt,
                            end=utc_th_end,
                            freq=f'{T.M.DELT}min'
                        ).to_list()
                    except:
                        delt = pd.to_timedelta(T.M.DELT, unit="min")
                        utc_th = []
                        utc = utc_th_strt
                        for i1 in range(mts.I_N):
                            utc_th.append(utc)
                            utc += delt
                    
                    if T.M.LT == False:
                        m = pd.Series(utc_th).dt.month.values
                        d = pd.Series(utc_th).dt.day.values
                        h = pd.Series(utc_th).dt.hour.values
                        
                        sec = (d - 1) * 86400 + h * 3600
                        sec_m = np.array([calendar.monthrange(utc_th[i].year, m[i])[1] * 86400 
                                         for i in range(len(utc_th))])
                        
                        df_int_i["m_sin"] = np.sin(sec / sec_m * 2 * np.pi)
                        df_int_i["m_cos"] = np.cos(sec / sec_m * 2 * np.pi)
                    else:
                        utc_th = [pytz.utc.localize(dt) for dt in utc_th]
                        lt_th = [dt.astimezone(pytz.timezone(T.TZ)) for dt in utc_th]

                        sec = np.array([(dt.day - 1) * 86400 +
                                       dt.hour * 3600 +
                                       dt.minute * 60 +
                                       dt.second for dt in lt_th])

                        sec_m = np.array([calendar.monthrange(dt.year, dt.month)[1] * 86400
                                         for dt in lt_th])
                        
                        df_int_i["m_sin"] = np.sin(sec / sec_m * 2 * np.pi)
                        df_int_i["m_cos"] = np.cos(sec / sec_m * 2 * np.pi)
                
                elif T.M.SPEC == "Aktuelle Zeit":
                    if T.M.LT == False:
                        d = utc_ref.day
                        h = utc_ref.hour
                        sec = (d - 1) * 86400 + h * 3600
                        sec_m = calendar.monthrange(utc_ref.year, utc_ref.month)[1] * 86400
                        
                        df_int_i["m_sin"] = [np.sin(sec / sec_m * 2 * np.pi)] * mts.I_N
                        df_int_i["m_cos"] = [np.cos(sec / sec_m * 2 * np.pi)] * mts.I_N
                    else:
                        lt = pytz.utc.localize(utc_ref).astimezone(pytz.timezone(T.TZ))
                        sec = (lt.day - 1) * 86400 + lt.hour * 3600 + lt.minute * 60 + lt.second
                        sec_m = calendar.monthrange(lt.year, lt.month)[1] * 86400
                        
                        df_int_i["m_sin"] = [np.sin(sec / sec_m * 2 * np.pi)] * mts.I_N
                        df_int_i["m_cos"] = [np.cos(sec / sec_m * 2 * np.pi)] * mts.I_N
            
            if T.W.IMP:
                if T.W.SPEC == "Zeithorizont":
                    utc_th_strt = utc_ref + datetime.timedelta(hours=T.W.TH_STRT)
                    utc_th_end = utc_ref + datetime.timedelta(hours=T.W.TH_END)
                    
                    try:
                        utc_th = pd.date_range(
                            start=utc_th_strt,
                            end=utc_th_end,
                            freq=f'{T.W.DELT}min'
                        ).to_list()
                    except:
                        delt = pd.to_timedelta(T.W.DELT, unit="min")
                        utc_th = []
                        utc = utc_th_strt
                        for i1 in range(mts.I_N):
                            utc_th.append(utc)
                            utc += delt
                    
                    if T.W.LT == False:
                        wd = pd.Series(utc_th).dt.weekday.values
                        h = pd.Series(utc_th).dt.hour.values
                        m = pd.Series(utc_th).dt.minute.values
                        
                        sec = wd * 86400 + h * 3600 + m * 60
                        df_int_i["w_sin"] = np.sin(sec / 604800 * 2 * np.pi)
                        df_int_i["w_cos"] = np.cos(sec / 604800 * 2 * np.pi)
                    else:
                        utc_th = [pytz.utc.localize(dt) for dt in utc_th]
                        lt_th = [dt.astimezone(pytz.timezone(T.TZ)) for dt in utc_th]

                        sec = np.array([dt.weekday() * 86400 +
                                       dt.hour * 3600 +
                                       dt.minute * 60 +
                                       dt.second for dt in lt_th])
                        
                        df_int_i["w_sin"] = np.sin(sec / 604800 * 2 * np.pi)
                        df_int_i["w_cos"] = np.cos(sec / 604800 * 2 * np.pi)
                
                elif T.W.SPEC == "Aktuelle Zeit":
                    if T.W.LT == False:
                        wd = utc_ref.weekday()
                        h = utc_ref.hour
                        m = utc_ref.minute
                        sec = wd * 86400 + h * 3600 + m * 60
                        
                        df_int_i["w_sin"] = [np.sin(sec / 604800 * 2 * np.pi)] * mts.I_N
                        df_int_i["w_cos"] = [np.cos(sec / 604800 * 2 * np.pi)] * mts.I_N
                    else:
                        lt = pytz.utc.localize(utc_ref).astimezone(pytz.timezone(T.TZ))
                        sec = lt.weekday() * 86400 + lt.hour * 3600 + lt.minute * 60 + lt.second
                        
                        df_int_i["w_sin"] = [np.sin(sec / 604800 * 2 * np.pi)] * mts.I_N
                        df_int_i["w_cos"] = [np.cos(sec / 604800 * 2 * np.pi)] * mts.I_N
            
            if T.D.IMP:
                if T.D.SPEC == "Zeithorizont":
                    utc_th_strt = utc_ref + datetime.timedelta(hours=T.D.TH_STRT)
                    utc_th_end = utc_ref + datetime.timedelta(hours=T.D.TH_END)
                    
                    try:
                        utc_th = pd.date_range(
                            start=utc_th_strt,
                            end=utc_th_end,
                            freq=f'{T.D.DELT}min'
                        ).to_list()
                    except:
                        delt = pd.to_timedelta(T.D.DELT, unit="min")
                        utc_th = []
                        utc = utc_th_strt
                        for i1 in range(mts.I_N):
                            utc_th.append(utc)
                            utc += delt
                    
                    if T.D.LT == False:
                        h = pd.Series(utc_th).dt.hour.values
                        m = pd.Series(utc_th).dt.minute.values
                        
                        sec = h * 3600 + m * 60
                        df_int_i["d_sin"] = np.sin(sec / 86400 * 2 * np.pi)
                        df_int_i["d_cos"] = np.cos(sec / 86400 * 2 * np.pi)
                    else:
                        utc_th = [pytz.utc.localize(dt) for dt in utc_th]
                        lt_th = [dt.astimezone(pytz.timezone(T.TZ)) for dt in utc_th]

                        sec = np.array([dt.hour * 3600 +
                                       dt.minute * 60 +
                                       dt.second for dt in lt_th])
                        
                        df_int_i["d_sin"] = np.sin(sec / 86400 * 2 * np.pi)
                        df_int_i["d_cos"] = np.cos(sec / 86400 * 2 * np.pi)
                
                elif T.D.SPEC == "Aktuelle Zeit":
                    if T.D.LT == False:
                        h = utc_ref.hour
                        m = utc_ref.minute
                        s = utc_ref.second
                        sec = h * 3600 + m * 60 + s
                        
                        df_int_i["d_sin"] = [np.sin(sec / 86400 * 2 * np.pi)] * mts.I_N
                        df_int_i["d_cos"] = [np.cos(sec / 86400 * 2 * np.pi)] * mts.I_N
                    else:
                        lt = pytz.utc.localize(utc_ref).astimezone(pytz.timezone(T.TZ))
                        sec = lt.hour * 3600 + lt.minute * 60 + lt.second
                        
                        df_int_i["d_sin"] = [np.sin(sec / 86400 * 2 * np.pi)] * mts.I_N
                        df_int_i["d_cos"] = [np.cos(sec / 86400 * 2 * np.pi)] * mts.I_N
            
            if T.H.IMP:
                if T.H.SPEC == "Zeithorizont":
                    # Zeithorizont implementation - check each timestep for holidays
                    utc_th_strt = utc_ref + datetime.timedelta(hours=T.H.TH_STRT)
                    utc_th_end = utc_ref + datetime.timedelta(hours=T.H.TH_END)

                    # Generate timesteps
                    try:
                        utc_th = pd.date_range(
                            start=utc_th_strt,
                            end=utc_th_end,
                            freq=f'{T.H.DELT}min'
                        ).to_list()
                    except:
                        # Fallback: manually create timestamps
                        delt = pd.to_timedelta(T.H.DELT, unit="min")
                        utc_th = []
                        utc = utc_th_strt
                        for i1 in range(mts.I_N):
                            utc_th.append(utc)
                            utc += delt

                    if T.H.LT == False:
                        # Compare UTC dates against holidays
                        df_int_i["h"] = np.array([1 if dt.date() in hol_d else 0 for dt in utc_th])
                    else:
                        # Convert to local time first
                        utc_th_loc = [pytz.utc.localize(dt) if dt.tzinfo is None else dt for dt in utc_th]
                        lt_th = [dt.astimezone(pytz.timezone(T.TZ)) for dt in utc_th_loc]
                        # Compare local dates against holidays
                        df_int_i["h"] = np.array([1 if dt.date() in hol_d else 0 for dt in lt_th])

                elif T.H.SPEC == "Aktuelle Zeit":
                    if T.H.LT == False:
                        df_int_i["h"] = [1 if utc_ref.date() in hol_d else 0] * mts.I_N
                    else:
                        lt = pytz.utc.localize(utc_ref).astimezone(pytz.timezone(T.TZ))
                        df_int_i["h"] = [1 if lt.date() in hol_d else 0] * mts.I_N
        
        
        if df_int_i.shape[1] > 0 and df_int_o.shape[1] > 0:
            i_arrays.append(df_int_i.values)
            o_arrays.append(df_int_o.values)
            utc_ref_log.append(utc_ref)
        else:
            pass
        
        utc_ref = utc_ref + datetime.timedelta(minutes=mts.DELT)
    else:
        error = False
    
    if len(i_arrays) > 0 and len(o_arrays) > 0:
        i_array_3D = np.array(i_arrays)
        o_array_3D = np.array(o_arrays)
        
        n_dat = i_array_3D.shape[0]
        
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
