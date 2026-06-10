"""
Visualization routes for training API.

Contains 5 endpoints for plots and visualizations:
- plot-variables/<session_id>
- visualizations/<session_id>
- generate-plot
- evaluation-tables/<session_id>
- save-evaluation-tables/<session_id>

W11-A T7: hardened with @limiter.limit, UUID guard, and the
standardized error contract (shared.responses.errors.error_response).
"""

from flask import Blueprint

from .common import (
    datetime, request, jsonify, g, logging,
    require_auth, require_subscription,
    get_supabase_client, create_or_get_session_uuid,
    create_error_response,
    get_logger
)

from core.rate_limits import limiter, training_limit_string
from shared.responses.errors import error_response as _err
from shared.storage.errors import is_storage_not_found
from shared.validators.uuid import validate_training_session_format

from domains.training.services.visualization import Visualizer
from domains.training.constants import calculate_time_deltas
from shared.database.lifecycle import update_workflow_phase
from shared.auth.ownership import assert_session_ownership, SessionOwnershipError

bp = Blueprint('training_visualization', __name__)
logger = get_logger(__name__)


@bp.route('/plot-variables/<session_id>', methods=['GET'])
@limiter.limit(training_limit_string)
@require_auth
def get_plot_variables(session_id):
    """Get available input and output variables for plotting."""
    # W11-BE2: validate session_id BEFORE auth-ed work.
    err = validate_training_session_format(session_id)
    if err:
        return err

    try:
        user_id = g.user_id
        visualizer = Visualizer()
        variables = visualizer.get_available_variables(session_id, user_id=user_id)

        return jsonify({
            'success': True,
            'session_id': session_id,
            'input_variables': variables['input_variables'],
            'output_variables': variables['output_variables'],
            'time_components': variables.get('time_components', [])
        })

    except PermissionError:
        logger.warning("plot-variables: ownership violation", exc_info=True)
        return _err('FORBIDDEN', 'You do not have access to this session', 403)
    except Exception:
        logger.exception("Failed to retrieve plot variables")
        return _err(
            'INTERNAL_ERROR',
            'Failed to retrieve plot variables',
            500,
            suggestion='Please try again. If the problem persists, contact support.',
        )


@bp.route('/visualizations/<session_id>', methods=['GET'])
@limiter.limit(training_limit_string)
@require_auth
def get_training_visualizations(session_id):
    """Get training visualizations (violin plots) for a session."""
    # W11-BE2: validate session_id BEFORE auth-ed work.
    err = validate_training_session_format(session_id)
    if err:
        return err

    try:
        user_id = g.user_id
        visualizer = Visualizer()
        viz_data = visualizer.get_session_visualizations(session_id, user_id=user_id)

        # Fetch n_dat from sessions table
        n_dat = 0
        try:
            supabase = get_supabase_client(use_service_role=True)
            uuid_session_id = create_or_get_session_uuid(session_id, user_id)
            session_response = supabase.table('sessions').select('n_dat').eq('id', uuid_session_id).single().execute()
            if session_response.data:
                n_dat = session_response.data.get('n_dat', 0) or 0
        except Exception:
            logger.warning("Could not fetch n_dat for session", exc_info=True)

        if not viz_data.get('plots'):
            return _err(
                'VISUALIZATION_NOT_FOUND',
                'No visualizations found for this session',
                404,
            )

        return jsonify({
            'session_id': session_id,
            'plots': viz_data['plots'],
            'metadata': viz_data['metadata'],
            'created_at': viz_data['created_at'],
            'n_dat': n_dat,
            'message': viz_data['message']
        })

    except PermissionError:
        logger.warning("visualizations: ownership violation", exc_info=True)
        return _err('FORBIDDEN', 'You do not have access to this session', 403)
    except Exception:
        logger.exception("Failed to retrieve training visualizations")
        return _err(
            'INTERNAL_ERROR',
            'Failed to retrieve training visualizations',
            500,
            suggestion='Please try again. If the problem persists, contact support.',
        )


@bp.route('/generate-plot', methods=['POST'])
@limiter.limit(training_limit_string)
@require_auth
@require_subscription
def generate_plot():
    """Generate plot based on user selections."""
    data = request.get_json(silent=True) or {}
    session_id = data.get('sessionId') or data.get('session_id')
    if not session_id:
        return _err('MISSING_BODY', 'sessionId is required', 400)
    # W11-BE2: validate session_id BEFORE auth-ed work.
    err = validate_training_session_format(session_id)
    if err:
        return err

    try:
        plot_settings = data.get('plot_settings', {})
        df_plot_in = data.get('df_plot_in', {})
        df_plot_out = data.get('df_plot_out', {})
        df_plot_fcst = data.get('df_plot_fcst', {})
        model_id = data.get('model_id')

        visualizer = Visualizer()
        result = visualizer.generate_custom_plot(
            session_id=session_id,
            plot_settings=plot_settings,
            df_plot_in=df_plot_in,
            df_plot_out=df_plot_out,
            df_plot_fcst=df_plot_fcst,
            model_id=model_id
        )

        # [WORKFLOW_DEBUG] Update workflow_phase to 'completed' after plot generation
        # This marks the end of the workflow - refresh will reset to 'upload'
        try:
            uuid_session_id = create_or_get_session_uuid(session_id, g.user_id)
            update_workflow_phase(str(uuid_session_id), 'completed')
            logger.info(f"[WORKFLOW_DEBUG] generate_plot: workflow_phase updated to 'completed' for session {session_id}")
        except Exception:
            logger.exception("[WORKFLOW_DEBUG] Failed to update workflow_phase to completed")

        return jsonify({
            'success': True,
            'session_id': session_id,
            'plot_data': result['plot_data'],
            'message': result['message']
        })

    except ValueError:
        logger.warning("generate-plot: invalid plot params", exc_info=True)
        return _err('INVALID_PLOT_PARAMS', 'Invalid plot parameters', 400)
    except Exception:
        logger.exception("Failed to generate plot")
        return _err(
            'PLOT_GENERATION_ERROR',
            'Failed to generate plot',
            500,
            suggestion='Please try again. If the problem persists, contact support.',
        )


@bp.route('/evaluation-tables/<session_id>', methods=['GET'])
@limiter.limit(training_limit_string)
@require_auth
def get_evaluation_tables(session_id):
    """Get evaluation metrics formatted as tables for display."""
    # W11-BE2: validate session_id BEFORE auth-ed work.
    err = validate_training_session_format(session_id)
    if err:
        return err

    try:
        from utils.training_storage import download_training_results_metrics_only
        supabase = get_supabase_client(use_service_role=True)
        uuid_session_id = create_or_get_session_uuid(session_id, g.user_id)

        try:
            assert_session_ownership(uuid_session_id)
        except SessionOwnershipError:
            logger.warning("evaluation-tables: ownership violation", exc_info=True)
            return _err('FORBIDDEN', 'You do not have access to this session', 403)

        response = supabase.table('training_results') \
            .select('id, results_file_path, compressed, results') \
            .eq('session_id', uuid_session_id) \
            .order('created_at', desc=True) \
            .limit(1) \
            .execute()

        if not response.data:
            return _err(
                'EVALUATION_TABLES_NOT_FOUND',
                'No training results found for this session',
                404,
            )

        record = response.data[0]

        if record.get('results_file_path'):
            try:
                # #112: metrics-only download — skip Keras model reconstruction so a
                # model version-skew deserialization failure cannot hide df_eval, which
                # lives in the same pickle and is all this endpoint needs.
                results = download_training_results_metrics_only(record['results_file_path'])
            except Exception as exc:
                if is_storage_not_found(exc):
                    logger.warning("evaluation-tables: results file missing in storage", exc_info=True)
                    return _err(
                        'EVALUATION_TABLES_NOT_FOUND',
                        'Training results file not found in storage',
                        404,
                    )
                raise
        else:
            results = record.get('results')

        eval_metrics = results.get('evaluation_metrics', {})
        if not eval_metrics or eval_metrics.get('error'):
            eval_metrics = results.get('metrics', {})

        # =====================================================================
        # PRIORITET: Koristi df_eval i df_eval_ts iz 12-level averaging sistema
        # ako postoje (generisani u evaluation.py)
        # =====================================================================
        df_eval = eval_metrics.get('df_eval', {})
        df_eval_ts = eval_metrics.get('df_eval_ts', {})

        # Ako df_eval postoji i nije prazan, koristi ga direktno
        if df_eval and len(df_eval) > 0:

            return jsonify({
                'success': True,
                'session_id': session_id,
                'df_eval': df_eval,
                'df_eval_ts': df_eval_ts,
                'model_type': eval_metrics.get('model_type', 'Unknown')
            })

        # =====================================================================
        # FALLBACK: Rekonstruiši iz _TS metrika (stari način)
        # =====================================================================
        if eval_metrics and eval_metrics.get('test_metrics_scaled'):
            pass
        elif not eval_metrics or (eval_metrics.get('error') and not eval_metrics.get('test_metrics_scaled')):
            return _err(
                'EVALUATION_TABLES_NOT_FOUND',
                'No valid evaluation metrics found',
                404,
            )

        # Get output_features from results or dynamically from files table
        output_features = results.get('output_features', [])
        if not output_features:
            output_features = results.get('output_columns', [])
        if not output_features:
            output_features = results.get('data_info', {}).get('output_columns', [])

        # Fallback: Get from files table if still empty
        if not output_features:
            file_response = supabase.table('files').select('columns, file_type').eq('session_id', uuid_session_id).execute()
            if file_response.data:
                for f in file_response.data:
                    if f.get('file_type') == 'output':
                        columns = f.get('columns', [])
                        output_features = [c for c in columns if c.lower() not in ['timestamp', 'utc', 'zeit', 'datetime']]
                        break

        if not output_features:
            return _err(
                'EVALUATION_TABLES_NOT_FOUND',
                'No output features found for this session',
                404,
            )

        df_eval = {}
        df_eval_ts = {}

        # Get time_deltas dynamically from zeitschritte table
        zeitschritte_response = supabase.table('zeitschritte').select('*').eq('session_id', uuid_session_id).execute()
        zeitschritte = zeitschritte_response.data[0] if zeitschritte_response.data else {}
        time_deltas = calculate_time_deltas(zeitschritte)

        for feature_name in output_features:
            delt_list = []
            mae_list = []
            mape_list = []
            mse_list = []
            rmse_list = []
            nrmse_list = []
            wape_list = []
            smape_list = []
            mase_list = []

            df_eval_ts[feature_name] = {}

            test_metrics = eval_metrics.get('test_metrics_scaled', {})

            # Izvuci _TS metrike (po vremenskim koracima) - ako postoje
            mae_ts = test_metrics.get('MAE_TS', [])
            mape_ts = test_metrics.get('MAPE_TS', [])
            mse_ts = test_metrics.get('MSE_TS', [])
            rmse_ts = test_metrics.get('RMSE_TS', [])
            nrmse_ts = test_metrics.get('NRMSE_TS', [])
            wape_ts = test_metrics.get('WAPE_TS', [])
            smape_ts = test_metrics.get('sMAPE_TS', [])
            mase_ts = test_metrics.get('MASE_TS', [])

            # Broj timestepova - koristi _TS ako postoji, inače fallback
            n_timesteps = len(mae_ts) if mae_ts else len(time_deltas)

            for i in range(min(n_timesteps, len(time_deltas))):
                delta = time_deltas[i]
                delt_list.append(float(delta))

                # Koristi _TS verzije ako postoje, inače fallback na ukupne metrike
                mae_list.append(float(mae_ts[i]) if i < len(mae_ts) else float(test_metrics.get('MAE', 0.0)))
                mape_list.append(float(mape_ts[i]) if i < len(mape_ts) else float(test_metrics.get('MAPE', 0.0)))
                mse_list.append(float(mse_ts[i]) if i < len(mse_ts) else float(test_metrics.get('MSE', 0.0)))
                rmse_list.append(float(rmse_ts[i]) if i < len(rmse_ts) else float(test_metrics.get('RMSE', 0.0)))
                nrmse_list.append(float(nrmse_ts[i]) if i < len(nrmse_ts) else float(test_metrics.get('NRMSE', 0.0)))
                wape_list.append(float(wape_ts[i]) if i < len(wape_ts) else float(test_metrics.get('WAPE', 0.0)))
                smape_list.append(float(smape_ts[i]) if i < len(smape_ts) else float(test_metrics.get('sMAPE', 0.0)))
                mase_list.append(float(mase_ts[i]) if i < len(mase_ts) else float(test_metrics.get('MASE', 0.0)))

                # Timestep metrike za df_eval_ts
                df_eval_ts[feature_name][float(delta)] = {
                    'MAE': mae_list[-1],
                    'MAPE': mape_list[-1],
                    'MSE': mse_list[-1],
                    'RMSE': rmse_list[-1],
                    'NRMSE': nrmse_list[-1],
                    'WAPE': wape_list[-1],
                    'sMAPE': smape_list[-1],
                    'MASE': mase_list[-1]
                }

            df_eval[feature_name] = {
                "delta [min]": delt_list,
                "MAE": mae_list,
                "MAPE": mape_list,
                "MSE": mse_list,
                "RMSE": rmse_list,
                "NRMSE": nrmse_list,
                "WAPE": wape_list,
                "sMAPE": smape_list,
                "MASE": mase_list
            }

        return jsonify({
            'success': True,
            'session_id': session_id,
            'df_eval': df_eval,
            'df_eval_ts': df_eval_ts,
            'model_type': eval_metrics.get('model_type', 'Unknown')
        })

    except Exception:
        logger.exception("Failed to get evaluation tables")
        return _err(
            'INTERNAL_ERROR',
            'Failed to get evaluation tables',
            500,
            suggestion='Please try again. If the problem persists, contact support.',
        )


@bp.route('/save-evaluation-tables/<session_id>', methods=['POST'])
@limiter.limit(training_limit_string)
@require_auth
@require_subscription
def save_evaluation_tables(session_id):
    """Save evaluation tables to database."""
    # W11-BE2: validate session_id BEFORE auth-ed work.
    err = validate_training_session_format(session_id)
    if err:
        return err

    data = request.get_json(silent=True)
    if not data:
        return _err('MISSING_BODY', 'Request body is required', 400)

    df_eval = data.get('df_eval', {})
    df_eval_ts = data.get('df_eval_ts', {})
    model_type = data.get('model_type', 'Unknown')

    if not df_eval and not df_eval_ts:
        return _err('BAD_REQUEST', 'No evaluation tables provided', 400)

    try:
        supabase = get_supabase_client(use_service_role=True)
        uuid_session_id = create_or_get_session_uuid(session_id, g.user_id)

        try:
            assert_session_ownership(uuid_session_id)
        except SessionOwnershipError:
            logger.warning("save-evaluation-tables: ownership violation", exc_info=True)
            return _err('FORBIDDEN', 'You do not have access to this session', 403)

        evaluation_data = {
            'session_id': uuid_session_id,
            'df_eval': df_eval,
            'df_eval_ts': df_eval_ts,
            'model_type': model_type,
            'created_at': datetime.now().isoformat(),
            'table_format': 'original_training_format'
        }

        response = supabase.table('evaluation_tables').upsert(evaluation_data).execute()

        if response.data:
            return jsonify({
                'success': True,
                'session_id': session_id,
                'message': 'Evaluation tables saved successfully',
                'saved_at': evaluation_data['created_at']
            })
        else:
            logger.error("save-evaluation-tables: upsert returned empty data")
            return _err(
                'INTERNAL_ERROR',
                'Failed to save evaluation tables to database',
                500,
                suggestion='Please try again. If the problem persists, contact support.',
            )

    except Exception:
        logger.exception("Failed to save evaluation tables")
        return _err(
            'INTERNAL_ERROR',
            'Failed to save evaluation tables',
            500,
            suggestion='Please try again. If the problem persists, contact support.',
        )
