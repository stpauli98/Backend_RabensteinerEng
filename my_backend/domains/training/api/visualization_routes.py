"""
Visualization routes for training API.

Contains 5 endpoints for plots and visualizations:
- plot-variables/<session_id>
- visualizations/<session_id>
- generate-plot
- evaluation-tables/<session_id>
- save-evaluation-tables/<session_id>
"""

from flask import Blueprint

from .common import (
    datetime, request, jsonify, g, logging,
    require_auth, require_subscription,
    get_supabase_client, create_or_get_session_uuid,
    create_error_response,
    get_logger
)

from domains.training.services.visualization import Visualizer

bp = Blueprint('training_visualization', __name__)
logger = get_logger(__name__)


@bp.route('/plot-variables/<session_id>', methods=['GET'])
@require_auth
def get_plot_variables(session_id):
    """Get available input and output variables for plotting."""
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

    except PermissionError as e:
        return jsonify({'success': False, 'error': str(e)}), 403
    except Exception as e:
        logger.error(f"Error getting plot variables for {session_id}: {str(e)}")
        return jsonify({
            'success': True,
            'session_id': session_id,
            'input_variables': ['Temperature', 'Load'],
            'output_variables': ['Predicted_Load'],
            'time_components': ['Y_sin', 'Y_cos', 'M_sin', 'M_cos', 'W_sin', 'W_cos', 'D_sin', 'D_cos', 'Holiday']
        })


@bp.route('/visualizations/<session_id>', methods=['GET'])
@require_auth
def get_training_visualizations(session_id):
    """Get training visualizations (violin plots) for a session."""
    try:
        user_id = g.user_id
        visualizer = Visualizer()
        viz_data = visualizer.get_session_visualizations(session_id, user_id=user_id)

        if not viz_data.get('plots'):
            return jsonify({
                'session_id': session_id,
                'plots': {},
                'message': viz_data.get('message', 'No visualizations found for this session')
            }), 404

        return jsonify({
            'session_id': session_id,
            'plots': viz_data['plots'],
            'metadata': viz_data['metadata'],
            'created_at': viz_data['created_at'],
            'message': viz_data['message']
        })

    except PermissionError as e:
        return create_error_response(str(e), 403)
    except Exception as e:
        logger.error(f"Error retrieving visualizations for {session_id}: {str(e)}")
        return create_error_response(f'Failed to retrieve training visualizations: {str(e)}', 500)


@bp.route('/generate-plot', methods=['POST'])
@require_auth
@require_subscription
def generate_plot():
    """Generate plot based on user selections."""
    try:
        data = request.json
        session_id = data.get('session_id')

        if not session_id:
            return create_error_response('Session ID is required', 400)

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

        return jsonify({
            'success': True,
            'session_id': session_id,
            'plot_data': result['plot_data'],
            'message': result['message']
        })

    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    except Exception as e:
        logger.error(f"Error in generate_plot endpoint: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return create_error_response(f'Failed to generate plot: {str(e)}', 500)


@bp.route('/evaluation-tables/<session_id>', methods=['GET'])
@require_auth
def get_evaluation_tables(session_id):
    """Get evaluation metrics formatted as tables for display."""
    try:
        from utils.training_storage import download_training_results
        supabase = get_supabase_client(use_service_role=True)
        uuid_session_id = create_or_get_session_uuid(session_id, g.user_id)

        response = supabase.table('training_results') \
            .select('id, results_file_path, compressed, results') \
            .eq('session_id', uuid_session_id) \
            .order('created_at.desc') \
            .limit(1) \
            .execute()

        if not response.data:
            return jsonify({
                'success': False,
                'error': 'No training results found for this session',
                'session_id': session_id
            }), 404

        record = response.data[0]

        if record.get('results_file_path'):
            results = download_training_results(
                file_path=record['results_file_path'],
                decompress=record.get('compressed', False)
            )
        else:
            results = record.get('results')

        eval_metrics = results.get('evaluation_metrics', {})
        if not eval_metrics or eval_metrics.get('error'):
            eval_metrics = results.get('metrics', {})

        if eval_metrics and eval_metrics.get('test_metrics_scaled'):
            pass
        elif not eval_metrics or (eval_metrics.get('error') and not eval_metrics.get('test_metrics_scaled')):
            return jsonify({
                'success': False,
                'error': f"No valid evaluation metrics found. Metrics: {eval_metrics}",
                'session_id': session_id
            }), 404

        output_features = results.get('output_features', ['Netzlast [kW]'])
        if not output_features:
            output_features = ['Netzlast [kW]']

        df_eval = {}
        df_eval_ts = {}

        time_deltas = [15, 30, 45, 60, 90, 120, 180, 240, 300, 360, 420, 480]

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

    except Exception as e:
        logger.error(f"Error getting evaluation tables: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'session_id': session_id
        }), 500


@bp.route('/save-evaluation-tables/<session_id>', methods=['POST'])
@require_auth
@require_subscription
def save_evaluation_tables(session_id):
    """Save evaluation tables to database."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400

        df_eval = data.get('df_eval', {})
        df_eval_ts = data.get('df_eval_ts', {})
        model_type = data.get('model_type', 'Unknown')

        if not df_eval and not df_eval_ts:
            return jsonify({
                'success': False,
                'error': 'No evaluation tables provided'
            }), 400

        supabase = get_supabase_client(use_service_role=True)
        uuid_session_id = create_or_get_session_uuid(session_id, g.user_id)

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
            return jsonify({
                'success': False,
                'error': 'Failed to save to database'
            }), 500

    except Exception as e:
        logger.error(f"Error saving evaluation tables: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'session_id': session_id
        }), 500
