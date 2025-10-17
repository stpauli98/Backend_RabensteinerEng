```
================================================================================
ENDPOINT ANALYSIS REPORT
================================================================================

📊 SUMMARY
--------------------------------------------------------------------------------
Total Endpoints Found: 59
Training Endpoints (Used): 26
Training Endpoints (Unused): 11
Other Endpoints: 22

📁 ENDPOINTS BY BLUEPRINT
--------------------------------------------------------------------------------

ADJUSTMENTS: 5 endpoints
  • /api/adjustmentsOfData/adjust-data-chunk                     [adjust_data]
  • /api/adjustmentsOfData/adjustdata/complete                   [complete_adjustment]
  • /api/adjustmentsOfData/download/<<file_id>                   [download_file]
  • /api/adjustmentsOfData/prepare-save                          [prepare_save]
  • /api/adjustmentsOfData/upload-chunk                          [upload_chunk]

CLOUD: 6 endpoints
  • /api/cloud/clouddata                                         [clouddata]
  • /api/cloud/complete                                          [complete_redirect]
  • /api/cloud/download/<<file_id>                               [download_file]
  • /api/cloud/interpolate-chunked                               [interpolate_chunked]
  • /api/cloud/prepare-save                                      [prepare_save]
  • /api/cloud/upload-chunk                                      [upload_chunk]

DATA_PROCESSING: 3 endpoints
  • /api/dataProcessingMain/api/dataProcessingMain/download/<<file_id> [download_file]
  • /api/dataProcessingMain/api/dataProcessingMain/prepare-save  [prepare_save]
  • /api/dataProcessingMain/api/dataProcessingMain/upload-chunk  [upload_chunk]

FIRST_PROCESSING: 3 endpoints
  • /api/firstProcessing/download/<<file_id>                     [download_file]
  • /api/firstProcessing/prepare-save                            [prepare_save]
  • /api/firstProcessing/upload_chunk                            [upload_chunk]

LOAD_DATA: 5 endpoints
  • /api/loadRowData/cancel-upload                               [cancel_upload]
  • /api/loadRowData/download/<<file_id>                         [download_file]
  • /api/loadRowData/finalize-upload                             [finalize_upload]
  • /api/loadRowData/prepare-save                                [prepare_save]
  • /api/loadRowData/upload-chunk                                [upload_chunk]

TRAINING: 37 endpoints
  • /api/training/cleanup-uploads                                [cleanup_uploads]
  • /api/training/create-database-session                        [create_database_session]
  • /api/training/csv-files                                      [create_csv_file]
  • /api/training/csv-files/<<file_id>                           [update_csv_file]
  • /api/training/csv-files/<<file_id>                           [delete_csv_file]
  • /api/training/csv-files/<<session_id>                        [get_csv_files]
  • /api/training/debug-env                                      [debug_env]
  • /api/training/debug-files-table/<<session_id>                [debug_files_table]
  • /api/training/delete-all-sessions                            [delete_all_sessions]
  • /api/training/evaluation-tables/<<session_id>                [get_evaluation_tables]
  • /api/training/file/download/<<session_id>/<<file_type>/<<file_name> [download_file]
  • /api/training/finalize-session                               [finalize_session]
  • /api/training/generate-datasets/<<session_id>                [generate_datasets]
  • /api/training/get-all-files-metadata/<<session_id>           [get_all_files_metadata]
  • /api/training/get-file-metadata/<<session_id>                [get_file_metadata]
  • /api/training/get-session-uuid/<<session_id>                 [get_session_uuid]
  • /api/training/get-time-info/<<session_id>                    [get_time_info]
  • /api/training/get-zeitschritte/<<session_id>                 [get_zeitschritte]
  • /api/training/init-session                                   [init_session]
  • /api/training/list-sessions                                  [list_sessions]
  • /api/training/results/<<session_id>                          [get_training_results]
  • /api/training/run-analysis/<<session_id>                     [run_analysis]
  • /api/training/save-evaluation-tables/<<session_id>           [save_evaluation_tables]
  • /api/training/save-time-info                                 [save_time_info_endpoint]
  • /api/training/save-zeitschritte                              [save_zeitschritte_endpoint]
  • /api/training/scale-data/<<session_id>                       [scale_input_data]
  • /api/training/scalers/<<session_id>                          [get_scalers]
  • /api/training/scalers/<<session_id>/download                 [download_scalers_as_save_files]
  • /api/training/scalers/<<session_id>/info                     [get_scalers_info]
  • /api/training/session-name-change                            [change_session_name]
  • /api/training/session-status/<<session_id>                   [session_status]
  • /api/training/session/<<session_id>                          [get_session]
  • /api/training/session/<<session_id>/database                 [get_session_from_database]
  • /api/training/session/<<session_id>/delete                   [delete_session]
  • /api/training/test-data-loading/<<session_id>                [test_data_loading]
  • /api/training/train-models/<<session_id>                     [train_models]
  • /api/training/upload-chunk                                   [upload_chunk]


❌ UNUSED TRAINING ENDPOINTS (SAFE TO REMOVE)
--------------------------------------------------------------------------------
  • /api/training/cleanup-uploads                                [cleanup_uploads]
  • /api/training/debug-env                                      [debug_env]
  • /api/training/debug-files-table/<<session_id>                [debug_files_table]
  • /api/training/file/download/<<session_id>/<<file_type>/<<file_name> [download_file]
  • /api/training/get-all-files-metadata/<<session_id>           [get_all_files_metadata]
  • /api/training/get-file-metadata/<<session_id>                [get_file_metadata]
  • /api/training/run-analysis/<<session_id>                     [run_analysis]
  • /api/training/save-evaluation-tables/<<session_id>           [save_evaluation_tables]
  • /api/training/scale-data/<<session_id>                       [scale_input_data]
  • /api/training/scalers/<<session_id>/info                     [get_scalers_info]
  • /api/training/test-data-loading/<<session_id>                [test_data_loading]


⚠️  NON-TRAINING ENDPOINTS (NEED MANUAL VERIFICATION)
--------------------------------------------------------------------------------
These endpoints are NOT in training module - manual check needed:


ADJUSTMENTS: 5 endpoints
  • /api/adjustmentsOfData/adjust-data-chunk                     [adjust_data]
  • /api/adjustmentsOfData/adjustdata/complete                   [complete_adjustment]
  • /api/adjustmentsOfData/download/<<file_id>                   [download_file]
  • /api/adjustmentsOfData/prepare-save                          [prepare_save]
  • /api/adjustmentsOfData/upload-chunk                          [upload_chunk]

CLOUD: 6 endpoints
  • /api/cloud/clouddata                                         [clouddata]
  • /api/cloud/complete                                          [complete_redirect]
  • /api/cloud/download/<<file_id>                               [download_file]
  • /api/cloud/interpolate-chunked                               [interpolate_chunked]
  • /api/cloud/prepare-save                                      [prepare_save]
  • /api/cloud/upload-chunk                                      [upload_chunk]

DATA_PROCESSING: 3 endpoints
  • /api/dataProcessingMain/api/dataProcessingMain/download/<<file_id> [download_file]
  • /api/dataProcessingMain/api/dataProcessingMain/prepare-save  [prepare_save]
  • /api/dataProcessingMain/api/dataProcessingMain/upload-chunk  [upload_chunk]

FIRST_PROCESSING: 3 endpoints
  • /api/firstProcessing/download/<<file_id>                     [download_file]
  • /api/firstProcessing/prepare-save                            [prepare_save]
  • /api/firstProcessing/upload_chunk                            [upload_chunk]

LOAD_DATA: 5 endpoints
  • /api/loadRowData/cancel-upload                               [cancel_upload]
  • /api/loadRowData/download/<<file_id>                         [download_file]
  • /api/loadRowData/finalize-upload                             [finalize_upload]
  • /api/loadRowData/prepare-save                                [prepare_save]
  • /api/loadRowData/upload-chunk                                [upload_chunk]
```
