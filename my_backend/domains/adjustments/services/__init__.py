# Adjustments Domain Services

from domains.adjustments.services.state_manager import (
    adjustment_chunks,
    adjustment_chunks_timestamps,
    chunk_buffer,
    chunk_buffer_timestamps,
    stored_data,
    stored_data_timestamps,
    info_df_cache,
    info_df_cache_timestamps,
    cleanup_all_expired_data,
    get_file_info_from_cache,
    check_files_need_methods
)

from domains.adjustments.services.progress import (
    ProgressStages,
    ProgressTracker,
    emit_progress,
    emit_file_result,
    emit_file_error
)

from domains.adjustments.services.processing import (
    apply_processing_method,
    prepare_data,
    filter_by_time_range,
    get_method_for_file,
    create_info_record,
    create_records,
    convert_data_without_processing,
    process_data_detailed
)

from domains.adjustments.services.utils import (
    detect_delimiter,
    get_time_column,
    sanitize_filename,
    analyse_data,
    info_df
)

__all__ = [
    # State management
    'adjustment_chunks',
    'adjustment_chunks_timestamps',
    'chunk_buffer',
    'chunk_buffer_timestamps',
    'stored_data',
    'stored_data_timestamps',
    'info_df_cache',
    'info_df_cache_timestamps',
    'cleanup_all_expired_data',
    'get_file_info_from_cache',
    'check_files_need_methods',

    # Progress tracking
    'ProgressStages',
    'ProgressTracker',
    'emit_progress',
    'emit_file_result',
    'emit_file_error',

    # Processing
    'apply_processing_method',
    'prepare_data',
    'filter_by_time_range',
    'get_method_for_file',
    'create_info_record',
    'create_records',
    'convert_data_without_processing',
    'process_data_detailed',

    # Utils
    'detect_delimiter',
    'get_time_column',
    'sanitize_filename',
    'analyse_data',
    'info_df'
]
