create table public.csv_file_refs (
  id uuid not null default gen_random_uuid (),
  file_id uuid not null,
  session_id uuid not null,
  file_name character varying(255) null,
  storage_path character varying(500) null,
  file_size bigint null,
  created_at timestamp with time zone null default now(),
  updated_at timestamp with time zone null default now(),
  constraint csv_file_refs_pkey primary key (id),
  constraint csv_file_refs_file_id_fkey foreign KEY (file_id) references files (id) on delete CASCADE,
  constraint csv_file_refs_session_id_fkey foreign KEY (session_id) references sessions (id) on delete CASCADE
) TABLESPACE pg_default;

create index IF not exists idx_csv_file_refs_file_id on public.csv_file_refs using btree (file_id) TABLESPACE pg_default;

create index IF not exists idx_csv_file_refs_session_id on public.csv_file_refs using btree (session_id) TABLESPACE pg_default;

create trigger update_csv_file_refs_updated_at BEFORE
update on csv_file_refs for EACH row
execute FUNCTION update_updated_at_column ();
___________________________________________________________________________
create table public.files (
  id uuid not null default gen_random_uuid (),
  session_id uuid not null,
  file_name character varying(255) null,
  bezeichnung character varying(255) null,
  min character varying(100) null,
  max character varying(100) null,
  offsett character varying(100) null,
  datenpunkte character varying(100) null,
  numerische_datenpunkte character varying(100) null,
  numerischer_anteil character varying(100) null,
  datenform character varying(100) null,
  datenanpassung character varying(100) null,
  zeitschrittweite character varying(100) null,
  zeitschrittweite_mittelwert character varying(100) null,
  zeitschrittweite_min character varying(100) null,
  skalierung character varying(100) null default 'nein'::character varying,
  skalierung_max character varying(100) null,
  skalierung_min character varying(100) null,
  zeithorizont_start character varying(100) null,
  zeithorizont_end character varying(100) null,
  zeitschrittweite_transferierten_daten character varying(100) null,
  offset_transferierten_daten character varying(100) null,
  mittelwertbildung_uber_den_zeithorizont character varying(100) null default 'nein'::character varying,
  storage_path character varying(500) null,
  type character varying(50) null,
  utc_min timestamp with time zone null,
  utc_max timestamp with time zone null,
  created_at timestamp with time zone null default now(),
  updated_at timestamp with time zone null default now(),
  constraint files_pkey primary key (id),
  constraint files_session_id_fkey foreign KEY (session_id) references sessions (id) on delete CASCADE
) TABLESPACE pg_default;

create index IF not exists idx_files_session_id on public.files using btree (session_id) TABLESPACE pg_default;

create index IF not exists idx_files_type on public.files using btree (type) TABLESPACE pg_default;

create trigger update_files_updated_at BEFORE
update on files for EACH row
execute FUNCTION update_updated_at_column ();

___________________________________________________________________________

create table public.session_mappings (
  id uuid not null default gen_random_uuid (),
  string_session_id character varying(255) not null,
  uuid_session_id uuid not null,
  created_at timestamp with time zone null default now(),
  constraint session_mappings_pkey primary key (id),
  constraint session_mappings_string_session_id_key unique (string_session_id),
  constraint session_mappings_uuid_session_id_key unique (uuid_session_id),
  constraint session_mappings_uuid_session_id_fkey foreign KEY (uuid_session_id) references sessions (id) on delete CASCADE
) TABLESPACE pg_default;

create index IF not exists idx_session_mappings_string_id on public.session_mappings using btree (string_session_id) TABLESPACE pg_default;

create index IF not exists idx_session_mappings_uuid_id on public.session_mappings using btree (uuid_session_id) TABLESPACE pg_default;

___________________________________________________________________________

create table public.sessions (
  id uuid not null default gen_random_uuid (),
  created_at timestamp with time zone null default now(),
  updated_at timestamp with time zone null default now(),
  constraint sessions_pkey primary key (id)
) TABLESPACE pg_default;

create index IF not exists idx_sessions_created_at on public.sessions using btree (created_at) TABLESPACE pg_default;

create trigger update_sessions_updated_at BEFORE
update on sessions for EACH row
execute FUNCTION update_updated_at_column ();


create table public.time_info (
  id uuid not null default gen_random_uuid (),
  session_id uuid not null,
  jahr boolean null default false,
  woche boolean null default false,
  monat boolean null default false,
  feiertag boolean null default false,
  tag boolean null default false,
  zeitzone character varying(100) null default 'UTC'::character varying,
  category_data jsonb null default '{}'::jsonb,
  created_at timestamp with time zone null default now(),
  updated_at timestamp with time zone null default now(),
  constraint time_info_pkey primary key (id),
  constraint time_info_session_id_key unique (session_id),
  constraint time_info_session_id_fkey foreign KEY (session_id) references sessions (id) on delete CASCADE
) TABLESPACE pg_default;

create index IF not exists idx_time_info_session_id on public.time_info using btree (session_id) TABLESPACE pg_default;

create trigger update_time_info_updated_at BEFORE
update on time_info for EACH row
execute FUNCTION update_updated_at_column ();


___________________________________________________________________________

create table public.training_logs (
  id uuid not null default gen_random_uuid (),
  session_id uuid not null,
  level character varying(20) not null default 'INFO'::character varying,
  message text not null,
  step character varying(100) null,
  model_name character varying(50) null,
  details jsonb null,
  batch_id uuid null,
  sequence_number integer null,
  created_at timestamp with time zone null default now(),
  step_number integer null,
  step_name character varying(255) null,
  progress_percentage integer null,
  constraint training_logs_pkey primary key (id),
  constraint training_logs_session_id_fkey foreign KEY (session_id) references sessions (id) on delete CASCADE
) TABLESPACE pg_default;

create index IF not exists idx_training_logs_batch_sequence on public.training_logs using btree (batch_id, sequence_number) TABLESPACE pg_default;

create index IF not exists idx_training_logs_created_at on public.training_logs using btree (created_at) TABLESPACE pg_default;

create index IF not exists idx_training_logs_level on public.training_logs using btree (level) TABLESPACE pg_default;

create index IF not exists idx_training_logs_level_created on public.training_logs using btree (level, created_at) TABLESPACE pg_default;

create index IF not exists idx_training_logs_session_id on public.training_logs using btree (session_id) TABLESPACE pg_default;

___________________________________________________________________________

create table public.training_progress (
  id uuid not null default gen_random_uuid (),
  session_id uuid not null,
  overall_progress integer null default 0,
  current_step character varying(100) null,
  total_steps integer null default 7,
  completed_steps integer null default 0,
  step_details jsonb null,
  model_progress jsonb null,
  estimated_time_remaining integer null,
  status character varying(20) null default 'idle'::character varying,
  process_id character varying(100) null,
  process_info jsonb null,
  last_heartbeat timestamp with time zone null default now(),
  started_at timestamp with time zone null,
  completed_at timestamp with time zone null,
  created_at timestamp with time zone null default now(),
  updated_at timestamp with time zone null default now(),
  step_number integer null,
  step_name character varying(255) null,
  error_message text null,
  constraint training_progress_pkey primary key (id),
  constraint training_progress_session_id_key unique (session_id),
  constraint training_progress_session_id_fkey foreign KEY (session_id) references sessions (id) on delete CASCADE,
  constraint training_progress_overall_progress_check check (
    (
      (overall_progress >= 0)
      and (overall_progress <= 100)
    )
  ),
  constraint training_progress_status_check check (
    (
      (status)::text = any (
        array[
          ('idle'::character varying)::text,
          ('running'::character varying)::text,
          ('completed'::character varying)::text,
          ('failed'::character varying)::text,
          ('abandoned'::character varying)::text
        ]
      )
    )
  )
) TABLESPACE pg_default;

create index IF not exists idx_training_progress_session_id on public.training_progress using btree (session_id) TABLESPACE pg_default;

create index IF not exists idx_training_progress_heartbeat on public.training_progress using btree (last_heartbeat) TABLESPACE pg_default;

create index IF not exists idx_training_progress_process_id on public.training_progress using btree (process_id) TABLESPACE pg_default;

create index IF not exists idx_training_progress_status_heartbeat on public.training_progress using btree (status, last_heartbeat) TABLESPACE pg_default;

create trigger update_training_progress_updated_at BEFORE
update on training_progress for EACH row
execute FUNCTION update_updated_at_column ();


create table public.training_results (
  id uuid not null default gen_random_uuid (),
  session_id uuid not null,
  status character varying(50) not null default 'pending'::character varying,
  results jsonb null,
  evaluation_metrics jsonb null,
  model_comparison jsonb null,
  training_metadata jsonb null,
  best_model_info jsonb null,
  error_message text null,
  error_traceback text null,
  processing_started_by character varying(100) null,
  processing_info jsonb null,
  heartbeat_history jsonb null,
  started_at timestamp with time zone null default now(),
  completed_at timestamp with time zone null,
  created_at timestamp with time zone null default now(),
  updated_at timestamp with time zone null default now(),
  model_performance jsonb null,
  best_model jsonb null,
  summary jsonb null,
  dataset_count integer null,
  train_dataset_size integer null,
  val_dataset_size integer null,
  test_dataset_size integer null,
  dataset_generation_time double precision null,
  datasets_info jsonb null,
  mts_configuration jsonb null,
  processing_summary jsonb null,
  constraint training_results_pkey primary key (id),
  constraint training_results_session_id_fkey foreign KEY (session_id) references sessions (id) on delete CASCADE
) TABLESPACE pg_default;

create index IF not exists idx_training_results_created_at on public.training_results using btree (created_at) TABLESPACE pg_default;

create index IF not exists idx_training_results_processing on public.training_results using btree (processing_started_by, status) TABLESPACE pg_default;

create index IF not exists idx_training_results_session_id on public.training_results using btree (session_id) TABLESPACE pg_default;

create index IF not exists idx_training_results_status on public.training_results using btree (status) TABLESPACE pg_default;

create trigger update_training_results_updated_at BEFORE
update on training_results for EACH row
execute FUNCTION update_updated_at_column ();


___________________________________________________________________________

create table public.training_visualizations (
  id uuid not null default gen_random_uuid (),
  session_id uuid not null,
  plot_type character varying(50) not null,
  plot_name character varying(100) not null,
  dataset_name character varying(100) null,
  model_name character varying(50) null,
  image_data text null,
  storage_path character varying(500) null,
  plot_metadata jsonb null,
  created_at timestamp with time zone null default now(),
  plot_data_base64 text null,
  metadata jsonb null,
  constraint training_visualizations_pkey primary key (id),
  constraint training_visualizations_session_id_fkey foreign KEY (session_id) references sessions (id) on delete CASCADE
) TABLESPACE pg_default;

create index IF not exists idx_training_visualizations_plot_type on public.training_visualizations using btree (plot_type) TABLESPACE pg_default;

create index IF not exists idx_training_visualizations_session_id on public.training_visualizations using btree (session_id) TABLESPACE pg_default;



create table public.zeitschritte (
  id uuid not null default gen_random_uuid (),
  session_id uuid not null,
  eingabe character varying(100) null,
  ausgabe character varying(100) null,
  zeitschrittweite character varying(100) null,
  offsett character varying(100) null,
  created_at timestamp with time zone null default now(),
  updated_at timestamp with time zone null default now(),
  constraint zeitschritte_pkey primary key (id),
  constraint zeitschritte_session_id_key unique (session_id),
  constraint zeitschritte_session_id_fkey foreign KEY (session_id) references sessions (id) on delete CASCADE
) TABLESPACE pg_default;

create index IF not exists idx_zeitschritte_session_id on public.zeitschritte using btree (session_id) TABLESPACE pg_default;

create trigger update_zeitschritte_updated_at BEFORE
update on zeitschritte for EACH row
execute FUNCTION update_updated_at_column ();

