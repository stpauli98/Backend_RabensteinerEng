-- Part 2: retire the vestigial dual-write trigger.
--
-- update_usage_tracking() (fired AFTER INSERT ON usage_events) maintained a
-- SECOND, parallel set of usage_tracking rows keyed by a personalized/anniversary
-- period. Those rows are write-only (no view/routine reads them; enforcement reads
-- the app-written calendar rows). Once the app itself moves to anniversary periods
-- (migration ...110...), keeping this trigger would DOUBLE-COUNT processing
-- (app increment_processing_count + this trigger writing the same row).
--
-- usage_events and log_compute_duration are intentionally LEFT IN PLACE — compute
-- hours (get_total_compute_seconds) read usage_events directly and must keep
-- working. We only drop the trigger that derives usage_tracking from events.
DROP TRIGGER IF EXISTS trigger_update_usage_tracking ON public.usage_events;

-- The trigger function is left defined-but-unused (no other caller) so this is
-- reversible by re-creating the trigger if a rollback is ever needed.
