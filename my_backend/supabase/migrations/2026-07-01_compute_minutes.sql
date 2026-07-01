-- Trello #136: compute allowance in minutes (stored as hours).
-- Basic 60 min (1h), Standard 300 min (5h), Premium 600 min (10h).
-- Applied to prod luvjebsltuttakatnzaa via MCP migration compute_minutes_2026_07_01.
UPDATE subscription_plans SET total_compute_hours = 1  WHERE slug = 'basic';
UPDATE subscription_plans SET total_compute_hours = 5  WHERE slug = 'standard';
UPDATE subscription_plans SET total_compute_hours = 10 WHERE slug = 'premium';
