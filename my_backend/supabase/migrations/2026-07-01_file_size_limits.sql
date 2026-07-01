-- Trello #139: per-plan max upload file size.
-- Basic 100 MB, Standard 500 MB, Premium unlimited (-1). internal_unlimited -> -1.
-- Sentinel: -1 = unlimited, 0 = uploads blocked (api_only), n>0 = n MB.
-- Applied to prod luvjebsltuttakatnzaa via MCP migration file_size_limits_2026_07_01.
UPDATE subscription_plans SET max_file_size_mb = 100 WHERE slug = 'basic';
UPDATE subscription_plans SET max_file_size_mb = 500 WHERE slug = 'standard';
UPDATE subscription_plans SET max_file_size_mb = -1  WHERE slug = 'premium';
UPDATE subscription_plans SET max_file_size_mb = -1  WHERE slug = 'internal_unlimited';
