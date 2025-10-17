#!/bin/bash

# FAZA 1: Brisanje Debug/Test Endpointa iz training.py
# SAFE TO RUN - ovi endpointi su samo za debugging

set -e  # Exit on error

echo "🗑️  Brisanje nekorištenih debug/test endpointa iz training.py"
echo "================================================================"
echo ""

TRAINING_FILE="api/routes/training.py"
BACKUP_FILE="api/routes/training.py.backup_$(date +%Y%m%d_%H%M%S)"

# Create backup
echo "📦 Creating backup: $BACKUP_FILE"
cp "$TRAINING_FILE" "$BACKUP_FILE"
echo "✅ Backup created"
echo ""

# Function to remove endpoint and its function
remove_endpoint() {
    local start_pattern="$1"
    local end_pattern="$2"
    local name="$3"

    echo "🔍 Removing: $name"

    # Use sed to remove the endpoint
    # This is a placeholder - actual implementation would need more sophisticated approach
    echo "   ⚠️  Manual removal recommended for complex functions"
}

echo "📝 Endpoints to remove:"
echo ""
echo "1. /debug-env (line ~1549)"
echo "   - GET endpoint"
echo "   - Returns environment variables"
echo "   - SAFE to delete"
echo ""

echo "2. /debug-files-table/<session_id> (line ~2782)"
echo "   - GET endpoint"
echo "   - Debug info for files table"
echo "   - SAFE to delete"
echo ""

echo "3. /test-data-loading/<session_id> (line ~1599)"
echo "   - GET endpoint"
echo "   - Test endpoint for data loading"
echo "   - SAFE to delete"
echo ""

echo "4. /cleanup-uploads (line ~3631)"
echo "   - POST endpoint"
echo "   - Manual cleanup trigger"
echo "   - VERIFY before delete"
echo ""

echo "5. /scalers/<session_id>/info (line ~3651)"
echo "   - GET endpoint"
echo "   - Duplicate of main scalers endpoint"
echo "   - SAFE to delete"
echo ""

echo "================================================================"
echo ""
echo "⚠️  MANUAL STEPS RECOMMENDED:"
echo ""
echo "1. Open: $TRAINING_FILE"
echo "2. Search for each endpoint pattern (use line numbers above)"
echo "3. Delete the @bp.route decorator AND the function"
echo "4. Save the file"
echo "5. Run tests: pytest tests/test_training_endpoints.py -v"
echo "6. If tests pass → commit"
echo "7. If tests fail → restore from backup: cp $BACKUP_FILE $TRAINING_FILE"
echo ""

echo "📋 Quick reference for manual deletion:"
echo ""
echo "# Pattern 1: debug-env"
echo '@bp.route('\''/debug-env'\'', methods=['\''GET'\''])'
echo "def debug_env():"
echo ""
echo "# Pattern 2: debug-files-table"
echo '@bp.route('\''/debug-files-table/<session_id>'\'', methods=['\''GET'\''])'
echo "def debug_files_table(session_id):"
echo ""
echo "# Pattern 3: test-data-loading"
echo '@bp.route('\''/test-data-loading/<session_id>'\'', methods=['\''GET'\''])'
echo "def test_data_loading(session_id):"
echo ""
echo "# Pattern 4: cleanup-uploads"
echo '@bp.route('\''/cleanup-uploads'\'', methods=['\''POST'\''])'
echo "def cleanup_uploads():"
echo ""
echo "# Pattern 5: scalers info"
echo '@bp.route('\''/scalers/<session_id>/info'\'', methods=['\''GET'\''])'
echo "def get_scalers_info(session_id):"
echo ""

echo "================================================================"
echo "✅ Backup completed at: $BACKUP_FILE"
echo "⏭️  Proceed with manual deletion"
echo ""
