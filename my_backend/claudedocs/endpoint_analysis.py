#!/usr/bin/env python3
"""
Endpoint Analysis Script
Analyzes all backend endpoints and identifies unused ones
"""

import re
from pathlib import Path
from collections import defaultdict

# Define blueprint prefixes from __init__.py
BLUEPRINT_PREFIXES = {
    'data_processing': '/api/dataProcessingMain',  # Already in routes
    'load_data': '/api/loadRowData',
    'first_processing': '/api/firstProcessing',
    'cloud': '/api/cloud',
    'adjustments': '/api/adjustmentsOfData',
    'training': '/api/training',
}

# Known used endpoints from training-api-endpoints.md
TRAINING_USED_ENDPOINTS = {
    '/api/training/generate-datasets/<session_id>',
    '/api/training/train-models/<session_id>',
    '/api/training/start-complete-pipeline/<session_id>',
    '/api/training/get-training-status/<session_id>',
    '/api/training/pipeline-overview/<session_id>',
    '/api/training/results/<session_id>',
    '/api/training/comprehensive-evaluation/<session_id>',
    '/api/training/save-model/<session_id>',
    '/api/training/list-models/<session_id>',
    '/api/training/download-model/<session_id>',
    '/api/training/list-models-database/<session_id>',
    '/api/training/download-model-h5/<session_id>',
    '/api/training/evaluation-tables/<session_id>',
    '/api/training/visualizations/<session_id>',
    '/api/training/plot-variables/<session_id>',
    '/api/training/generate-plot',
    '/api/training/csv-files/<session_id>',
    '/api/training/csv-files',
    '/api/training/csv-files/<file_id>',
    '/api/training/get-time-info/<session_id>',
    '/api/training/save-time-info',
    '/api/training/get-zeitschritte/<session_id>',
    '/api/training/save-zeitschritte',
    '/api/training/list-sessions',
    '/api/training/session-name-change',
    '/api/training/session/<session_id>/delete',
    '/api/training/session/<session_id>/database',
    '/api/training/session-status/<session_id>',
    '/api/training/create-database-session',
    '/api/training/delete-all-sessions',
    '/api/training/scalers/<session_id>',
    '/api/training/scalers/<session_id>/download',
    '/api/training/status/<session_id>',
    '/api/training/init-session',
    '/api/training/upload-chunk',
    '/api/training/finalize-session',
    '/api/training/get-session-uuid/<session_id>',
}

def extract_endpoints_from_file(file_path, blueprint_name):
    """Extract all endpoints from a route file"""
    endpoints = []

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find all @bp.route decorators
    pattern = r"@bp\.route\(['\"]([^'\"]+)['\"].*?\)\s*\ndef\s+(\w+)"
    matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)

    prefix = BLUEPRINT_PREFIXES.get(blueprint_name, '')

    for route, func_name in matches:
        full_path = prefix + route
        # Normalize path parameters
        full_path = re.sub(r'<[^>]+>', lambda m: '<' + m.group(0).split(':')[-1], full_path)

        endpoints.append({
            'path': full_path,
            'route': route,
            'function': func_name,
            'file': file_path.name,
            'blueprint': blueprint_name
        })

    return endpoints

def analyze_endpoints():
    """Main analysis function"""
    routes_dir = Path('api/routes')

    all_endpoints = []
    by_blueprint = defaultdict(list)

    # Analyze each route file
    for route_file in routes_dir.glob('*.py'):
        if route_file.name == '__init__.py':
            continue

        blueprint_name = route_file.stem
        endpoints = extract_endpoints_from_file(route_file, blueprint_name)

        all_endpoints.extend(endpoints)
        by_blueprint[blueprint_name].extend(endpoints)

    # Categorize endpoints
    used_training = []
    unused_training = []
    other_endpoints = []

    for ep in all_endpoints:
        if ep['blueprint'] == 'training':
            # Normalize for comparison
            normalized = ep['path']
            normalized = re.sub(r'<[^>]+>', lambda m: '<' + m.group(0).split(':')[-1].replace('>', ''), normalized)

            # Check multiple variations
            is_used = any(
                normalized in used or
                normalized.replace('<', '{').replace('>', '}') in used or
                ep['route'] in used
                for used in [str(u) for u in TRAINING_USED_ENDPOINTS]
            )

            if is_used:
                used_training.append(ep)
            else:
                unused_training.append(ep)
        else:
            other_endpoints.append(ep)

    return {
        'all': all_endpoints,
        'by_blueprint': by_blueprint,
        'used_training': used_training,
        'unused_training': unused_training,
        'other': other_endpoints,
        'total': len(all_endpoints)
    }

def generate_report(results):
    """Generate analysis report"""
    report = []
    report.append("=" * 80)
    report.append("ENDPOINT ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")

    # Summary
    report.append("📊 SUMMARY")
    report.append("-" * 80)
    report.append(f"Total Endpoints Found: {results['total']}")
    report.append(f"Training Endpoints (Used): {len(results['used_training'])}")
    report.append(f"Training Endpoints (Unused): {len(results['unused_training'])}")
    report.append(f"Other Endpoints: {len(results['other'])}")
    report.append("")

    # Breakdown by blueprint
    report.append("📁 ENDPOINTS BY BLUEPRINT")
    report.append("-" * 80)
    for blueprint, endpoints in sorted(results['by_blueprint'].items()):
        report.append(f"\n{blueprint.upper()}: {len(endpoints)} endpoints")
        for ep in sorted(endpoints, key=lambda x: x['route']):
            report.append(f"  • {ep['path']:<60} [{ep['function']}]")

    report.append("")
    report.append("")

    # Unused training endpoints
    report.append("❌ UNUSED TRAINING ENDPOINTS (SAFE TO REMOVE)")
    report.append("-" * 80)
    if results['unused_training']:
        for ep in sorted(results['unused_training'], key=lambda x: x['route']):
            report.append(f"  • {ep['path']:<60} [{ep['function']}]")
    else:
        report.append("  ✅ No unused training endpoints found!")

    report.append("")
    report.append("")

    # Other endpoints (need manual check)
    report.append("⚠️  NON-TRAINING ENDPOINTS (NEED MANUAL VERIFICATION)")
    report.append("-" * 80)
    report.append("These endpoints are NOT in training module - manual check needed:")
    report.append("")

    by_bp = defaultdict(list)
    for ep in results['other']:
        by_bp[ep['blueprint']].append(ep)

    for blueprint, endpoints in sorted(by_bp.items()):
        report.append(f"\n{blueprint.upper()}: {len(endpoints)} endpoints")
        for ep in sorted(endpoints, key=lambda x: x['route']):
            report.append(f"  • {ep['path']:<60} [{ep['function']}]")

    return "\n".join(report)

if __name__ == '__main__':
    print("🔍 Analyzing backend endpoints...")
    results = analyze_endpoints()
    report = generate_report(results)

    # Print to console
    print(report)

    # Save to file
    output_file = Path('claudedocs/ENDPOINT_ANALYSIS.md')
    output_file.write_text(f"```\n{report}\n```\n")
    print(f"\n✅ Report saved to: {output_file}")
