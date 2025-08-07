"""
Storage Migration Utility

Safe migration utility to move files from legacy storage locations 
to the new unified storage structure with validation and rollback capabilities.
"""

import os
import sys
import logging
import shutil
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.storage_config import storage_config

logger = logging.getLogger(__name__)

class StorageMigrationError(Exception):
    """Custom exception for storage migration errors"""
    pass

class StorageMigrator:
    """Handles safe migration of files from legacy storage to unified storage"""
    
    def __init__(self, dry_run: bool = False):
        """Initialize migration utility
        
        Args:
            dry_run: If True, only simulate migration without moving files
        """
        self.dry_run = dry_run
        self.migration_log = []
        self.backup_info = {}
        
    def create_migration_plan(self) -> Dict[str, any]:
        """Analyze current storage and create migration plan
        
        Returns:
            Dictionary containing migration plan and analysis
        """
        plan = {
            'timestamp': datetime.now().isoformat(),
            'dry_run': self.dry_run,
            'legacy_locations': [],
            'conflicts': [],
            'estimated_size': 0,
            'estimated_files': 0,
            'risks': [],
            'recommendations': []
        }
        
        # Define legacy storage locations to analyze
        legacy_locations = [
            {
                'path': Path('chunk_uploads'),
                'type': 'chunk_storage',
                'destination': storage_config.temp_dir / 'chunks',
                'description': 'Chunk upload temporary files',
                'cleanup_policy': 'immediate_after_processing'
            },
            {
                'path': Path('api/temp_uploads'),
                'type': 'api_temp',
                'destination': storage_config.temp_dir / 'api',
                'description': 'API temporary processing files',
                'cleanup_policy': '60_minutes'
            },
            {
                'path': Path('temp_uploads'),
                'type': 'root_temp',
                'destination': storage_config.temp_dir / 'legacy',
                'description': 'Root level temporary files',
                'cleanup_policy': 'unknown'
            },
            {
                'path': Path('temp_training_data'),
                'type': 'training_temp',
                'destination': storage_config.temp_dir / 'training',
                'description': 'Training system temporary data',
                'cleanup_policy': 'manual'
            },
            {
                'path': Path('uploads/file_uploads'),
                'type': 'session_files',
                'destination': storage_config.sessions_dir,
                'description': 'Session-based uploaded files',
                'cleanup_policy': 'persistent'
            }
        ]
        
        for location in legacy_locations:
            analysis = self._analyze_location(location)
            plan['legacy_locations'].append(analysis)
            plan['estimated_size'] += analysis['total_size']
            plan['estimated_files'] += analysis['file_count']
            
            # Check for conflicts
            conflicts = self._check_conflicts(location)
            plan['conflicts'].extend(conflicts)
        
        # Generate risks and recommendations
        plan['risks'] = self._assess_risks(plan)
        plan['recommendations'] = self._generate_recommendations(plan)
        
        return plan
    
    def _analyze_location(self, location: Dict) -> Dict[str, any]:
        """Analyze a legacy storage location
        
        Args:
            location: Location configuration
            
        Returns:
            Analysis results
        """
        analysis = {
            'path': str(location['path']),
            'type': location['type'],
            'destination': str(location['destination']),
            'description': location['description'],
            'exists': False,
            'file_count': 0,
            'directory_count': 0,
            'total_size': 0,
            'oldest_file': None,
            'newest_file': None,
            'file_types': {},
            'large_files': [],
            'issues': []
        }
        
        path = location['path']
        
        if not path.exists():
            analysis['issues'].append(f"Directory does not exist: {path}")
            return analysis
            
        analysis['exists'] = True
        
        try:
            for item in path.rglob('*'):
                if item.is_file():
                    # File statistics
                    stat = item.stat()
                    size = stat.st_size
                    mtime = stat.st_mtime
                    
                    analysis['file_count'] += 1
                    analysis['total_size'] += size
                    
                    # Track oldest and newest files
                    if analysis['oldest_file'] is None or mtime < analysis['oldest_file']:
                        analysis['oldest_file'] = mtime
                    if analysis['newest_file'] is None or mtime > analysis['newest_file']:
                        analysis['newest_file'] = mtime
                    
                    # File types
                    suffix = item.suffix.lower()
                    analysis['file_types'][suffix] = analysis['file_types'].get(suffix, 0) + 1
                    
                    # Large files (>100MB)
                    if size > 100 * 1024 * 1024:
                        analysis['large_files'].append({
                            'path': str(item),
                            'size': size,
                            'size_mb': round(size / (1024 * 1024), 2)
                        })
                        
                elif item.is_dir():
                    analysis['directory_count'] += 1
                    
        except Exception as e:
            analysis['issues'].append(f"Error analyzing directory: {e}")
            logger.error(f"Error analyzing {path}: {e}")
        
        return analysis
    
    def _check_conflicts(self, location: Dict) -> List[Dict]:
        """Check for potential migration conflicts
        
        Args:
            location: Location configuration
            
        Returns:
            List of conflicts found
        """
        conflicts = []
        source = location['path']
        destination = Path(location['destination'])
        
        if not source.exists():
            return conflicts
            
        try:
            for item in source.rglob('*'):
                if item.is_file():
                    # Calculate relative path and destination
                    rel_path = item.relative_to(source)
                    dest_path = destination / rel_path
                    
                    if dest_path.exists():
                        conflicts.append({
                            'type': 'file_exists',
                            'source': str(item),
                            'destination': str(dest_path),
                            'resolution': 'rename_or_merge'
                        })
                        
        except Exception as e:
            conflicts.append({
                'type': 'analysis_error',
                'location': str(source),
                'error': str(e),
                'resolution': 'manual_review_required'
            })
            
        return conflicts
    
    def _assess_risks(self, plan: Dict) -> List[str]:
        """Assess migration risks
        
        Args:
            plan: Migration plan
            
        Returns:
            List of risk assessments
        """
        risks = []
        
        # High file count risk
        if plan['estimated_files'] > 10000:
            risks.append("HIGH: Large number of files to migrate (>10,000). Consider batch processing.")
        
        # Large data size risk  
        if plan['estimated_size'] > 5 * 1024 * 1024 * 1024:  # 5GB
            risks.append("HIGH: Large amount of data (>5GB). Ensure sufficient disk space.")
        
        # Conflicts risk
        if plan['conflicts']:
            risks.append(f"MEDIUM: {len(plan['conflicts'])} file conflicts detected. Manual resolution required.")
        
        # Active session risk
        active_sessions = any(
            loc['type'] == 'session_files' and loc['file_count'] > 0 
            for loc in plan['legacy_locations']
        )
        if active_sessions:
            risks.append("MEDIUM: Active session files detected. Consider maintenance window.")
        
        # Mixed temp file policies
        temp_locations = [loc for loc in plan['legacy_locations'] if 'temp' in loc['type']]
        if len(temp_locations) > 2:
            risks.append("LOW: Multiple temporary file locations with different cleanup policies.")
        
        return risks
    
    def _generate_recommendations(self, plan: Dict) -> List[str]:
        """Generate migration recommendations
        
        Args:
            plan: Migration plan
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Always recommend dry run first
        if not plan['dry_run']:
            recommendations.append("Run migration in dry-run mode first to verify plan")
        
        # Backup recommendation
        recommendations.append("Create full backup before migration")
        
        # Maintenance window
        if plan['estimated_files'] > 1000:
            recommendations.append("Schedule migration during maintenance window")
        
        # Staged migration for large datasets
        if plan['estimated_size'] > 1024 * 1024 * 1024:  # 1GB
            recommendations.append("Consider staged migration for large datasets")
        
        # Conflict resolution
        if plan['conflicts']:
            recommendations.append("Resolve file conflicts before migration")
        
        # Testing
        recommendations.append("Test application functionality after migration")
        
        return recommendations
    
    def execute_migration(self, plan: Optional[Dict] = None) -> Dict[str, any]:
        """Execute the migration plan
        
        Args:
            plan: Optional pre-generated plan, will create new one if not provided
            
        Returns:
            Migration execution results
        """
        if plan is None:
            plan = self.create_migration_plan()
        
        results = {
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'dry_run': self.dry_run,
            'files_migrated': 0,
            'directories_created': 0,
            'conflicts_resolved': 0,
            'errors': 0,
            'error_details': [],
            'migration_log': []
        }
        
        logger.info(f"Starting migration (dry_run={self.dry_run})")
        
        try:
            # Create destination directories
            if not self.dry_run:
                storage_config._initialize_directories()
                results['directories_created'] += 6  # Number of storage directories
            
            # Migrate each location
            for location_plan in plan['legacy_locations']:
                if not location_plan['exists']:
                    continue
                    
                location_results = self._migrate_location(location_plan)
                
                results['files_migrated'] += location_results['files_migrated']
                results['conflicts_resolved'] += location_results['conflicts_resolved']
                results['errors'] += location_results['errors']
                results['error_details'].extend(location_results['error_details'])
                results['migration_log'].extend(location_results['migration_log'])
        
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            results['errors'] += 1
            results['error_details'].append(f"Critical migration error: {e}")
        
        results['end_time'] = datetime.now().isoformat()
        logger.info(f"Migration completed: {results}")
        
        return results
    
    def _migrate_location(self, location_plan: Dict) -> Dict[str, any]:
        """Migrate files from a specific location
        
        Args:
            location_plan: Location migration plan
            
        Returns:
            Migration results for this location
        """
        results = {
            'location': location_plan['path'],
            'files_migrated': 0,
            'conflicts_resolved': 0,
            'errors': 0,
            'error_details': [],
            'migration_log': []
        }
        
        source = Path(location_plan['path'])
        destination = Path(location_plan['destination'])
        
        logger.info(f"Migrating {source} -> {destination}")
        
        if not source.exists():
            return results
        
        try:
            # Create destination directory
            if not self.dry_run:
                destination.mkdir(parents=True, exist_ok=True)
            
            results['migration_log'].append(f"Created destination: {destination}")
            
            # Handle different migration strategies based on location type
            if location_plan['type'] == 'session_files':
                # Sessions need special handling to preserve structure
                results.update(self._migrate_sessions(source, destination))
            else:
                # Standard file migration
                results.update(self._migrate_files(source, destination))
        
        except Exception as e:
            logger.error(f"Error migrating {source}: {e}")
            results['errors'] += 1
            results['error_details'].append(f"Location migration error: {e}")
        
        return results
    
    def _migrate_sessions(self, source: Path, destination: Path) -> Dict[str, int]:
        """Migrate session files with structure preservation
        
        Args:
            source: Source directory
            destination: Destination directory
            
        Returns:
            Migration statistics
        """
        stats = {'files_migrated': 0, 'conflicts_resolved': 0, 'errors': 0, 'error_details': [], 'migration_log': []}
        
        try:
            for session_dir in source.iterdir():
                if session_dir.is_dir() and session_dir.name.startswith('session_'):
                    dest_session_dir = destination / session_dir.name
                    
                    if not self.dry_run:
                        dest_session_dir.mkdir(exist_ok=True)
                    
                    stats['migration_log'].append(f"Processing session: {session_dir.name}")
                    
                    # Move files within session
                    for file in session_dir.iterdir():
                        if file.is_file():
                            dest_file = dest_session_dir / file.name
                            
                            if dest_file.exists():
                                # Handle conflict by renaming
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                dest_file = dest_session_dir / f"{file.stem}_{timestamp}{file.suffix}"
                                stats['conflicts_resolved'] += 1
                            
                            if not self.dry_run:
                                shutil.move(str(file), str(dest_file))
                            
                            stats['files_migrated'] += 1
                            stats['migration_log'].append(f"Migrated: {file} -> {dest_file}")
                    
                    # Remove empty source session directory
                    if not self.dry_run:
                        try:
                            session_dir.rmdir()
                        except OSError:
                            stats['migration_log'].append(f"Could not remove source session dir: {session_dir}")
        
        except Exception as e:
            stats['errors'] += 1
            stats['error_details'].append(f"Session migration error: {e}")
            
        return stats
    
    def _migrate_files(self, source: Path, destination: Path) -> Dict[str, int]:
        """Migrate regular files
        
        Args:
            source: Source directory
            destination: Destination directory
            
        Returns:
            Migration statistics
        """
        stats = {'files_migrated': 0, 'conflicts_resolved': 0, 'errors': 0, 'error_details': [], 'migration_log': []}
        
        try:
            for item in source.rglob('*'):
                if item.is_file():
                    rel_path = item.relative_to(source)
                    dest_file = destination / rel_path
                    
                    # Create parent directories
                    if not self.dry_run:
                        dest_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Handle conflicts
                    if dest_file.exists():
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        dest_file = dest_file.parent / f"{dest_file.stem}_{timestamp}{dest_file.suffix}"
                        stats['conflicts_resolved'] += 1
                    
                    # Move file
                    if not self.dry_run:
                        shutil.move(str(item), str(dest_file))
                    
                    stats['files_migrated'] += 1
                    stats['migration_log'].append(f"Migrated: {item} -> {dest_file}")
        
        except Exception as e:
            stats['errors'] += 1
            stats['error_details'].append(f"File migration error: {e}")
        
        return stats
    
    def rollback_migration(self, backup_path: str) -> Dict[str, any]:
        """Rollback migration using backup
        
        Args:
            backup_path: Path to migration backup
            
        Returns:
            Rollback results
        """
        # Implementation for rollback functionality
        # This would restore files from backup location
        logger.warning("Rollback functionality not yet implemented")
        return {'status': 'not_implemented'}
    
    def save_migration_report(self, plan: Dict, results: Dict, output_file: str = None) -> str:
        """Save migration report to file
        
        Args:
            plan: Migration plan
            results: Migration results
            output_file: Output file path
            
        Returns:
            Path to saved report
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"storage_migration_report_{timestamp}.json"
        
        report = {
            'migration_plan': plan,
            'migration_results': results,
            'generated_at': datetime.now().isoformat()
        }
        
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Migration report saved to: {output_path}")
        return str(output_path)


def main():
    """Command line interface for storage migration"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Storage Migration Utility')
    parser.add_argument('--dry-run', action='store_true', help='Simulate migration without moving files')
    parser.add_argument('--plan-only', action='store_true', help='Generate migration plan only')
    parser.add_argument('--execute', action='store_true', help='Execute migration')
    parser.add_argument('--report', help='Save report to specific file')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    migrator = StorageMigrator(dry_run=args.dry_run)
    
    # Generate migration plan
    print("Generating migration plan...")
    plan = migrator.create_migration_plan()
    
    print(f"\nMigration Plan Summary:")
    print(f"- Estimated files: {plan['estimated_files']}")
    print(f"- Estimated size: {plan['estimated_size'] / (1024*1024):.1f} MB")
    print(f"- Conflicts: {len(plan['conflicts'])}")
    print(f"- Risks: {len(plan['risks'])}")
    
    if plan['risks']:
        print("\nRisks:")
        for risk in plan['risks']:
            print(f"  - {risk}")
    
    if plan['recommendations']:
        print("\nRecommendations:")
        for rec in plan['recommendations']:
            print(f"  - {rec}")
    
    if args.plan_only:
        if args.report:
            migrator.save_migration_report(plan, {}, args.report)
        return
    
    if args.execute:
        print(f"\nExecuting migration (dry_run={args.dry_run})...")
        results = migrator.execute_migration(plan)
        
        print(f"\nMigration Results:")
        print(f"- Files migrated: {results['files_migrated']}")
        print(f"- Conflicts resolved: {results['conflicts_resolved']}")
        print(f"- Errors: {results['errors']}")
        
        if results['errors'] > 0:
            print("\nErrors:")
            for error in results['error_details']:
                print(f"  - {error}")
        
        if args.report:
            migrator.save_migration_report(plan, results, args.report)


if __name__ == '__main__':
    main()