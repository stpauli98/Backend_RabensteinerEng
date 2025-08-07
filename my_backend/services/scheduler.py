"""
Scheduler Service

Centralized scheduling service for recurring tasks including:
- Storage cleanup operations
- Health checks and monitoring
- System maintenance tasks
"""

import logging
import time
from typing import Dict, Any, Callable, Optional
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

from config.storage_config import storage_config

logger = logging.getLogger(__name__)

class SchedulerService:
    """Centralized scheduler for system maintenance tasks"""
    
    def __init__(self, app=None):
        """Initialize scheduler service
        
        Args:
            app: Flask application instance (optional)
        """
        self.app = app
        self.scheduler = None
        self.is_running = False
        self.job_stats = {}
        
        # Configure scheduler
        executors = {
            'default': ThreadPoolExecutor(max_workers=3),
        }
        job_defaults = {
            'coalesce': True,
            'max_instances': 1,
            'misfire_grace_time': 300  # 5 minutes
        }
        
        self.scheduler = BackgroundScheduler(
            executors=executors,
            job_defaults=job_defaults,
            daemon=True
        )
        
        # Register default jobs
        self._register_default_jobs()
    
    def _register_default_jobs(self):
        """Register default system maintenance jobs"""
        
        # Storage cleanup - every 30 minutes
        self.scheduler.add_job(
            func=self._storage_cleanup_job,
            trigger=IntervalTrigger(minutes=30),
            id='storage_cleanup',
            name='Storage Cleanup',
            replace_existing=True
        )
        
        # Storage health check - every 15 minutes
        self.scheduler.add_job(
            func=self._storage_health_job,
            trigger=IntervalTrigger(minutes=15),
            id='storage_health',
            name='Storage Health Check',
            replace_existing=True
        )
        
        # Daily system summary - at 00:00
        self.scheduler.add_job(
            func=self._daily_summary_job,
            trigger=CronTrigger(hour=0, minute=0),
            id='daily_summary',
            name='Daily System Summary',
            replace_existing=True
        )
        
        # Weekly deep cleanup - Sunday at 02:00
        self.scheduler.add_job(
            func=self._weekly_cleanup_job,
            trigger=CronTrigger(day_of_week='sun', hour=2, minute=0),
            id='weekly_cleanup',
            name='Weekly Deep Cleanup',
            replace_existing=True
        )
        
        logger.info("Registered default scheduler jobs")
    
    def _storage_cleanup_job(self):
        """Scheduled storage cleanup job"""
        job_name = "storage_cleanup"
        start_time = time.time()
        
        try:
            logger.info("Starting scheduled storage cleanup")
            
            # Run cleanup
            cleanup_stats = storage_config.cleanup_expired_files()
            
            # Update job statistics
            execution_time = time.time() - start_time
            self.job_stats[job_name] = {
                'last_run': datetime.now().isoformat(),
                'execution_time': execution_time,
                'status': 'success',
                'stats': cleanup_stats
            }
            
            total_removed = sum([
                cleanup_stats.get('temp_files_removed', 0),
                cleanup_stats.get('session_files_removed', 0),
                cleanup_stats.get('processed_files_removed', 0),
                cleanup_stats.get('cache_files_removed', 0)
            ])
            
            logger.info(f"Storage cleanup completed: {total_removed} files removed in {execution_time:.2f}s")
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.job_stats[job_name] = {
                'last_run': datetime.now().isoformat(),
                'execution_time': execution_time,
                'status': 'error',
                'error': str(e)
            }
            logger.error(f"Storage cleanup failed: {e}")
    
    def _storage_health_job(self):
        """Scheduled storage health check job"""
        job_name = "storage_health"
        start_time = time.time()
        
        try:
            logger.debug("Running storage health check")
            
            # Get storage statistics
            storage_stats = storage_config.get_storage_stats()
            
            # Update job statistics
            execution_time = time.time() - start_time
            self.job_stats[job_name] = {
                'last_run': datetime.now().isoformat(),
                'execution_time': execution_time,
                'status': 'success',
                'stats': storage_stats
            }
            
            # Check for warnings
            health_status = storage_stats.get('storage_health', 'unknown')
            if health_status == 'warning':
                logger.warning(f"Storage usage at warning level: {storage_stats.get('total_size', 0) / (1024**3):.2f}GB")
            elif health_status == 'critical':
                logger.error(f"Storage usage at critical level: {storage_stats.get('total_size', 0) / (1024**3):.2f}GB")
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.job_stats[job_name] = {
                'last_run': datetime.now().isoformat(),
                'execution_time': execution_time,
                'status': 'error',
                'error': str(e)
            }
            logger.error(f"Storage health check failed: {e}")
    
    def _daily_summary_job(self):
        """Scheduled daily summary job"""
        job_name = "daily_summary"
        start_time = time.time()
        
        try:
            logger.info("Generating daily system summary")
            
            # Get storage statistics
            storage_stats = storage_config.get_storage_stats()
            
            # Generate summary
            summary = {
                'date': datetime.now().date().isoformat(),
                'storage_usage': {
                    'total_size_gb': storage_stats.get('total_size', 0) / (1024**3),
                    'health_status': storage_stats.get('storage_health', 'unknown'),
                    'directories': storage_stats.get('directories', {}),
                    'file_counts': storage_stats.get('file_counts', {})
                },
                'job_performance': {
                    name: stats for name, stats in self.job_stats.items()
                    if stats.get('last_run') and \
                       datetime.fromisoformat(stats['last_run']).date() == datetime.now().date()
                }
            }
            
            # Save summary to logs
            logger.info(f"Daily Summary: {summary}")
            
            # Update job statistics
            execution_time = time.time() - start_time
            self.job_stats[job_name] = {
                'last_run': datetime.now().isoformat(),
                'execution_time': execution_time,
                'status': 'success',
                'summary': summary
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.job_stats[job_name] = {
                'last_run': datetime.now().isoformat(),
                'execution_time': execution_time,
                'status': 'error',
                'error': str(e)
            }
            logger.error(f"Daily summary generation failed: {e}")
    
    def _weekly_cleanup_job(self):
        """Scheduled weekly deep cleanup job"""
        job_name = "weekly_cleanup"
        start_time = time.time()
        
        try:
            logger.info("Starting weekly deep cleanup")
            
            # Force cleanup of all expired files regardless of age
            cleanup_stats = storage_config.cleanup_expired_files(force=True)
            
            # Get storage stats before and after
            final_stats = storage_config.get_storage_stats()
            
            execution_time = time.time() - start_time
            self.job_stats[job_name] = {
                'last_run': datetime.now().isoformat(),
                'execution_time': execution_time,
                'status': 'success',
                'cleanup_stats': cleanup_stats,
                'final_storage_stats': final_stats
            }
            
            total_removed = sum([
                cleanup_stats.get('temp_files_removed', 0),
                cleanup_stats.get('session_files_removed', 0),
                cleanup_stats.get('processed_files_removed', 0),
                cleanup_stats.get('cache_files_removed', 0)
            ])
            
            logger.info(f"Weekly cleanup completed: {total_removed} files removed in {execution_time:.2f}s")
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.job_stats[job_name] = {
                'last_run': datetime.now().isoformat(),
                'execution_time': execution_time,
                'status': 'error',
                'error': str(e)
            }
            logger.error(f"Weekly cleanup failed: {e}")
    
    def start(self):
        """Start the scheduler"""
        if not self.is_running:
            try:
                self.scheduler.start()
                self.is_running = True
                logger.info("Scheduler service started")
                
                # Log registered jobs
                jobs = self.scheduler.get_jobs()
                logger.info(f"Active scheduler jobs: {[job.id for job in jobs]}")
                
            except Exception as e:
                logger.error(f"Failed to start scheduler: {e}")
                raise
    
    def stop(self):
        """Stop the scheduler"""
        if self.is_running:
            try:
                self.scheduler.shutdown(wait=True)
                self.is_running = False
                logger.info("Scheduler service stopped")
            except Exception as e:
                logger.error(f"Error stopping scheduler: {e}")
    
    def add_job(self, func: Callable, trigger: str = 'interval', **kwargs):
        """Add a custom job to the scheduler
        
        Args:
            func: Function to execute
            trigger: Trigger type ('interval', 'cron', 'date')
            **kwargs: Additional arguments for the trigger and job
        """
        try:
            if trigger == 'interval':
                trigger_obj = IntervalTrigger(**{k: v for k, v in kwargs.items() if k in ['seconds', 'minutes', 'hours', 'days']})
            elif trigger == 'cron':
                trigger_obj = CronTrigger(**{k: v for k, v in kwargs.items() if k in ['year', 'month', 'day', 'week', 'day_of_week', 'hour', 'minute', 'second']})
            else:
                raise ValueError(f"Unsupported trigger type: {trigger}")
            
            job_kwargs = {k: v for k, v in kwargs.items() if k in ['id', 'name', 'replace_existing']}
            
            self.scheduler.add_job(func=func, trigger=trigger_obj, **job_kwargs)
            logger.info(f"Added custom job: {job_kwargs.get('id', func.__name__)}")
            
        except Exception as e:
            logger.error(f"Failed to add job: {e}")
            raise
    
    def remove_job(self, job_id: str):
        """Remove a job from the scheduler
        
        Args:
            job_id: Job identifier
        """
        try:
            self.scheduler.remove_job(job_id)
            logger.info(f"Removed job: {job_id}")
        except Exception as e:
            logger.error(f"Failed to remove job {job_id}: {e}")
    
    def get_job_stats(self) -> Dict[str, Any]:
        """Get scheduler job statistics
        
        Returns:
            Dictionary with job statistics
        """
        return {
            'scheduler_status': 'running' if self.is_running else 'stopped',
            'active_jobs': len(self.scheduler.get_jobs()) if self.scheduler else 0,
            'job_stats': self.job_stats
        }
    
    def trigger_job(self, job_id: str) -> bool:
        """Manually trigger a scheduled job
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if job was triggered successfully
        """
        try:
            job = self.scheduler.get_job(job_id)
            if job:
                job.modify(next_run_time=datetime.now())
                logger.info(f"Triggered job: {job_id}")
                return True
            else:
                logger.warning(f"Job not found: {job_id}")
                return False
        except Exception as e:
            logger.error(f"Failed to trigger job {job_id}: {e}")
            return False


# Global scheduler instance
scheduler_service = SchedulerService()

# Convenience functions for Flask integration
def init_scheduler(app):
    """Initialize scheduler with Flask app"""
    scheduler_service.app = app
    return scheduler_service

def start_scheduler():
    """Start the scheduler service"""
    scheduler_service.start()

def stop_scheduler():
    """Stop the scheduler service"""
    scheduler_service.stop()

def get_scheduler_stats():
    """Get scheduler statistics"""
    return scheduler_service.get_job_stats()