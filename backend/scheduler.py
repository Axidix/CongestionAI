"""
Forecast Scheduler
==================

Triggers periodic forecast refresh.

Usage:
------
    # One-time refresh (for cron or manual)
    python -m backend.scheduler --once
    
    # Daemon mode (for development)
    python -m backend.scheduler --daemon
    
    # Crontab entry (every hour at minute 5)
    5 * * * * cd /Data/CongestionAI && /path/to/python -m backend.scheduler --once >> logs/scheduler.log 2>&1
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Setup logging
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "scheduler.log"),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


# Import after logging setup
from backend.service import refresh_forecast, is_forecast_stale


def run_once() -> bool:
    """
    Run single forecast refresh.
    
    Returns:
        True if successful
    """
    logger.info(f"Starting one-time forecast refresh at {datetime.utcnow()}")
    start = time.time()
    
    success = refresh_forecast()
    
    duration = time.time() - start
    if success:
        logger.info(f"One-time refresh completed in {duration:.1f}s")
    else:
        logger.error(f"One-time refresh FAILED after {duration:.1f}s")
    
    return success


def run_daemon(interval_minutes: int = 60) -> None:
    """
    Run as background scheduler.
    
    Args:
        interval_minutes: Refresh interval (default 60 = hourly)
    """
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.interval import IntervalTrigger
    except ImportError:
        logger.error("APScheduler not installed. Run: pip install apscheduler")
        logger.info("Falling back to simple loop...")
        _run_simple_loop(interval_minutes)
        return
    
    scheduler = BlockingScheduler()
    
    # Run immediately on start
    logger.info("Running initial forecast...")
    run_once()
    
    # Schedule periodic runs
    scheduler.add_job(
        run_once,
        IntervalTrigger(minutes=interval_minutes),
        id="forecast_refresh",
        name="Hourly Forecast Refresh",
        replace_existing=True,
    )
    
    logger.info(f"Scheduler started, refreshing every {interval_minutes} minutes")
    logger.info("Press Ctrl+C to stop")
    
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped")


def _run_simple_loop(interval_minutes: int = 60) -> None:
    """Simple loop fallback if APScheduler not available."""
    logger.info(f"Running simple loop with {interval_minutes}min interval")
    
    while True:
        run_once()
        logger.info(f"Sleeping {interval_minutes} minutes...")
        time.sleep(interval_minutes * 60)


def run_if_stale(max_age_minutes: int = 90) -> bool:
    """
    Run refresh only if forecast is stale.
    
    Useful for startup checks or resilient cron jobs.
    
    Args:
        max_age_minutes: Maximum acceptable forecast age
    
    Returns:
        True if refresh ran and succeeded, False otherwise
    """
    if is_forecast_stale(max_age_minutes):
        logger.info("Forecast is stale, refreshing...")
        return run_once()
    else:
        logger.info("Forecast is fresh, skipping refresh")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Forecast scheduler for continuous inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m backend.scheduler --once           # Run once and exit
  python -m backend.scheduler --daemon         # Run continuously
  python -m backend.scheduler --if-stale       # Run only if forecast is old
  python -m backend.scheduler --daemon --interval 30  # Every 30 minutes
        """
    )
    parser.add_argument(
        "--once", 
        action="store_true", 
        help="Run once and exit"
    )
    parser.add_argument(
        "--daemon", 
        action="store_true", 
        help="Run as continuous daemon"
    )
    parser.add_argument(
        "--if-stale", 
        action="store_true", 
        help="Run only if forecast is stale"
    )
    parser.add_argument(
        "--interval", 
        type=int, 
        default=60,
        help="Refresh interval in minutes for daemon mode (default: 60)"
    )
    parser.add_argument(
        "--max-age",
        type=int,
        default=90,
        help="Max forecast age in minutes for --if-stale (default: 90)"
    )
    
    args = parser.parse_args()
    
    if args.once:
        success = run_once()
        sys.exit(0 if success else 1)
    elif args.daemon:
        run_daemon(interval_minutes=args.interval)
    elif args.if_stale:
        success = run_if_stale(max_age_minutes=args.max_age)
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
#     main()
