import logging
import time
from datetime import datetime
import psutil
import os


def setup_logging(log_file='output/analysis.log'):
    """Setup logging configuration"""
    ensure_directory('output')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


class PerformanceMonitor:
    def __init__(self):
        self.start_time = None
        self.phase_times = {}
        self.memory_usage = []
        self.logger = setup_logging()

    def start_phase(self, phase_name: str):
        """Start timing a phase"""
        self.phase_times[phase_name] = {
            'start': time.time(),
            'memory': psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        }
        self.logger.info(f"Starting phase: {phase_name}")

    def end_phase(self, phase_name: str):
        """End timing a phase"""
        if phase_name in self.phase_times:
            end_time = time.time()
            start_time = self.phase_times[phase_name]['start']
            duration = end_time - start_time

            current_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            start_memory = self.phase_times[phase_name]['memory']
            memory_used = current_memory - start_memory

            self.logger.info(f"Completed phase: {phase_name} in {duration:.2f} seconds, "
                             f"Memory used: {memory_used:.2f} MB")

    def start_analysis(self):
        """Start the entire analysis"""
        self.start_time = time.time()
        self.logger.info("Starting factor analysis framework")

    def end_analysis(self):
        """End the entire analysis"""
        if self.start_time:
            total_duration = time.time() - self.start_time
            self.logger.info(f"Analysis completed in {total_duration:.2f} seconds")

            # Log peak memory usage
            process = psutil.Process(os.getpid())
            peak_memory = process.memory_info().rss / 1024 / 1024
            self.logger.info(f"Peak memory usage: {peak_memory:.2f} MB")


class ProgressTracker:
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()

    def update(self, message: str = ""):
        """Update progress"""
        self.current_step += 1
        elapsed = time.time() - self.start_time
        progress = self.current_step / self.total_steps
        eta = (elapsed / progress) * (1 - progress) if progress > 0 else 0

        print(f"[{self.current_step}/{self.total_steps}] {message} "
              f"({progress:.1%}) ETA: {eta:.1f}s")

    def complete(self):
        """Mark as complete"""
        total_time = time.time() - self.start_time
        print(f"Completed in {total_time:.2f} seconds")


def ensure_directory(directory: str):
    """Ensure directory exists (duplicate from helpers for independence)"""
    if not os.path.exists(directory):
        os.makedirs(directory)