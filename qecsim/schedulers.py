# from __future__ import annotations
from message import DecodeJob
# ===============================================================================
# SCHEDULERS
# A scheduler is the decode cluster queue ordering policy.
# when a decoder unit frees up and several jobs are waiting, the scheduler decides
# which one to run next. 
# ===============================================================================
# TODO: Simple schedulers for now -- Need to improve in the future
class FifoScheduler:
    """First-in, first-out."""
    def insert(self, queue: list, job: DecodeJob) -> None:
        """Add a job to the back of the queue."""
        queue.append(job)
 
    def pop(self, queue: list) -> DecodeJob:
        """Take the oldest job (first in, first out)."""
        return queue.pop(0)
    
# TODO: Simple schedulers for now -- Need to improve in the future    
class EarliestDeadlineScheduler:
    """Send the job with the nearest deadline first (critical-path aware)."""
    def insert(self, queue: list, job: DecodeJob) -> None:
        """Add a job to the queue."""
        queue.append(job)
 
    def pop(self, queue: list) -> DecodeJob:
        """Take the job with the earliest deadline."""
        queue.sort(key=lambda j: j.deadline)
        return queue.pop(0)