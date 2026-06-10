from __future__ import annotations

from .message import DecodeJob
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
    """Send the job with the nearest deadline first (critical-path aware). Only meaningful
    when the cluster's DeadlinePolicy assigns real deadlines (e.g. ReactionPathDeadline);
    under the default EnqueueTimeDeadline every deadline equals the enqueue time and this
    scheduler behaves like FIFO."""
    def insert(self, queue: list, job: DecodeJob) -> None:
        """Add a job to the queue."""
        queue.append(job)

    def pop(self, queue: list) -> DecodeJob:
        """Take the job with the earliest deadline."""
        queue.sort(key=lambda j: j.deadline)
        return queue.pop(0)


# ===============================================================================
# DEADLINE POLICIES
# A DeadlinePolicy assigns each window job its deadline at enqueue time -- the
# knob that gives EarliestDeadlineScheduler something real to order by. This is
# a workload-manager policy seam (arXiv:2511.10633 Sec III): the cluster queues
# decoding jobs, and WHICH job should run first under contention is a policy.
# ===============================================================================

class EnqueueTimeDeadline:
    """The default policy: deadline = enqueue time. Reproduces the original behaviour
    exactly (EDF degenerates to FIFO); swap in ReactionPathDeadline for a real policy."""
    def deadline(self, op, window, now: int, on_reaction_path: bool) -> int:
        """Return the enqueue time (all jobs equally urgent)."""
        return now


class ReactionPathDeadline:
    """Reaction-path-first: a window whose operation's decode result GATES a waiting
    non-Clifford gate gets a tight deadline (now); every other window gets `now + slack`.
    Under contention, EDF then runs reaction-path windows first -- attacking the reaction
    time gamma that arXiv:2511.10633 shows dominates utility-scale runtime (the T-gate
    cannot proceed until the gating decode returns), without starving other windows
    (slack-aged jobs eventually win)."""
    def __init__(self, slack_ticks: int):
        """`slack_ticks` is how long a non-reaction-path window may be deferred (e.g. one
        logical cycle = d * round time)."""
        self.slack_ticks = int(slack_ticks)

    def deadline(self, op, window, now: int, on_reaction_path: bool) -> int:
        """Tight deadline on the reaction path; now + slack off it."""
        return now if on_reaction_path else now + self.slack_ticks
