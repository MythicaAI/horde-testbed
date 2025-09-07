from collections import deque
import time
from multiprocessing import Process
from typing import Callable, Dict, Iterable, List, Optional, Tuple


# Standardized exit code for CUDA OOM from workers
OOM_EXIT_CODE = 42


def run_job_queue(
    jobs: Iterable[Dict],
    spawn_fn: Callable[[Dict], Process],
    *,
    sleep_seconds: float = 5.0,
    poll_results: Optional[Callable[[], None]] = None,
) -> None:
    """
    Launch a set of jobs using a user-provided spawn_fn that returns a started Process.
    Handles OOM requeue with backpressure and general non-zero exits by requeuing.

    - jobs: iterable of job dicts
    - spawn_fn: function that accepts a job dict, starts a Process, and returns it
    - sleep_seconds: monitor tick interval
    - poll_results: optional callback executed each tick (and on worker exit) to drain result queues
    """
    queue = deque(jobs)
    running: List[Tuple[Process, Dict]] = []
    mem_full = False

    def launch_next():
        nonlocal mem_full
        if not queue or mem_full:
            return
        job = queue.popleft()
        p = spawn_fn(job)
        running.append((p, job))
        time.sleep(sleep_seconds)
        launch_next()

    launch_next()

    while queue or any(p.is_alive() for p, _ in running):
        still_running: List[Tuple[Process, Dict]] = []
        for p, job in running:
            if p.is_alive():
                still_running.append((p, job))
            else:
                exitcode = p.exitcode
                if poll_results is not None:
                    poll_results()
                if exitcode == OOM_EXIT_CODE:
                    print(f"Job {job.get('name','<unnamed>')} OOM, requeuing.")
                    mem_full = True
                    queue.appendleft(job)
                elif exitcode != 0:
                    print(f"Job {job.get('name','<unnamed>')} exited with code {exitcode}, requeuing.")
                    queue.append(job)
                else:
                    print(f"Job {job.get('name','<unnamed>')} completed successfully.")
                    mem_full = False
                    launch_next()
        running[:] = still_running
        if poll_results is not None:
            poll_results()
        time.sleep(sleep_seconds)

