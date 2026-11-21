"""Process-tree resource sampling for bounded benchmark runs (issue #1269)."""

from __future__ import annotations

import resource
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import psutil


@dataclass(frozen=True)
class ResourceSample:
    """Peak and average resources observed over one benchmark invocation.

    ``cpu_cores_avg`` is a deliberate LOWER BOUND: it takes the max of (a)
    per-pid cpu_times deltas between consecutive samples (misses children that
    live and die between samples) and (b) RUSAGE_CHILDREN deltas (misses
    children still alive at the end). When both populations contribute, their
    union is not reconstructable, so the reported value can undercount a mixed
    workload — never overcount.
    """

    threads_max: int
    rss_peak_bytes: int
    pss_peak_bytes: int | None
    swap_peak_bytes: int | None
    cpu_cores_avg: float
    wall_seconds: float


def sample_process_tree[T](
    fn: Callable[[], T], interval_s: float = 0.5
) -> tuple[T, ResourceSample]:
    """Run ``fn`` while sampling the process and all live descendants."""
    process = psutil.Process()
    lock = threading.Lock()
    stop = threading.Event()
    threads_max = 0
    rss_peak_bytes = 0
    pss_peak_bytes = 0
    swap_peak_bytes = 0
    observed_pss = False
    observed_swap = False
    sampled_cpu_seconds = 0.0
    previous_cpu: dict[int, float] = {}

    def collect() -> None:
        nonlocal threads_max, rss_peak_bytes, pss_peak_bytes, swap_peak_bytes
        nonlocal observed_pss, observed_swap, sampled_cpu_seconds
        try:
            processes = [process, *process.children(recursive=True)]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            processes = [process]
        current_cpu: dict[int, float] = {}
        thread_count = 0
        rss_bytes = 0
        pss_bytes = 0
        swap_bytes = 0
        memory_processes = 0
        pss_processes = 0
        swap_processes = 0
        for candidate in processes:
            try:
                thread_count += candidate.num_threads()
                rss_bytes += candidate.memory_info().rss
                memory_processes += 1
                try:
                    full_memory = candidate.memory_full_info()
                except (psutil.NoSuchProcess, psutil.AccessDenied, NotImplementedError):
                    full_memory = None
                if full_memory is not None and hasattr(full_memory, "pss"):
                    pss_bytes += int(full_memory.pss)
                    pss_processes += 1
                if full_memory is not None and hasattr(full_memory, "swap"):
                    swap_bytes += int(full_memory.swap)
                    swap_processes += 1
                cpu = candidate.cpu_times()
                current_cpu[candidate.pid] = cpu.user + cpu.system
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        with lock:
            threads_max = max(threads_max, thread_count)
            rss_peak_bytes = max(rss_peak_bytes, rss_bytes)
            sample_has_pss = memory_processes > 0 and pss_processes == memory_processes
            sample_has_swap = memory_processes > 0 and swap_processes == memory_processes
            if sample_has_pss:
                observed_pss = True
                pss_peak_bytes = max(pss_peak_bytes, pss_bytes)
            if sample_has_swap:
                observed_swap = True
                swap_peak_bytes = max(swap_peak_bytes, swap_bytes)
            for pid, cpu_seconds in current_cpu.items():
                previous = previous_cpu.get(pid)
                if previous is not None and cpu_seconds >= previous:
                    sampled_cpu_seconds += cpu_seconds - previous
            previous_cpu.clear()
            previous_cpu.update(current_cpu)

    def sampler() -> None:
        while not stop.is_set():
            collect()
            stop.wait(interval_s)

    child_usage_start = resource.getrusage(resource.RUSAGE_CHILDREN)
    sampler_thread = threading.Thread(
        target=sampler, name="process-tree-resource-sampler", daemon=True
    )
    # Baseline sample BEFORE the thread starts: cpu_cores_avg is built from
    # per-pid deltas between consecutive samples, so without a synchronous
    # baseline a GIL-hogging in-process ``fn`` can starve the sampler until the
    # final post-join collect, which would then only establish the baseline and
    # report zero CPU.
    collect()
    started = time.perf_counter()
    sampler_thread.start()
    try:
        result = fn()
    finally:
        stop.set()
        sampler_thread.join()
        collect()
    wall_seconds = time.perf_counter() - started
    child_usage_end = resource.getrusage(resource.RUSAGE_CHILDREN)
    reaped_cpu_seconds = (child_usage_end.ru_utime - child_usage_start.ru_utime) + (
        child_usage_end.ru_stime - child_usage_start.ru_stime
    )
    with lock:
        cpu_seconds = max(sampled_cpu_seconds, reaped_cpu_seconds)
        sample = ResourceSample(
            threads_max=threads_max,
            rss_peak_bytes=rss_peak_bytes,
            pss_peak_bytes=pss_peak_bytes if observed_pss else None,
            swap_peak_bytes=swap_peak_bytes if observed_swap else None,
            cpu_cores_avg=cpu_seconds / wall_seconds if wall_seconds > 0 else 0.0,
            wall_seconds=wall_seconds,
        )
    return result, sample
