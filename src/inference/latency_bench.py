"""
Measure end-to-end latency: mic capture + buffering + inference + playback.

Owner: hardware track. This is the number that determines if the system
is usable for real-time comms — not raw model inference time alone.
"""

import time


def measure_latency(n_iterations: int = 100):
    # TODO: wrap one full chunk cycle (capture -> inference -> output) in
    # time.perf_counter() calls, run n_iterations times, report mean/p95.
    raise NotImplementedError


if __name__ == "__main__":
    pass
