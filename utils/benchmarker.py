"""
benchmarker.py
----------------
Tracks execution time and RAM usage for any function call.
Used to demonstrate the exponential resource wall as qubit count grows.
"""

import time
import psutil
import os


def benchmark(func, *args, **kwargs):
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)  # MB
    start = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start
    mem_after = process.memory_info().rss / (1024 * 1024)
    mem_used = mem_after - mem_before
    print(f"Execution Time: {elapsed:.4f}s | RAM Delta: {mem_used:.2f}MB | Total RAM: {mem_after:.2f}MB")
    return result