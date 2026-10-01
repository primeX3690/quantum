"""
run_tests.py -- dependency-free test runner.
Runs every tests/test_*.py::test_* function (same tests pytest would run),
so the suite works even where pytest is not installed:

    python run_tests.py            # or:  pytest -q
"""
import importlib
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def main():
    files = sorted(f[:-3] for f in os.listdir(os.path.join(ROOT, "tests"))
                   if f.startswith("test_") and f.endswith(".py"))
    passed, failed = 0, []
    t0 = time.time()
    for mod_name in files:
        mod = importlib.import_module(f"tests.{mod_name}")
        for name in sorted(dir(mod)):
            if not name.startswith("test_"):
                continue
            fn = getattr(mod, name)
            t1 = time.time()
            try:
                fn()
                passed += 1
                print(f"  PASS  {mod_name}::{name}  ({time.time() - t1:.2f}s)")
            except Exception:
                failed.append(f"{mod_name}::{name}")
                print(f"  FAIL  {mod_name}::{name}")
                traceback.print_exc()
    print(f"\n{passed} passed, {len(failed)} failed in {time.time() - t0:.1f}s")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
