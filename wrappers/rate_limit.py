import time
from threading import Lock

_last_time = 0.0
_lock = Lock()

def wait_one_second():
    """Block until at least one second has passed since the last call."""
    global _last_time
    with _lock:
        now = time.monotonic()
        elapsed = now - _last_time
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
        _last_time = time.monotonic()
