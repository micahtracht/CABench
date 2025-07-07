import time
from threading import Lock

_last_time = 0.0
_lock = Lock()
_min_interval = 1.0

def set_tpm(tpm: int) -> None:
    """Adjust the minimum delay based on requests per minute."""
    global _min_interval
    if tpm <= 0:
        _min_interval = 1.0
    else:
        _min_interval = min(60.0 / tpm, 1.0)

def wait_one_second():
    """Block until `_min_interval` seconds have passed since the last call."""
    global _last_time
    with _lock:
        now = time.monotonic()
        elapsed = now - _last_time
        if elapsed < _min_interval:
            time.sleep(_min_interval - elapsed)
        _last_time = time.monotonic()
