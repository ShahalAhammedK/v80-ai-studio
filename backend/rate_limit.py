import time
from collections import defaultdict, deque
from threading import Lock

from .config import RATE_LIMIT_PER_MINUTE
from .errors import UserFacingError

_hits: dict[str, deque] = defaultdict(deque)
_lock = Lock()


def check_rate_limit(client_id: str) -> None:
    now = time.time()
    window = 60.0
    with _lock:
        bucket = _hits[client_id]
        while bucket and now - bucket[0] > window:
            bucket.popleft()
        if len(bucket) >= RATE_LIMIT_PER_MINUTE:
            raise UserFacingError(
                "You've made too many requests in a short time. Please wait a minute and try again.", 429
            )
        bucket.append(now)
