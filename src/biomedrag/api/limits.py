"""Two small guards that keep a free demo from running out of its free allowance."""
import threading
import time
from collections import defaultdict, deque
from datetime import UTC, datetime


class RateLimiter:
    """At most `per_minute` requests per visitor in any rolling 60 seconds. Lives in memory, so a
    restart forgets everyone, which is fine for a demo."""

    def __init__(self, per_minute: int, clock=time.monotonic):
        self.per_minute, self.clock = per_minute, clock
        self.hits: dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()
        self.calls = 0

    def allow(self, key: str) -> bool:
        with self.lock:
            now = self.clock()
            self.calls += 1
            if self.calls % 500 == 0:                       # forget visitors who went quiet
                for k in [k for k, w in self.hits.items() if not w or now - w[-1] >= 60]:
                    del self.hits[k]
            window = self.hits[key]
            while window and now - window[0] >= 60:
                window.popleft()
            if len(window) >= self.per_minute:
                return False
            window.append(now)
            return True


class DailyBudget:
    """A cap on language-model calls per (UTC) day, shared by all visitors."""

    def __init__(self, limit: int, today=lambda: datetime.now(UTC).date()):
        self.limit, self.today = limit, today
        self.day, self.used = today(), 0
        self.lock = threading.Lock()

    def _roll(self) -> None:
        if self.today() != self.day:
            self.day, self.used = self.today(), 0

    def exhausted(self) -> bool:
        with self.lock:
            self._roll()
            return self.used >= self.limit

    def record(self) -> None:
        with self.lock:
            self._roll()
            self.used += 1
