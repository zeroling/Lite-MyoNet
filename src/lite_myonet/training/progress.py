"""Report batch progress without letting child processes share terminal cursors."""

from __future__ import annotations

import json
import os
import sys
import time

from tqdm import tqdm


PROGRESS_PREFIX = "__LITE_MYONET_PROGRESS_V1__ "


class JsonProgress:
    """Emit throttled progress events for rendering in the matrix controller."""

    def __init__(self, iterable, description, stream=None, mininterval=0.2):
        self.iterable = iterable
        self.description = str(description)
        self.stream = stream if stream is not None else sys.stdout
        self.mininterval = float(mininterval)
        self.total = len(iterable) if hasattr(iterable, "__len__") else None
        self.n = 0
        self.postfix = {}
        self.last_report = 0.0
        self.closed = False

    def _emit(self, event, force=False):
        """Write one complete event line at most once per reporting interval."""
        now = time.monotonic()
        if not force and now - self.last_report < self.mininterval:
            return
        payload = {
            "event": event, "n": self.n, "total": self.total,
            "description": self.description, "postfix": self.postfix,
        }
        self.stream.write(PROGRESS_PREFIX + json.dumps(payload, ensure_ascii=False) + "\n")
        self.stream.flush()
        self.last_report = now

    def __iter__(self):
        """Yield batches and report completed batches, including final cleanup."""
        self._emit("start", force=True)
        try:
            for item in self.iterable:
                yield item
                self.n += 1
                self._emit("update")
        finally:
            self.close()

    def set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
        """Save loss and other scalar text for the next throttled update."""
        del refresh
        values = dict(ordered_dict or {})
        values.update(kwargs)
        self.postfix = {str(key): str(value) for key, value in values.items()}

    def close(self):
        """Emit the final count exactly once."""
        if not self.closed:
            self.closed = True
            self._emit("close", force=True)


def create_progress_bar(loader, description):
    """Use parent-rendered events in a matrix, or a red bar for standalone use."""
    if os.environ.get("LITE_MYONET_PROGRESS_TRANSPORT") == "json":
        return JsonProgress(loader, description)
    try:
        position = max(0, int(os.environ.get("LITE_MYONET_PROGRESS_POSITION", "0")))
    except ValueError:
        position = 0
    return tqdm(
        loader, desc=description, leave=False, dynamic_ncols=True,
        colour="red", position=position, mininterval=0.2,
    )
