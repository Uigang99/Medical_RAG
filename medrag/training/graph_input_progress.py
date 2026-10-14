"""One owner for a time-weighted, pinned two-line training workflow footer."""
from __future__ import annotations

import datetime
import shutil
import sys
import threading
import time


def duration(value):
    if value is None:
        return "calibrating"
    value = max(0, int(value))
    return f"{value // 3600:02}:{value // 60 % 60:02}:{value % 60:02}"


class Progress:
    def __init__(self, names):
        self.names = names
        self.weights = [None] * len(names)
        self.started = self.stage_started = time.monotonic()
        self.index, self.done, self.total, self.initial = 0, 0, 1, 0
        self.unit, self.kind = "item", "setup"
        self.last = 0
        self.lock = threading.RLock()
        self.tty = sys.stderr.isatty()
        self.stop = threading.Event()
        self.rows = shutil.get_terminal_size((100, 24)).lines
        if self.tty:
            sys.stderr.write(f"\033[1;{self.rows - 2}r\033[{self.rows - 2};1H\n")
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()

    def _heartbeat(self):
        while not self.stop.wait(3 if self.tty else 45):
            self.render(force=True)

    def stage(self, index, total, *, initial=0, unit="item", kind="setup"):
        with self.lock:
            self.index, self.total, self.initial = index, max(1, total), initial
            self.done, self.unit, self.kind = initial, unit, kind
            self.stage_started = time.monotonic()
            self.render(force=True)

    def update(self, amount=1):
        with self.lock:
            self.done = min(self.total, self.done + amount)
            fresh = self.done - self.initial
            if fresh >= (16 if self.kind in ("train", "validation") else 1):
                elapsed = time.monotonic() - self.stage_started
                weight = elapsed / fresh * self.total
                self.weights[self.index] = weight
                prefix = self.names[self.index].split(" epoch")[0]
                if self.kind in ("train", "validation"):
                    for j in range(self.index + 1, len(self.names)):
                        if self.names[j].split(" epoch")[0] == prefix:
                            self.weights[j] = weight
            self.render()

    def lines(self):
        elapsed = time.monotonic() - self.stage_started
        fresh = self.done - self.initial
        rate = fresh / elapsed if fresh and elapsed > 0 else None
        eta = ((self.total - self.done) / rate if rate else
               0 if self.done == self.total else None)
        if all(w is not None for w in self.weights):
            weights = [max(.001, w) for w in self.weights]
            part = self.done / self.total
            fraction = (sum(weights[:self.index]) + weights[self.index] * part) / sum(weights)
            overall_eta = None if eta is None else eta + sum(weights[self.index + 1:])
        else:
            fraction, overall_eta = None, None

        def bar(p):
            if p is None:
                return "unknown|----------|"
            n = min(10, int(p * 10))
            return f"{p * 100:.0f}%|{'█' * n}{'-' * (10-n)}|"

        first = (f"Overall: {bar(fraction)} [stage {self.index + 1}/{len(self.names)}, "
                 f"elapsed {duration(time.monotonic()-self.started)}, ETA {duration(overall_eta)}]")
        speed = "calibrating" if rate is None else f"{rate:.2f} {self.unit}/s"
        second = (f"Stage {self.index + 1}/{len(self.names)} - {self.names[self.index]}: "
                  f"{bar(self.done/self.total)} {self.done}/{self.total} [{speed}, ETA {duration(eta)}]")
        return first, second

    def render(self, force=False):
        with self.lock:
            now = time.monotonic()
            if not force and now - self.last < (1 if self.tty else 45):
                return
            self.last = now
            a, b = self.lines()
            if self.tty:
                cols, rows = shutil.get_terminal_size((120, 24))
                if rows != self.rows:
                    self.rows = rows
                    sys.stderr.write(f"\033[1;{rows-2}r")
                # Compact descriptions, never wrap a footer row.
                if len(b) > cols:
                    b = b.replace("Llama ", "").replace("mixed ", "").replace("question", "q")
                sys.stderr.write(f"\0337\033[{rows-1};1H\033[2K{a[:cols]}"
                                 f"\033[{rows};1H\033[2K{b[:cols]}\0338")
                sys.stderr.flush()
            else:
                stamp = datetime.datetime.now().isoformat(timespec="seconds")
                print(f"[{stamp}] {a}\n[{stamp}] {b}", flush=True)

    def log(self, message):
        with self.lock:
            if self.tty:
                sys.stderr.write(f"\0337\033[{self.rows-1};1H\033[2K"
                                 f"\033[{self.rows};1H\033[2K\0338")
                print(message, file=sys.stderr, flush=True)
                self.render(force=True)
            else:
                print(message, flush=True)

    def complete(self, message):
        with self.lock:
            self.done = self.total
            elapsed = time.monotonic() - self.stage_started
            # Cached stages do not create fake processing rates.
            if self.initial < self.total:
                self.weights[self.index] = max(.001, elapsed * self.total / max(1, self.total-self.initial))
            elif self.weights[self.index] is None:
                self.weights[self.index] = max(.001, elapsed)
            self.render(force=True)
            self.log(f"[stage complete] {self.names[self.index]} duration={duration(elapsed)}; {message}")

    def close(self):
        self.stop.set()
        self.thread.join(timeout=1)
        if self.tty:
            sys.stderr.write(f"\033[r\033[{self.rows};1H\n")
            sys.stderr.flush()
