from __future__ import annotations

import time

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


class OverallProgress:
    def __init__(self, total: int, desc: str = "RAG Eval", enabled: bool = True) -> None:
        self.total = max(0, int(total))
        self.enabled = bool(enabled)
        self.done = 0
        self.stage = ""
        self.started_at = time.time()
        self._last_render = 0.0
        self._pbar = tqdm(total=self.total, desc=desc, unit="unit", dynamic_ncols=True) if tqdm and enabled else None

    def update(self, n: int = 1, stage: str = "") -> None:
        n = max(0, int(n))
        self.done += n
        if stage:
            self.stage = stage
        if self._pbar is not None:
            if stage:
                elapsed = time.time() - self.started_at
                self._pbar.set_postfix_str(f"stage={stage} elapsed={elapsed:.1f}s", refresh=False)
            self._pbar.update(n)
            return
        if not self.enabled:
            return
        now = time.time()
        if now - self._last_render < 0.5 and self.done < self.total:
            return
        self._last_render = now
        elapsed = now - self.started_at
        pct = (self.done / self.total * 100.0) if self.total else 100.0
        eta = (elapsed / max(1, self.done)) * max(0, self.total - self.done)
        print(
            f"\r[RAG Eval] {pct:6.2f}% {self.done}/{self.total} "
            f"stage={self.stage or stage} elapsed={elapsed:.1f}s eta={eta:.1f}s",
            end="",
            flush=True,
        )

    def close(self) -> None:
        if self._pbar is not None:
            self._pbar.close()
        elif self.enabled:
            print()


class StageProgress:
    def __init__(self, total: int, desc: str, enabled: bool = True) -> None:
        self.total = max(0, int(total))
        self.desc = desc
        self.enabled = bool(enabled)
        self.done = 0
        self.started_at = time.time()
        self._last_render = 0.0
        self._pbar = (
            tqdm(total=self.total, desc=desc, unit="sample", dynamic_ncols=True)
            if tqdm and enabled and self.total > 0
            else None
        )

    def update(self, n: int = 1) -> None:
        n = max(0, int(n))
        self.done += n
        if self._pbar is not None:
            elapsed = time.time() - self.started_at
            self._pbar.set_postfix_str(f"elapsed={elapsed:.1f}s", refresh=False)
            self._pbar.update(n)
            return
        if not self.enabled or self.total <= 0:
            return
        now = time.time()
        if now - self._last_render < 0.5 and self.done < self.total:
            return
        self._last_render = now
        elapsed = now - self.started_at
        pct = (self.done / self.total * 100.0) if self.total else 100.0
        eta = (elapsed / max(1, self.done)) * max(0, self.total - self.done)
        print(
            f"\r[{self.desc}] {pct:6.2f}% {self.done}/{self.total} "
            f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
            end="",
            flush=True,
        )

    def close(self) -> None:
        if self._pbar is not None:
            self._pbar.close()
        elif self.enabled and self.total > 0:
            print()
