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
        self.detail = ""
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
            detail = f" {self.detail}" if self.detail else ""
            self._pbar.set_postfix_str(
                f"elapsed={elapsed:.1f}s{detail}", refresh=False
            )
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

    def set_detail(self, detail: str) -> None:
        self.detail = str(detail)
        if self._pbar is not None:
            elapsed = time.time() - self.started_at
            self._pbar.set_postfix_str(
                f"elapsed={elapsed:.1f}s {self.detail}", refresh=False
            )

    def close(self) -> None:
        if self._pbar is not None:
            self._pbar.close()
        elif self.enabled and self.total > 0:
            print()


class PipelineProgress:
    """One stable progress line with both pipeline and active-stage status.

    ``overall_initial`` allows independent, sequential processes to represent
    their position in one logical pipeline (for example generation, artifact
    materialization, and hidden-state extraction) without keeping both models
    resident on the GPU at once.
    """

    def __init__(
        self,
        *,
        overall_total: int,
        overall_initial: int = 0,
        desc: str = "Pipeline",
        enabled: bool = True,
    ) -> None:
        self.overall_total = max(0, int(overall_total))
        self.overall_done = max(0, int(overall_initial))
        self.enabled = bool(enabled)
        self.stage = "initializing"
        self.stage_total = 0
        self.stage_done = 0
        self._stage_initial = 0
        self.stage_started_at = time.time()
        self._pbar = (
            tqdm(
                total=self.overall_total,
                initial=min(self.overall_done, self.overall_total),
                desc=desc,
                unit="sample",
                dynamic_ncols=True,
            )
            if tqdm and enabled
            else None
        )
        self._render()

    def set_stage(self, stage: str, *, total: int, initial: int = 0) -> None:
        self.stage = str(stage)
        self.stage_total = max(0, int(total))
        self.stage_done = min(max(0, int(initial)), self.stage_total)
        self._stage_initial = self.stage_done
        self.stage_started_at = time.time()
        self._render()

    def update(self, n: int = 1) -> None:
        value = max(0, int(n))
        self.stage_done = min(self.stage_total, self.stage_done + value)
        self.overall_done = min(self.overall_total, self.overall_done + value)
        if self._pbar is not None:
            self._pbar.update(value)
        self._render()

    def _render(self) -> None:
        if self._pbar is None:
            return
        elapsed = max(1e-9, time.time() - self.stage_started_at)
        processed = max(0, self.stage_done)
        newly_processed = max(0, processed - self._stage_initial)
        remaining = max(0, self.stage_total - processed)
        stage_eta = (
            elapsed / newly_processed * remaining
            if newly_processed
            else float("inf")
        )
        eta_text = "?" if stage_eta == float("inf") else _format_duration(stage_eta)
        self._pbar.set_postfix_str(
            f"stage={self.stage} {processed}/{self.stage_total} stage_eta={eta_text}",
            refresh=False,
        )

    def close(self) -> None:
        if self._pbar is not None:
            self._render()
            self._pbar.close()


def _format_duration(seconds: float) -> str:
    value = max(0, int(seconds))
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"
