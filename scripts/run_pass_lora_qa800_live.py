#!/usr/bin/env python3
"""Persistent PASS QA worker with a reconnectable foreground progress display."""
from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_pass_lora_qa800.sh"
DEFAULT = ROOT / "results/pass_pced_chain_v5_800_v1"


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def write_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False))
    os.replace(tmp, path)


def identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return fields[19]  # starttime, field 22 of /proc/PID/stat
    except (OSError, IndexError):
        return None


def alive(state):
    start = state.get("start_ticks")
    return bool(state.get("state") == "running" and start and identity(state.get("pid", 0)) == start)


def busy(path):
    with path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return False
        except BlockingIOError:
            return True


def supervise(folder, command):
    """Separate session survives viewer/SSH termination; retain the true exit code."""
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    state_path = folder / "live_job.json"
    state = {"pid": os.getpid(), "start_ticks": identity(os.getpid()),
             "state": "running", "started": time.time(), "command": command}
    write_json(state_path, state)
    code = 1
    try:
        with (folder / "live.log").open("ab", buffering=0) as log:
            log.write(f"\n[live start] {datetime.datetime.now().isoformat(timespec='seconds')}\n".encode())
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            state["worker_pid"] = child.pid
            write_json(state_path, state)
            code = child.wait()
    finally:
        state.update(state="completed" if code == 0 else "failed", exit_code=code, ended=time.time())
        write_json(state_path, state)
    return code


def runner_command(folder, extra_args=()):
    return ["bash", str(RUNNER), "--phase", "run", "--output-root", str(folder), *extra_args]


def connect(folder, extra_args=()):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "live_launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = read_json(folder / "live_job.json")
        if state.get("command") and state["command"] != runner_command(folder, extra_args):
            raise RuntimeError("이 폴더의 기존 실행 옵션과 다릅니다. 별도의 --output-root를 지정하세요.")
        contract = read_json(folder / "manifest.json").get("contract", {})
        if extra_args and contract:
            if "--fallback" in extra_args and contract.get("decoder", {}).get("all_pass") != extra_args[extra_args.index("--fallback")+1]:
                raise RuntimeError("기존 결과의 fallback과 다릅니다. 새 출력 폴더를 사용하세요.")
            if "--arms" in extra_args:
                end = extra_args.index("--fallback") if "--fallback" in extra_args else len(extra_args)
                if contract.get("arms") != list(extra_args[extra_args.index("--arms")+1:end]):
                    raise RuntimeError("기존 결과의 비교 조건과 다릅니다. 새 출력 폴더를 사용하세요.")
        if alive(state):
            return "기존 작업에 다시 연결했습니다."
        if busy(folder / ".run.lock"):
            return "이미 실행 중인 평가의 진행 화면에 연결했습니다."
        if state.get("state") == "completed":
            return "이미 완료된 작업입니다. 결과를 표시합니다."
        # A new viewer may read stale status during the CPU preflight; never remove it.
        with (folder / "live_supervisor.log").open("ab", buffering=0) as log:
            proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                "--output-root", str(folder), "--worker", *extra_args], stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(100):
            current = read_json(folder / "live_job.json")
            if current.get("pid") == proc.pid:
                return "평가를 시작했습니다. SSH가 끊겨도 작업은 계속됩니다."
            if proc.poll() is not None:
                raise RuntimeError(f"진행 실행기 시작 실패: {folder / 'live_supervisor.log'}")
            time.sleep(.05)
        raise RuntimeError("작업 시작 확인이 지연됩니다. 같은 명령으로 다시 연결하세요.")


def tail_messages(path):
    try:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - 24000))
            lines = fh.read().decode("utf-8", errors="replace").splitlines()
        return [line for line in lines if line.startswith(("[stage complete]", "[calibration]", "[FAILED]", "[complete]"))][-4:]
    except OSError:
        return []


def display(folder, message):
    tty, last = sys.stdout.isatty(), 0.
    if tty:
        sys.stdout.write("\033[?1049h\033[?25l")
    try:
        while True:
            state, status = read_json(folder / "live_job.json"), read_json(folder / "status.json")
            running = alive(state) or busy(folder / ".run.lock")
            completed = state.get("state") == "completed" and not running
            failed = not running and not completed
            if not state and not running:
                completed = (folder / "summary.json").is_file()
                failed = not completed
            label = "완료" if completed else "중단/오류" if failed else "실행 중"
            lines = [f"PASS-PCED 800 | {label} | {message}",
                     "Ctrl+C: 화면만 닫기 (평가 유지) · 같은 명령: 다시 연결",
                     f"결과: {folder}", f"최근 갱신: {status.get('updated', '초기화 중')}", ""]
            lines.extend(tail_messages(folder / "live.log"))
            bars = [status.get("overall", "Overall: unknown|----------| [ETA calibrating]"),
                    status.get("current", "Stage 1/10 - CPU contract: 시작 준비 중")]
            now = time.monotonic()
            if tty:
                width, height = os.get_terminal_size()
                # Keep the two existing workflow bars at the bottom; no competing progress estimate.
                def fit(line):
                    # Korean text takes two columns; never let terminal lines wrap.
                    import unicodedata
                    used, out = 0, []
                    for c in line:
                        used += 2 if unicodedata.east_asian_width(c) in ("W", "F") else 1
                        if used >= width:
                            break
                        out.append(c)
                    return "".join(out)
                sys.stdout.write("\033[H\033[2J" + "\n".join(fit(x) for x in lines[:max(1, height-3)]))
                sys.stdout.write(f"\033[{max(1,height-1)};1H{fit(bars[0])}\033[{height};1H{fit(bars[1])}")
                sys.stdout.flush()
            elif now - last >= 30 or not running:
                print("\n".join(lines + bars), flush=True)
                last = now
            if not running:
                break
            time.sleep(1)
    except KeyboardInterrupt:
        label = "화면 닫음 — 실행 중인 평가는 계속됩니다"
        return_code = 0
    else:
        return_code = 0 if completed else 1
    finally:
        if tty:
            sys.stdout.write("\033[?25h\033[?1049l")
            sys.stdout.flush()
    print(f"[{label}] {folder}")
    if label == "완료":
        print(f"결과 요약: {folder / 'summary.json'}")
    elif label == "중단/오류":
        print(f"로그: {folder / 'live.log'}\n같은 명령으로 완료된 기록부터 재개할 수 있습니다.")
    return return_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    arms = ("no_rag", "all", "pced", "pced_lora", "pass_pced", "pass_pced_safe")
    parser.add_argument("--arms", nargs="+", choices=arms)
    parser.add_argument("--fallback", choices=("pced", "amateur"))
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (args.arms is not None or args.fallback is not None) and args.output_root is None:
        parser.error("비교 조건을 변경할 때에는 별도의 --output-root를 지정하세요.")
    folder = (args.output_root or DEFAULT).resolve()
    extra_args = []
    if args.arms is not None:
        extra_args += ["--arms", *[arm for arm in arms if arm in args.arms]]
    if args.fallback is not None:
        extra_args += ["--fallback", args.fallback]
    if not RUNNER.is_file():
        parser.error(f"실행 파일 없음: {RUNNER}")
    if args.worker:
        return supervise(folder, runner_command(folder, extra_args))
    return display(folder, connect(folder, extra_args))


if __name__ == "__main__":
    raise SystemExit(main())
