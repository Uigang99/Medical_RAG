# Medical_RAG Repository Instructions

## Scope and safety

- Treat `/home/user/Uiheon/Medical_RAG` as the repository root.
- Use GPU 1 for GPU workloads by default. Use another GPU only when the user
  explicitly authorizes it for the current work.
- Preserve datasets, vector databases, model weights, checkpoints, caches,
  traces, and experiment results as local artifacts. Never add them to Git.
- Never delete historical scripts merely because they are excluded from Git.
  They may still be needed to audit past experiments.

## Tracked code

- The active script allow-list is defined at the bottom of `.gitignore`.
- When a newly created or revived script becomes part of the active research
  pipeline, add its exact path to that allow-list before committing it.
- Do not add obsolete one-off scripts solely to make the worktree look clean.
- Keep reusable logic in `medrag/`; keep executable experiment entry points in
  `scripts/`; keep focused regression coverage in `tests/`.

## Validation and commits

- Before committing Python changes, run `py_compile` on the affected files and
  run the smallest relevant `unittest` suite. Validate changed shell scripts
  with `bash -n`.
- Inspect the staged diff and verify that it contains no secrets, credentials,
  generated outputs, absolute data dumps, or model artifacts.
- Automatically commit a completed and verified material code change according
  to the global Git working agreements.
- Use a `codex/<topic>` branch for major methodology changes, data-contract
  migrations, or evaluator rewrites. Small bug fixes may remain on `main`.
- Do not push automatically. Push only when the user explicitly asks.
- Show runnable commands with absolute paths and preserve resumability for
  long-running generation, retrieval, attribution, and training jobs.
