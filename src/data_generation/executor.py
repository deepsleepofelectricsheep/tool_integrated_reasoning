"""Execute Python code in an isolated subprocess with resource guardrails.

Protections applied to the child process (Linux/macOS):
  - Wall-clock timeout via subprocess (already catches infinite loops)
  - Virtual memory cap  — prevents memory bombs
  - Max child processes — prevents fork bombs
  - Isolated temp working directory — writes stay inside a directory that is
    deleted on exit; the rest of the filesystem is still readable but the
    child cannot easily pollute the project tree

Not covered (out of scope for local academic use):
  - Network isolation  (would require Linux namespaces / Docker)
  - Import whitelist   (fragile; would break legitimate sympy/numpy usage)
"""

import os
import subprocess
import sys
import tempfile

# Resource limits applied inside the child process before exec.
# Both values are (soft, hard) in bytes / count.
_MAX_MEMORY_MB  = 512          # virtual address space cap
_MAX_NPROC      = 64           # max spawnable sub-processes (fork bomb guard)


def _child_limits() -> None:
    """Preexec hook: apply resource limits inside the forked child."""
    try:
        import resource  # Unix only; silently skipped on Windows

        mem = _MAX_MEMORY_MB * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS,    (mem, mem))
        resource.setrlimit(resource.RLIMIT_NPROC, (_MAX_NPROC, _MAX_NPROC))
    except Exception:
        pass  # best-effort; don't crash data generation if unavailable


def execute_code(code: str, timeout: int = 10) -> str:
    """Run code in a sandboxed subprocess and return stdout or an error string."""
    with tempfile.TemporaryDirectory() as workdir:
        code_file = os.path.join(workdir, "solution.py")
        with open(code_file, "w") as f:
            f.write(code)

        try:
            proc = subprocess.run(
                [sys.executable, code_file],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=workdir,          # working dir = isolated temp directory
                preexec_fn=_child_limits,
            )
        except subprocess.TimeoutExpired:
            return "Error: Execution timed out."
        except Exception as e:
            return f"Error: {e}"

    if proc.returncode != 0 and proc.stderr.strip():
        last_line = proc.stderr.strip().splitlines()[-1]
        return f"Error: {last_line}"
    return proc.stdout.strip() or "(no output)"
