from __future__ import annotations

import ast
import compileall
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TESTS = ROOT / "tests"


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def run_command(args: list[str]) -> int:
    print(f"[check] {' '.join(args)}", flush=True)
    completed = subprocess.run(args, cwd=ROOT)
    return int(completed.returncode)


def check_compile() -> int:
    print("[check] compileall src/game/world", flush=True)
    return 0 if compileall.compile_dir(SRC / "game" / "world", quiet=1) else 1


def check_engine_dependency_direction() -> int:
    print("[check] engine package does not import game package", flush=True)
    failures: list[tuple[Path, int, str]] = []
    for path in sorted((SRC / "engine").rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "game" or alias.name.startswith("game."):
                        failures.append((path, node.lineno, alias.name))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == "game" or module.startswith("game."):
                    failures.append((path, node.lineno, module))

    for path, line, module in failures:
        print(f"[check] {_relative(path)}:{line}: engine imports {module}")
    return 1 if failures else 0


def check_wildcard_imports() -> int:
    print("[check] no wildcard imports under src", flush=True)
    failures: list[tuple[Path, int, str]] = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if any(alias.name == "*" for alias in node.names):
                    failures.append((path, node.lineno, node.module or ""))

    for path, line, module in failures:
        print(f"[check] {_relative(path)}:{line}: wildcard import from {module}")
    return 1 if failures else 0


def main() -> int:
    checks = [
        check_compile,
        lambda: run_command(
            [sys.executable, "-m", "unittest", "discover", "-s", str(TESTS)]
        ),
        check_engine_dependency_direction,
        check_wildcard_imports,
    ]
    failures = 0
    for check in checks:
        failures += 1 if check() else 0

    if failures:
        print(f"[check] failed: {failures} check(s)", flush=True)
        return 1
    print("[check] all checks passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
