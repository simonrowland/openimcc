"""Run the public install/use quickstart from outside the repository."""

from __future__ import annotations

import importlib
import os
import re
import shlex
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
FENCE_RE = re.compile(r"^```(?P<language>[^\n]*)\n(?P<body>.*?)^```", re.MULTILINE | re.DOTALL)
PYTHON_LANGUAGES = {"py", "python", "python3"}
SHELL_LANGUAGES = {"bash", "sh", "shell", "zsh"}
OUTPUT_LANGUAGES = {"console", "text"}


@dataclass(frozen=True)
class CodeFence:
    language: str
    body: str
    start: int
    end: int


def _quickstart_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    return text


def _fences(path: Path) -> list[CodeFence]:
    text = _quickstart_text(path)
    return [
        CodeFence(
            language=match.group("language").strip().lower(),
            body=match.group("body"),
            start=match.start(),
            end=match.end(),
        )
        for match in FENCE_RE.finditer(text)
    ]


def _document_paths() -> list[Path]:
    paths = [README]
    extra = REPO_ROOT / "docs" / "quickstart.md"
    if extra.is_file():
        paths.append(extra)
    return paths


def _is_pip_install(command: str) -> bool:
    tokens = shlex.split(command, comments=True)
    return (
        len(tokens) >= 4
        and tokens[0] in {"python", "python3"}
        and tokens[1:4] == ["-m", "pip", "install"]
    )


def _check_install_fence(body: str) -> int:
    commands = _shell_commands(body)
    assert commands[:2] == [
        "git clone https://github.com/simonrowland/openimcc",
        "cd openimcc",
    ]

    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    optional_dependencies = project["project"]["optional-dependencies"]
    expected_extras = ["gas", "bench"]
    assert all(extra in optional_dependencies for extra in expected_extras)

    install_commands = [command for command in commands if _is_pip_install(command)]
    assert len(install_commands) == len(expected_extras)
    observed_extras: list[str] = []
    for command in install_commands:
        tokens = shlex.split(command, comments=True)
        assert len(tokens) == 6
        assert tokens[:5] == ["python", "-m", "pip", "install", "-e"]
        target = tokens[5]
        assert target.startswith(".[") and target.endswith("]")
        observed_extras.append(target[2:-1])
    assert observed_extras == expected_extras
    return len(commands)


def _shell_commands(body: str) -> list[str]:
    """Join ordinary shell continuation lines and return one command per line."""
    commands: list[str] = []
    pending = ""
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        pending = f"{pending} {line}".strip()
        if pending.endswith("\\"):
            pending = pending[:-1].rstrip()
            continue
        commands.append(pending)
        pending = ""
    if pending:
        commands.append(pending)
    return commands


def _console_command(name: str, args: list[str]) -> tuple[list[str], str]:
    executable = shutil.which(name)
    if executable:
        return [executable, *args], f"{name} console script"
    # No console script on PATH (source checkout, no install): check the
    # mapping a real install would generate, then run that module with -m.
    expected = {"openimcc": "openimcc.cli:main", "openimcc-bench": "openimcc.bench:main"}
    scripts = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]["scripts"]
    mapping = scripts.get(name)
    assert mapping == expected[name], (
        f"pyproject [project.scripts] {name} must map to {expected[name]}"
    )
    module_name, _, attr = mapping.partition(":")
    module = importlib.import_module(module_name)
    assert callable(getattr(module, attr, None)), f"{mapping} is not importable"
    return [sys.executable, "-m", module_name, *args], (
        f"python -m {module_name} fallback"
    )


def _run_command(
    command: str, *, cwd: Path, env: dict[str, str]
) -> tuple[subprocess.CompletedProcess[str], str]:
    tokens = shlex.split(command, comments=True)
    if not tokens:
        raise AssertionError("empty documented shell command")

    label = "shell command"
    if tokens[0] in {"openimcc", "openimcc-bench"}:
        argv, label = _console_command(tokens[0], tokens[1:])
    elif tokens[0] in {"python", "python3"} and len(tokens) >= 3 and tokens[1:3] == ["-m", "pip"]:
        argv = [sys.executable, *tokens[1:]]
        label = "python -m pip"
    else:
        argv = tokens

    completed = subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return completed, label


def test_documented_python_and_shell_commands_run_from_outside_repo(
    tmp_path: Path,
) -> None:
    """The README is an executable user path, including checkout-only commands."""
    assert not tmp_path.is_relative_to(REPO_ROOT)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    env.pop("OPENIMCC_VAPOROCK_ROOT", None)
    env["PYTHONWARNINGS"] = "error"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"

    python_blocks = 0
    shell_commands: list[str] = []
    text_checked_commands: list[str] = []
    bench_commands: list[str] = []
    executed_cli_modes: list[str] = []
    bench_n: int | None = None

    for path in _document_paths():
        fences = _fences(path)
        source = _quickstart_text(path)
        if path == README:
            assert "Python >= 3.11" in source
            assert not re.search(r"python(?:3)?\s+-m\s+openimcc(?:\s|$)", source)
        for index, fence in enumerate(fences):
            if fence.language in PYTHON_LANGUAGES:
                python_blocks += 1
                completed = subprocess.run(
                    [sys.executable, "-c", fence.body],
                    cwd=tmp_path,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                assert completed.returncode == 0, (
                    f"{path}:{fence.language} failed\n"
                    f"stdout:\n{completed.stdout}\n"
                    f"stderr:\n{completed.stderr}"
                )

                if index + 1 < len(fences) and fences[index + 1].language in OUTPUT_LANGUAGES:
                    between = source[fence.end : fences[index + 1].start]
                    if re.search(r"^Output:\s*$", between, re.MULTILINE):
                        expected = fences[index + 1].body.strip()
                        for line in expected.splitlines():
                            assert line in completed.stdout, (
                                f"{path}: documented output line missing: {line!r}\n"
                                f"actual stdout:\n{completed.stdout}"
                            )

            elif fence.language in SHELL_LANGUAGES:
                commands = _shell_commands(fence.body)
                if any(_is_pip_install(command) for command in commands):
                    # Installation commands are text-checked, not executed: this
                    # test is offline and pip would mutate the test environment.
                    checked_count = _check_install_fence(fence.body)
                    text_checked_commands.extend(commands)
                    assert checked_count == len(commands)
                    continue
                for command in commands:
                    documented = f"{path.name}: {command}"
                    shell_commands.append(documented)
                    command_cwd = REPO_ROOT if command.startswith("openimcc-bench ") else tmp_path
                    completed, mode = _run_command(command, cwd=command_cwd, env=env)
                    assert completed.returncode == 0, (
                        f"{path.name}: {command!r} failed via {mode}\n"
                        f"stdout:\n{completed.stdout}\n"
                        f"stderr:\n{completed.stderr}"
                    )
                    if command.startswith(("openimcc ", "openimcc-bench ")):
                        executed_cli_modes.append(f"{command.split()[0]}: {mode}")
                    if command.startswith("openimcc solve "):
                        assert "basis type = wt" in completed.stdout
                        assert "mole total (from wt%) =" in completed.stdout
                        assert "notice: Na and K activities from IMCC-SF04" in completed.stdout
                    if command.startswith("openimcc-bench "):
                        bench_commands.append(command)
                        if command == (
                            "openimcc-bench benchmarks/sets/basalt-bench-set-v1.yaml "
                            "src/openimcc/data/packs/imcc-sf04-v1.0.2.json"
                        ):
                            match = re.search(r"\bN=(\d+)\b", completed.stdout)
                            assert match, completed.stdout
                            bench_n = int(match.group(1))
                            from openimcc.bench import load_bench_set

                            bench_set = load_bench_set(
                                REPO_ROOT / "benchmarks/sets/basalt-bench-set-v1.yaml"
                            )
                            assert bench_n == len(bench_set["points"])

    assert python_blocks > 0
    assert shell_commands
    assert len(text_checked_commands) == 4
    assert len(bench_commands) == 4
    assert bench_n == 402

    print(f"documented python blocks: {python_blocks}")
    print(f"documented shell commands executed: {len(shell_commands)}")
    print(f"documented install commands text-checked: {len(text_checked_commands)}")
    print(f"documented bench commands executed from checkout: {len(bench_commands)}")
    print(f"documented bench N: {bench_n}")
    for mode in executed_cli_modes:
        print(mode)
