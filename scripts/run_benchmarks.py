"""Benchmark runner for no-magic algorithm scripts.

Discovers and runs algorithm scripts across all tiers, measuring wall-clock
execution time. Supports filtering by section or script name, with table
or JSON output.

Usage:
    python scripts/run_benchmarks.py                          # full suite
    python scripts/run_benchmarks.py --section 01-foundations  # one tier
    python scripts/run_benchmarks.py microgpt.py micrornn.py   # specific scripts
    python scripts/run_benchmarks.py --json                    # JSON output
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SECTIONS = ["01-foundations", "02-alignment", "03-systems"]
STDERR_TAIL_LINES = 20


def discover_scripts() -> dict[str, list[Path]]:
    """Walk tier directories and collect algorithm .py files, sorted alphabetically."""
    result: dict[str, list[Path]] = {}
    for section in SECTIONS:
        section_dir = REPO_ROOT / section
        if not section_dir.is_dir():
            continue
        scripts = sorted(
            p for p in section_dir.glob("*.py")
            if p.name != "__init__.py"
        )
        if scripts:
            result[section] = scripts
    return result


def format_duration(seconds: float) -> str:
    """Convert seconds to 'Xm Ys' display string."""
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{minutes}m {secs:02d}s"


def run_script(script_path: Path) -> dict:
    """Execute a single script and return timing + status data."""
    start = time.monotonic()
    proc = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    elapsed = time.monotonic() - start

    return {
        "name": script_path.name,
        "path": str(script_path.relative_to(REPO_ROOT)),
        "status": "pass" if proc.returncode == 0 else "fail",
        "exit_code": proc.returncode,
        "wall_time_seconds": round(elapsed, 1),
        "wall_time_display": format_duration(elapsed),
        "stderr_tail": proc.stderr.decode(errors="replace").splitlines()[-STDERR_TAIL_LINES:]
        if proc.returncode != 0 else [],
    }


def filter_by_section(
    all_scripts: dict[str, list[Path]], section: str
) -> dict[str, list[Path]]:
    """Return scripts for a single section, or error if section is invalid."""
    if section not in all_scripts:
        valid = ", ".join(all_scripts.keys())
        print(f"Error: unknown section '{section}'. Valid sections: {valid}", file=sys.stderr)
        sys.exit(1)
    return {section: all_scripts[section]}


def filter_by_names(
    all_scripts: dict[str, list[Path]], names: list[str]
) -> dict[str, list[Path]]:
    """Return scripts matching the given filenames, preserving section grouping."""
    # Build a lookup: filename -> (section, path)
    lookup: dict[str, tuple[str, Path]] = {}
    for section, paths in all_scripts.items():
        for p in paths:
            lookup[p.name] = (section, p)

    unrecognized = [n for n in names if n not in lookup]
    if unrecognized:
        print(f"Error: unrecognized script(s): {', '.join(unrecognized)}", file=sys.stderr)
        print(f"Available: {', '.join(sorted(lookup.keys()))}", file=sys.stderr)
        sys.exit(1)

    result: dict[str, list[Path]] = {}
    for name in names:
        section, path = lookup[name]
        result.setdefault(section, []).append(path)
    return result


def print_table(sections_results: dict[str, list[dict]], total_seconds: float) -> None:
    """Print human-readable table output."""
    py_version = platform.python_version()
    os_info = f"{platform.system()} {platform.release()}"
    machine = platform.machine()

    print(f"\nno-magic benchmark results")
    print(f"Python {py_version} | {os_info} | {machine}\n")

    total_scripts = 0
    total_passed = 0

    for section, results in sections_results.items():
        print(f"Section: {section} ({len(results)} scripts)")
        print("\u2500" * 50)
        for r in results:
            total_scripts += 1
            if r["status"] == "pass":
                total_passed += 1
            dots = "." * (35 - len(r["name"]))
            status = "Pass" if r["status"] == "pass" else "FAIL"
            print(f"  {r['name']} {dots} {status}  {r['wall_time_display']}")
            if r["status"] == "fail" and r["stderr_tail"]:
                print(f"    stderr (last {STDERR_TAIL_LINES} lines):")
                for line in r["stderr_tail"]:
                    print(f"      {line}")
        print()

    failed = total_scripts - total_passed
    summary = f"Summary: {total_passed}/{total_scripts} passed"
    if failed:
        summary += f" | {failed} failed"
    summary += f" | Total: {format_duration(total_seconds)}"
    print(summary)


def build_json(sections_results: dict[str, list[dict]], total_seconds: float) -> dict:
    """Build JSON-serializable results dict."""
    total = sum(len(v) for v in sections_results.values())
    passed = sum(1 for results in sections_results.values() for r in results if r["status"] == "pass")

    sections_out = {}
    for section, results in sections_results.items():
        sections_out[section] = {
            "scripts": [
                {
                    "name": r["name"],
                    "status": r["status"],
                    "exit_code": r["exit_code"],
                    "wall_time_seconds": r["wall_time_seconds"],
                    "wall_time_display": r["wall_time_display"],
                }
                for r in results
            ]
        }

    return {
        "python_version": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sections": sections_out,
        "summary": {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "total_seconds": round(total_seconds, 1),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run no-magic algorithm scripts and report timing results."
    )
    parser.add_argument(
        "--section",
        choices=SECTIONS,
        help="Run scripts from a specific section only.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output results as JSON instead of a table.",
    )
    parser.add_argument(
        "scripts",
        nargs="*",
        help="Specific script filenames to run (e.g. microgpt.py micrornn.py).",
    )

    args = parser.parse_args()

    if args.section and args.scripts:
        parser.error("--section and positional script names are mutually exclusive.")

    all_scripts = discover_scripts()
    if not all_scripts:
        print("Error: no algorithm scripts found.", file=sys.stderr)
        sys.exit(1)

    if args.section:
        targets = filter_by_section(all_scripts, args.section)
    elif args.scripts:
        targets = filter_by_names(all_scripts, args.scripts)
    else:
        targets = all_scripts

    # Run scripts and collect results
    sections_results: dict[str, list[dict]] = {}
    any_failed = False
    total_start = time.monotonic()

    for section in SECTIONS:
        if section not in targets:
            continue
        results = []
        for script_path in targets[section]:
            print(f"Running {script_path.relative_to(REPO_ROOT)} ...", file=sys.stderr, flush=True)
            result = run_script(script_path)
            if result["status"] == "fail":
                any_failed = True
            results.append(result)
        sections_results[section] = results

    total_seconds = time.monotonic() - total_start

    # Output
    if args.json_output:
        print(json.dumps(build_json(sections_results, total_seconds), indent=2))
    else:
        print_table(sections_results, total_seconds)

    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
