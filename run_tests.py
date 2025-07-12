#!/usr/bin/env python3
"""
Test runner script for GMI package.
"""
import sys
import subprocess
import argparse
from pathlib import Path


def run_pytest(test_path=None, markers=None, verbose=False, coverage=True):
    """Run pytest with specified options."""
    cmd = ["python", "-m", "pytest"]
    
    if test_path:
        cmd.append(test_path)
    
    if markers:
        cmd.extend(["-m", markers])
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend([
            "--cov=gmi",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov"
        ])
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run GMI tests")
    parser.add_argument(
        "test_path", 
        nargs="?", 
        help="Specific test path (e.g., tests/linear_operator/)"
    )
    parser.add_argument(
        "-m", "--markers", 
        help="pytest markers (e.g., 'unit', 'integration', 'slow')"
    )
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true", 
        help="Verbose output"
    )
    parser.add_argument(
        "--no-coverage", 
        action="store_true", 
        help="Disable coverage reporting"
    )
    parser.add_argument(
        "--list-tests", 
        action="store_true", 
        help="List available test categories"
    )
    
    args = parser.parse_args()
    
    if args.list_tests:
        print("Available test categories:")
        test_dirs = [d for d in Path("tests").iterdir() if d.is_dir() and d.name != "__pycache__"]
        for test_dir in sorted(test_dirs):
            print(f"  {test_dir.name}/")
        return 0
    
    coverage = not args.no_coverage
    return run_pytest(args.test_path, args.markers, args.verbose, coverage)


if __name__ == "__main__":
    sys.exit(main()) 