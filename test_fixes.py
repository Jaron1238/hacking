#!/usr/bin/env python3

import subprocess
import sys

def run_test_subset():
    """Run a subset of tests to check if major fixes work."""
    
    test_files = [
        "tests/unit/test_utils_failed.py",
        "pytest/test_utils.py::TestOUIFunctions::test_lookup_vendor_apple_mac",
        "pytest/test_capture.py::TestPacketParsing::test_packet_to_event_data_frame",
    ]
    
    for test in test_files:
        print(f"\n=== Running {test} ===")
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", test, "-v"
            ], capture_output=True, text=True, cwd="/Users/Jaron/hacking")
            
            print(f"Exit code: {result.returncode}")
            if result.stdout:
                print("STDOUT:", result.stdout[-500:])  # Last 500 chars
            if result.stderr:
                print("STDERR:", result.stderr[-500:])  # Last 500 chars
                
        except Exception as e:
            print(f"Error running test: {e}")

if __name__ == "__main__":
    run_test_subset()