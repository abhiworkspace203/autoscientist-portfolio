"""
Pytest configuration and shared fixtures for the test suite.
"""

import sys
from pathlib import Path

# Ensure repository root is on sys.path for test discovery and execution
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
