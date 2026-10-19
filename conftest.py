"""Make the in-repo package importable for pytest without installation."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
