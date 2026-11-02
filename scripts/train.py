#!/usr/bin/env python3
"""Train the existing mean3/content-preference objective on a private full pool."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'evicheck3d'))
from public_cli import main

if __name__ == '__main__':
    main('train')
