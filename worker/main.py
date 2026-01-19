"""Worker script for processing evaluation bundles."""

import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.config import settings


def main() -> None:
    """Main worker loop."""
    print("worker started")
    print(f"polling every {settings.worker_poll_interval_seconds} seconds...")

    while True:
        # TODO: Poll database for UPLOADING bundles to validate
        print("polling...")
        time.sleep(settings.worker_poll_interval_seconds)


if __name__ == "__main__":
    main()
