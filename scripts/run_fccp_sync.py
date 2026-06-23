"""CLI: sync FCCP judgments into JAMS dataset."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import core
from backend.scraper import fccp as fccp_scraper


def main():
    parser = argparse.ArgumentParser(description="Sync FCCP judgments into JAMS")
    parser.add_argument("--start", type=int, default=1, help="Start page")
    parser.add_argument("--end", type=int, default=None, help="End page (default: all)")
    args = parser.parse_args()

    print("Starting FCCP sync...")
    result = fccp_scraper.sync_fccp_judgments(
        start_page=args.start,
        end_page=args.end,
        auto_index=True,
        index_callback=core.index_fccp_judgment,
    )
    print(result)


if __name__ == "__main__":
    main()
