"""CLI: sync LHC approved judgments into JAMS dataset."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import core
from backend.scraper import lhc as lhc_scraper


def main():
    parser = argparse.ArgumentParser(description="Sync LHC judgments into JAMS")
    parser.add_argument("--metadata-only", action="store_true", help="Fetch list only (~4683 rows)")
    parser.add_argument("--year", type=str, default="", help="Year filter (empty = all)")
    parser.add_argument("--court", type=str, default="All Courts", help="Judge/court filter")
    parser.add_argument("--limit", type=int, default=50, help="Max PDFs to download per run")
    args = parser.parse_args()

    print("Starting LHC sync...")
    result = lhc_scraper.sync_lhc_judgments(
        year=args.year,
        court_name=args.court,
        metadata_only=args.metadata_only,
        auto_index=not args.metadata_only,
        download_limit=None if args.metadata_only else args.limit,
        index_callback=core.index_lhc_judgment,
    )
    print(result)


if __name__ == "__main__":
    main()
