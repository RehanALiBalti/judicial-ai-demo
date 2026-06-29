"""CLI: sync LHC approved judgments into JAMS dataset."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.scraper import lhc as lhc_scraper


def main():
    parser = argparse.ArgumentParser(description="Sync LHC judgments into JAMS")
    parser.add_argument("--metadata-only", action="store_true", help="Fetch list only (~4683 rows)")
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download PDFs only — no indexing (much faster; index later on server)",
    )
    parser.add_argument("--year", type=str, default="", help="Year filter (empty = all)")
    parser.add_argument("--court", type=str, default="All Courts", help="Judge/court filter")
    parser.add_argument("--limit", type=int, default=50, help="Max PDFs to download per run")
    args = parser.parse_args()

    auto_index = not args.metadata_only and not args.download_only
    index_callback = None
    if auto_index:
        from backend import core

        index_callback = core.index_lhc_judgment
        print("Mode: download + index (slow — embeddings per PDF)")
    elif args.download_only:
        print("Mode: download only (fast — no indexing)")
    else:
        print("Mode: metadata list only")

    print("Starting LHC sync...")
    result = lhc_scraper.sync_lhc_judgments(
        year=args.year,
        court_name=args.court,
        metadata_only=args.metadata_only,
        auto_index=auto_index,
        download_limit=None if args.metadata_only else args.limit,
        refresh_metadata=not args.download_only,
        index_callback=index_callback,
        progress_callback=print,
    )
    print(result)


if __name__ == "__main__":
    main()
