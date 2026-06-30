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
        help="Download PDFs only — no indexing (fast)",
    )
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Index already-downloaded PDFs only (no API fetch, no re-download)",
    )
    parser.add_argument("--year", type=str, default="", help="Year filter (empty = all)")
    parser.add_argument("--court", type=str, default="All Courts", help="Judge/court filter")
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Max PDFs to download or index per run (use 0 for no limit)",
    )
    args = parser.parse_args()

    if sum([args.metadata_only, args.download_only, args.index_only]) > 1:
        parser.error("Use only one of --metadata-only, --download-only, --index-only")

    auto_index = args.index_only or (not args.metadata_only and not args.download_only)
    index_callback = None
    if auto_index:
        from backend import core

        index_callback = core.index_lhc_judgment

    if args.index_only:
        print("Mode: index only (embeddings — slow, ~5–15 min/PDF on CPU)")
    elif args.download_only:
        print("Mode: download only (fast — no indexing)")
    elif args.metadata_only:
        print("Mode: metadata list only")
    else:
        print("Mode: download + index")

    limit = None if args.limit == 0 else args.limit
    if args.metadata_only:
        limit = None

    print("Starting LHC sync...")
    result = lhc_scraper.sync_lhc_judgments(
        year=args.year,
        court_name=args.court,
        metadata_only=args.metadata_only,
        auto_index=auto_index,
        download_limit=limit,
        refresh_metadata=not (args.download_only or args.index_only),
        index_callback=index_callback,
        progress_callback=print,
    )
    print(result)


if __name__ == "__main__":
    main()
