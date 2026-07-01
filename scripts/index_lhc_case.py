"""Index one LHC case on this machine by title/case-number fragment."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.core import (
    ensure_case_loaded_from_manifest,
    get_dashboard_stats,
    search_manifest_by_metadata,
)


def main():
    parser = argparse.ArgumentParser(description="Index one LHC case for AI chat")
    parser.add_argument("query", help='Case fragment e.g. "7652-26" or "KHIZAR HAYYAT"')
    args = parser.parse_args()

    hits = search_manifest_by_metadata(args.query, limit=3)
    if not hits:
        print(f"No manifest match for: {args.query}")
        sys.exit(1)

    item = hits[0]
    print(f"Match: {item.get('case_title')}")
    loaded, err = ensure_case_loaded_from_manifest(item)
    if loaded:
        stats = get_dashboard_stats()
        print(f"OK — indexed as {loaded.get('case_id')}: {loaded.get('title')}")
        print(f"Store now: {stats['indexed_cases']} cases, {stats['chunks']} chunks")
        return

    print(f"FAILED: {err or 'unknown error'}")
    sys.exit(1)


if __name__ == "__main__":
    main()
