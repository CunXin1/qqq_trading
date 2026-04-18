"""CLI: Check data staleness and auto-refresh from IBKR.
CLI：检查数据过期状态并从 IBKR 自动刷新。

Usage / 用法:
    python -m cli.refresh              # refresh if stale
    python -m cli.refresh --force      # force refresh
    python -m cli.refresh --port 7496  # use live IBKR port
    python -m cli.refresh --check      # only check, don't refresh
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description="Auto-refresh data from IBKR")
    parser.add_argument("--force", action="store_true", help="Force refresh even if fresh")
    parser.add_argument("--port", type=int, default=7497, help="IBKR port (7497=paper, 7496=live)")
    parser.add_argument("--check", action="store_true", help="Only check staleness, don't fetch")
    args = parser.parse_args()

    from data.refresh import check_staleness, refresh_sync

    if args.check:
        status = check_staleness()
        print(f"Last date:   {status['last_date']}")
        print(f"Target date: {status['target_date']}")
        print(f"Gap:         {status['gap_days']} days")
        print(f"Stale:       {status['stale']}")
        return

    result = refresh_sync(force=args.force, ibkr_port=args.port)
    if result["refreshed"]:
        print("\nData refreshed successfully.")
    else:
        print("\nNo refresh needed (or refresh failed).")


if __name__ == "__main__":
    main()
