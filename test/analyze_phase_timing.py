import argparse
import re
import statistics


def parse_rows(path):
    rows = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("FULL_FITTING_PHASE"):
                continue
            pairs = dict(re.findall(r"(\w+)=([0-9.]+)", line))
            if pairs:
                rows.append({key: float(value) for key, value in pairs.items()})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+")
    args = parser.parse_args()
    for path in args.logs:
        rows = parse_rows(path)
        if not rows:
            print(f"{path}: NO_PHASE_ROWS")
            continue
        print(f"== {path} ==")
        print(f"hits={len(rows)}")
        keys = [key for key in rows[0] if key != "hit"]
        for key in keys:
            values = [row[key] for row in rows]
            print(
                f"{key:24s} median={statistics.median(values):10.3f} "
                f"mean={statistics.mean(values):10.3f} "
                f"min={min(values):10.3f} max={max(values):10.3f}"
            )
        slow = sum(row["graph_device_us"] > 2000.0 for row in rows)
        print(f"graph_device_over_2ms={slow}/{len(rows)}")


if __name__ == "__main__":
    main()
