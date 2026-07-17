"""Count has_content True / False in train.csv."""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "train.csv"


def main() -> None:
    n_true = 0
    n_false = 0
    total = 0

    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            value = row["has_content"].strip().lower()
            if value == "true":
                n_true += 1
            elif value == "false":
                n_false += 1

    print(f"train.csv: {CSV_PATH}")
    print(f"total rows        : {total}")
    print(f"has_content True  : {n_true}")
    print(f"has_content False : {n_false}")


if __name__ == "__main__":
    main()
