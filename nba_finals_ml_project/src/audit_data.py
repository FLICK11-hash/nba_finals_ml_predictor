"""Basic audit script for the NBA Finals dataset."""
from pathlib import Path
import pandas as pd

FEATURES = [
    "ortg", "drtg", "seed", "wins", "mvp5", "dpoy5", "allstars",
    "finals5", "titles5", "threept_pct", "twopt_pct", "ft_pct",
]


def main() -> None:
    path = Path("data/nba_finals_1980_2026.csv")
    df = pd.read_csv(path, skip_blank_lines=True)
    df["champion"] = pd.to_numeric(df["champion"], errors="coerce")

    print("Rows:", len(df))
    print("Years:", df["year"].min(), "to", df["year"].max())

    counts = df.groupby("year").size()
    print("\nTeams per year check:")
    print(counts[counts != 2] if (counts != 2).any() else "OK: every year has exactly 2 teams")

    completed = df[df["year"] <= 2025]
    champion_counts = completed.groupby("year")["champion"].sum()
    print("\nChampion label check:")
    print(champion_counts[champion_counts != 1] if (champion_counts != 1).any() else "OK: every completed year has exactly 1 champion")

    pred = df[df["year"] == 2026]
    print("\n2026 label check:")
    if pred.empty:
        print("WARNING: no 2026 rows found")
    elif pred["champion"].isna().all():
        print("OK: 2026 champion labels are blank")
    else:
        print("WARNING: 2026 has champion labels filled in")

    missing = df[FEATURES].isna().sum()
    print("\nMissing feature values:")
    print(missing[missing > 0] if (missing > 0).any() else "OK: no missing feature values")

    print("\nFeature ranges:")
    print(df[FEATURES].describe().T[["min", "max", "mean"]])


if __name__ == "__main__":
    main()
