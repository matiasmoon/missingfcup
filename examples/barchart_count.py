import pandas as pd
from missingfcup import MissingData


def main():
    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    df = pd.read_csv(
        "https://raw.githubusercontent.com/ResidentMario/missingno-data/master/nyc_collision_factors.csv"
    )

    md = MissingData(df)

    print("\n=== Missing Count by Column (Bar Chart) ===")
    print(f"Rows: {len(df)} | Columns: {len(df.columns)}")
    print(f"Overall missingness: {md.total_missing_rate:.2%}")

    # ------------------------------------------------------------------
    # 1️⃣ Basic default (missing counts)
    # ------------------------------------------------------------------
    md.barchart_count().show()

    # ------------------------------------------------------------------
    # 2️⃣ Missing counts with a title
    # ------------------------------------------------------------------
    bc_titled = md.barchart_count(title="Missing Count by Column")
    bc_titled.show()

    # ------------------------------------------------------------------
    # 3️⃣ Missing counts for a small subset
    # ------------------------------------------------------------------
    bc_subset = md.barchart_count(
        selected_columns=[
            "NUMBER OF PERSONS INJURED",
            "NUMBER OF PERSONS KILLED",
            "CONTRIBUTING FACTOR VEHICLE 1",
        ],
        title="Missing Count (Selected Columns)",
    )
    bc_subset.show()

    # ------------------------------------------------------------------
    # 4️⃣ Missing counts (top 15 columns)
    # ------------------------------------------------------------------
    bc_top15 = md.barchart_count(
        order_by=[{"column": "__missing__", "type": "numeric", "ascending": False}],
        max_columns=15,
        title="Missing Count (Top 15 Columns)",
    )
    bc_top15.show()

    # ------------------------------------------------------------------
    # 5️⃣ Missing counts (most missing first)
    # ------------------------------------------------------------------
    bc_most_missing = md.barchart_count(
        order_by=[{"column": "__missing__", "type": "numeric", "ascending": False}],
        title="Missing Count (Most Missing First)",
        max_columns=20,
    )
    bc_most_missing.show()

    # ------------------------------------------------------------------
    # 6️⃣ Missing counts (least missing first)
    # ------------------------------------------------------------------
    bc_least_missing = md.barchart_count(
        order_by=[{"column": "__missing__", "type": "numeric", "ascending": True}],
        title="Missing Count (Least Missing First)",
        max_columns=20,
    )
    bc_least_missing.show()

    # ------------------------------------------------------------------
    # 7️⃣ Missing counts for injury-related columns
    # ------------------------------------------------------------------
    bc_injury = md.barchart_count(
        selected_columns=[
            "NUMBER OF PERSONS INJURED",
            "NUMBER OF PERSONS KILLED",
            "NUMBER OF PEDESTRIANS INJURED",
            "NUMBER OF CYCLIST INJURED",
            "NUMBER OF MOTORIST INJURED",
        ],
        title="Missing Count (Injury-Related Columns)",
    )
    bc_injury.show()

    # ------------------------------------------------------------------
    # 8️⃣ Missing counts after excluding high-missingness columns
    # ------------------------------------------------------------------
    bc_filtered = md.barchart_count(
        ignore_high_missingness=True,
        high_missingness_threshold=0.6,
        title="Missing Count (Exclude ≥ 60% Missing)",
    )
    bc_filtered.show()

    # ------------------------------------------------------------------
    # 9️⃣ Present counts with completeness-based filtering
    # ------------------------------------------------------------------
    bc_complete = md.barchart_count(
        completeness_mode="most",
        completeness_threshold=0.9,
        max_columns_by_completeness=15,
        value="present",
        title="Present Count (Most Complete Columns)",
    )
    bc_complete.show()

    # ------------------------------------------------------------------
    # 🔟 Missing vs Present (stacked counts)
    # ------------------------------------------------------------------
    bc_stacked = md.barchart_count(
        selected_columns=[
            "NUMBER OF PERSONS INJURED",
            "NUMBER OF PERSONS KILLED",
            "CONTRIBUTING FACTOR VEHICLE 1",
            "CONTRIBUTING FACTOR VEHICLE 2",
        ],
        show_both=True,
        title="Missing vs Present (Counts)",
    )
    bc_stacked.show()

if __name__ == "__main__":
    main()
