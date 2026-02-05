import pandas as pd
from missingfcup.core.MissingData import MissingData
from missingfcup.plots.Panel import Panel


def main():
    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    df = pd.read_csv(
        "https://raw.githubusercontent.com/ResidentMario/missingno-data/master/nyc_collision_factors.csv"
    )

    md = MissingData(df)

    print("\n=== MissingFCUP Bar Chart Showcase ===")
    print(f"Rows: {len(df)} | Columns: {len(df.columns)}")
    print(f"Overall missingness: {md.total_missingness:.2%}")

    # ------------------------------------------------------------------
    # 1️⃣ Global missingness ranking (percentage)
    # ------------------------------------------------------------------
    bc_global_pct = md.barchart(
        mode="percentage",
        order_by=[{"column": "__missing__", "type": "numeric", "ascending": False}],
        title="Global Missingness Ranking (%)",
        max_columns=20,
    )
    bc_global_pct.show()

    # ------------------------------------------------------------------
    # 2️⃣ Global missingness ranking (absolute counts)
    # ------------------------------------------------------------------
    bc_global_count = md.barchart(
        mode="count",
        order_by=[{"column": "__missing__", "type": "numeric", "ascending": False}],
        title="Global Missingness Ranking (Count)",
        max_columns=20,
    )
    bc_global_count.show()

    # ------------------------------------------------------------------
    # 3️⃣ Almost-complete columns (≤ 5% missing)
    # ------------------------------------------------------------------
    bc_almost_complete = md.barchart(
        mode="percentage",
        completeness_threshold=0.95,
        order_by=[{"column": "__missing__", "type": "numeric", "ascending": True}],
        title="Almost Complete Columns (≤ 5% Missing)",
    )
    bc_almost_complete.show()

    # ------------------------------------------------------------------
    # 4️⃣ Heavily missing columns only (≥ 50%)
    # ------------------------------------------------------------------
    bc_heavily_missing = md.barchart(
        mode="percentage",
        threshold=50,
        order_by=[{"column": "__missing__", "type": "numeric", "ascending": False}],
        title="Heavily Missing Columns (≥ 50%)",
    )
    bc_heavily_missing.show()

    # ------------------------------------------------------------------
    # 5️⃣ Present vs Missing (stacked composition)
    # ------------------------------------------------------------------
    bc_stacked = md.barchart(
        selected_columns=[
            "NUMBER OF PERSONS INJURED",
            "NUMBER OF PERSONS KILLED",
            "CONTRIBUTING FACTOR VEHICLE 1",
            "CONTRIBUTING FACTOR VEHICLE 2",
        ],
        stacked=True,
        mode="percentage",
        title="Present vs Missing Composition (%)",
    )
    bc_stacked.show()

    # ------------------------------------------------------------------
    # 6️⃣ Horizontal layout (readability for long names)
    # ------------------------------------------------------------------
    bc_horizontal = md.barchart(
        mode="percentage",
        orientation="horizontal",
        max_columns=15,
        title="Missingness (%) – Horizontal Layout",
    )
    bc_horizontal.show()

    # ------------------------------------------------------------------
    # 7️⃣ Domain-focused subset (accident severity)
    # ------------------------------------------------------------------
    bc_domain = md.barchart(
        selected_columns=[
            "NUMBER OF PERSONS INJURED",
            "NUMBER OF PERSONS KILLED",
            "NUMBER OF PEDESTRIANS INJURED",
            "NUMBER OF CYCLIST INJURED",
            "NUMBER OF MOTORIST INJURED",
        ],
        mode="percentage",
        title="Missingness in Injury-Related Variables",
    )
    bc_domain.show()

    # ------------------------------------------------------------------
    # 8️⃣ Top-K most informative columns only
    # ------------------------------------------------------------------
    bc_topk = md.barchart(
        mode="percentage",
        order_by=[{"column": "__missing__", "type": "numeric", "ascending": False}],
        max_columns=10,
        title="Top 10 Most Missing Columns",
    )
    bc_topk.show()

    # ------------------------------------------------------------------
    # 9️⃣ Alphabetical ordering (baseline sanity check)
    # ------------------------------------------------------------------
    bc_alpha = md.barchart(
        mode="percentage",
        order_by=[{"column": "__column__", "type": "categorical", "ascending": True}],
        title="Missingness by Column Name (Alphabetical)",
        max_columns=20,
    )
    bc_alpha.show()

if __name__ == "__main__":
    main()
