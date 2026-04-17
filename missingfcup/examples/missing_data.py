import pandas as pd

from missingfcup import MissingData


def summarize_properties(md: MissingData) -> None:
    """Showcase MissingData properties."""
    print("\nDataframe overview")
    print("------------------")
    print(f"Rows: {md.number_of_rows}")
    print(f"Columns: {md.number_of_columns}")
    print(f"Total missing rate: {md.total_missing_rate:.1%}")
    print(f"Total missing count: {md.total_missing_count:,}")
    print(f"Columns with no missing values: {len(md.columns_complete)}")
    print(f"Rows with no missing values: {len(md.rows_complete)}")
    print(f"Rows with at least one missing value: {len(md.rows_with_missing)}")
    print()

    print("Column-level properties (top 5)")
    print("-------------------------------")
    top_missing_rate = md.col_missing_rate.sort_values(ascending=False).head(5)
    print("Missing rate:")
    print(top_missing_rate.to_string())
    print("Missing count:")
    print(md.col_missing_count.loc[top_missing_rate.index].to_string())
    print("Missing percent:")
    print(md.col_missing_percent.loc[top_missing_rate.index].round(2).to_string())
    print("Completeness:")
    print(md.col_completeness.loc[top_missing_rate.index].round(3).to_string())
    print()

    print("Row-level properties (top 5)")
    print("----------------------------")
    top_row_missing = md.row_missing_rate.sort_values(ascending=False).head(5)
    print("Missing rate:")
    print(top_row_missing.to_string())
    print("Missing count:")
    print(md.row_missing_count.loc[top_row_missing.index].to_string())
    print("Missing percent:")
    print(md.row_missing_percent.loc[top_row_missing.index].round(2).to_string())
    print("Completeness:")
    print(md.row_completeness.loc[top_row_missing.index].round(3).to_string())
    print()


def summarize_methods(md: MissingData) -> None:
    """Showcase MissingData methods."""
    print("Missingness patterns")
    print("--------------------")
    print("Most common row missing patterns (top 3):")
    print(md.missing_pattern_counts(max_patterns=3).to_string())
    print(f"Unique row missingness patterns: {len(md.missing_pattern_in_rows_unique)}")
    print()

    print("Threshold-based filtering")
    print("-------------------------")
    print(f"Rows missing more than 20%: {len(md.rows_above_missing_threshold(0.2))}")
    print(f"Rows missing more than 60%: {len(md.rows_above_missing_threshold(0.6))}")
    print(f"Columns missing more than 20%: {len(md.columns_above_missing_threshold(0.2))}")
    print(f"Columns missing more than 60%: {len(md.columns_above_missing_threshold(0.6))}")
    print()

    print("Column correlations")
    print("-------------------")
    correlation = md.missing_corr
    print("Correlation (first 3x3 block):")
    print(correlation.iloc[:3, :3].round(2))
    print(f"Perfectly correlated columns: {md.perfectly_correlated_missing_columns()}")
    print()


def conclusions(md: MissingData) -> None:
    """Print simple conclusions derived from MissingData metrics."""
    print("Conclusions")
    print("-----------")
    if md.total_missing_rate == 0:
        print("- The dataset has no missing values.")
    elif md.total_missing_rate < 0.05:
        print("- Overall missingness is low (<5%).")
    elif md.total_missing_rate < 0.2:
        print("- Overall missingness is moderate (5%-20%).")
    else:
        print("- Overall missingness is high (>=20%).")

    if len(md.columns_complete) > 0:
        print(f"- There are {len(md.columns_complete)} columns with complete data.")
    else:
        print("- No columns are completely filled.")

    top_missing_column = md.col_missing_rate.sort_values(ascending=False).head(1)
    if not top_missing_column.empty:
        col_name = top_missing_column.index[0]
        col_rate = top_missing_column.iloc[0]
        print(f"- The most missing column is '{col_name}' with {col_rate:.1%} missing.")

    max_row_missing = md.row_missing_rate.max()
    if max_row_missing >= 0.9:
        print("- At least one row is almost entirely missing (>=90%).")
    elif max_row_missing >= 0.5:
        print("- Some rows are heavily incomplete (>=50%).")
    else:
        print("- No row is missing more than half of its values.")

    perfect_pairs = md.perfectly_correlated_missing_columns()
    if perfect_pairs:
        print("- Some columns always go missing together; consider redundant fields.")
    else:
        print("- No perfectly correlated missingness pairs were found.")

    print()


# def missingness_mechanism_demo(md: MissingData) -> None:
#     """Show heuristic evidence for MCAR/MAR (MNAR not testable from observed data)."""
#     print("Missingness mechanism (heuristic)")
#     print("---------------------------------")
#     print(
#         "Note: MCAR/MAR are assessed heuristically via associations with observed variables; "
#         "MNAR cannot be confirmed from observed data alone."
#     )
#     summary = md.missingness_mechanism_summary()
#     if summary.empty:
#         print("No columns with missing values to analyze.")
#         print()
#         return

#     print("Summary by column (top 10 by missing rate):")
#     print(summary.head(10).to_string(index=False))
#     print()

#     top_missing_columns = (
#         md.col_missing_rate.sort_values(ascending=False).loc[lambda s: s > 0].head(3)
#     )
#     if not top_missing_columns.empty:
#         print("Top missing columns and their strongest associations:")
#         for column in top_missing_columns.index:
#             associations = md.missingness_associations(column).head(5)
#             print(f"\n{column}")
#             if associations.empty:
#                 print("  No usable associations found.")
#             else:
#                 print(associations.to_string(index=False))
#         print()


def main() -> None:
    """Load sample data and run the MissingData demo."""
    url = "https://raw.githubusercontent.com/ResidentMario/missingno-data/master/nyc_collision_factors.csv"
    df = pd.read_csv(url)
    md = MissingData(df)

    summarize_properties(md)
    summarize_methods(md)
    conclusions(md)
    # missingness_mechanism_demo(md)


if __name__ == "__main__":
    main()
