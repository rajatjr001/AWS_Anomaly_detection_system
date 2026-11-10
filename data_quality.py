def check_data_quality(df):
    issues = []

    for col in df.columns:
        missing = int(df[col].isna().sum())

        if missing > 0:
            issues.append({
                "column": col,
                "issue": "Missing values",
                "count": missing
            })

        if df[col].nunique(dropna=False) <= 1:
            issues.append({
                "column": col,
                "issue": "Constant column",
                "count": 1
            })

    duplicates = int(df.duplicated().sum())

    if duplicates:
        issues.append({
            "column": "All columns",
            "issue": "Duplicate rows",
            "count": duplicates
        })

    return issues