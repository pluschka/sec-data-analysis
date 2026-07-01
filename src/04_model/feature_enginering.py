from pathlib import Path
import pandas as pd
from scipy.stats import zscore

project_root = Path().resolve()
X_train = pd.read_csv(project_root / "data/cleaned_X_train.csv")
X_test = pd.read_csv(project_root / "data/cleaned_X_test.csv")
datasets = [X_train, X_test]

for i, df in enumerate(datasets):
    df['reportingOwner.name'] = (
        df['reportingOwner.name']
        .str.replace(r'[^A-Za-z ]+', '', regex=True)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
        .str.upper()
    )
    df['past_filings_per_person_eda'] = (
        df.groupby('reportingOwner.name')
        .cumcount()
    )

    df['past_filings_per_person'] = zscore(df['past_filings_per_person_eda'])

    # high price dummy, median only from train_df to prefent data leackage
    median = X_train['amounts.pricePerShare'].median()

    df['high_price'] = (
        df['amounts.pricePerShare']
        .gt(median)
        .astype('bool')
    )

    # high change in holdings dummy
    # median only from train_df to prefent data leackage
    df['high_change_in_holdings'] = (
        df['holding_change_percent'] >
        X_train['holding_change_percent']
        .median()).astype('bool')

    # large trade flag
    df['is_large_trade'] = (
        df['amounts.shares'] >
        X_train['amounts.shares'].quantile(0.8)
        .astype('bool')
    )

    # large trade an high percent change in holdings
    df['is_large_trade_x_high_change_in_holdings'] = (
        df['is_large_trade'] & df['holding_change_percent']
    ).astype('bool')

X_train.to_csv(project_root / "data/cleaned_X_train.csv", index=False)
X_test.to_csv(project_root / "data/cleaned_X_test.csv", index=False)
