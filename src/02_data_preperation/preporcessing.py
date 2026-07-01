import pandas as pd
import numpy as np
import sys
from pathlib import Path

project_root = Path().resolve()
sys.path.append(str(project_root / "src/02_data_preperation"))

# add targets on close dataset
close_relative_to_filing = pd.read_csv(project_root /
                                       "data/close_relative_to_filing.csv",
                                       index_col=False)
close_relative_to_filing_target_sum = close_relative_to_filing[[
    "issuer.tradingSymbol", "transactionDate"]]

for _row in close_relative_to_filing:
    for i in ['2', '3', '4', '5', '6', '7', '8', '9', '10', '20', '60']:
        varname = "target_" + i
        close_relative_to_filing_target_sum[varname] = (
            close_relative_to_filing["0"] <
            close_relative_to_filing[i]
            )

close_relative_to_filing["transactionDate"] = pd.to_datetime(
    close_relative_to_filing["transactionDate"]
)

close_relative_to_filing_target_sum["transactionDate"] = pd.to_datetime(
    close_relative_to_filing_target_sum["transactionDate"]
)

df = pd.read_csv(project_root/"data/all_sec_concat.csv",
                 index_col=0)

df["transactionDate"] = pd.to_datetime(
    df["transactionDate"]
)

# postTransactionAmounts.sharesOwnedFollowingTransaction is the amount of
# shares after each filling
post_shares = 'postTransactionAmounts.sharesOwnedFollowingTransaction'
shares = 'amounts.shares'

holdings_before_filing = (
    df[post_shares]
    - df[shares]
)

# calculation: (amount of shares in this filling/pre amount holdings)*100 for
# percent to know how much the person bought in comparison to what they owned
# pre amount of shares = post_shares - shares
# if old amount of shares = 0 then division by 0 would cause problems
df['holding_change_percent'] = np.where(
    holdings_before_filing == 0, 0, (df['amounts.shares'] /
                                     holdings_before_filing) * 100)
# remove implausible data
df['holding_change_percent'] = abs(df['holding_change_percent'])

# change string into bool, D = direct, I = indirect
df['direct_ownership'] = (
    df['ownershipNature.directOrIndirectOwnership'].eq('D')
).astype('bool')
df = df.drop(columns="ownershipNature.directOrIndirectOwnership")

# for model readability month_sin and month_cos
df['transaction_month'] = pd.DatetimeIndex(
    df['transactionDate']).month
df["month_sin"] = np.sin(2 * np.pi * df["transaction_month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["transaction_month"] / 12)

"""
plt.plot(df["month_sin"], df["month_cos"])
plt.title("Transformation of transaction_month in month_sin and month_cos")
plt.xlabel("month_sin")
plt.ylabel("month_cos")
plt.show()

df = df.drop(columns="transaction_month")

"""

merged_df = pd.merge(
    df,
    close_relative_to_filing_target_sum,
    on=["issuer.tradingSymbol", "transactionDate"],
    how="left"
)

merged_df.to_csv(project_root / "data/inner_close_sec_uncleaned.csv",
                 index=False)
