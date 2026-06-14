import pandas as pd
import yfinance as yf
import time
import requests
import os
from os import listdir
from os.path import isfile, join

# get list of symbols from SEC filings
yyyy = ["2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017",
        "2018", "2019", "2020", "2021", "2022", "2023", "2024"]
mm = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

file_list = [f"data/sec_relevant/relevant_{y}-{m}.csv"
             for y in yyyy for m in mm]
file_list = file_list[:-11]  # 2024/01 last date with 2y data after filing
symbols = set()

for f in file_list:
    insider_data = pd.read_csv(f)
    symbols.update(insider_data["issuer.tradingSymbol"].dropna())

symbols_df = pd.DataFrame({"ticker": list(symbols)})
mask = symbols_df["ticker"].astype(str).str.match(r"^(?=.*[A-Za-z])[A-Za-z]+$")
df_valid = symbols_df[mask]
symb_list = df_valid["ticker"].to_list()
symb_list = sorted(symb_list)

# excel export of symbol list
# tickers = df_valid["ticker"].sort_values()
# tickers.to_excel("data/2026_01/symbolliste_ohneduplikate.xlsx")


def download_symbol(symbol: str,
                    start_date="2024-01-01",
                    end_date="2024-01-02"):

    start_time = time.time()

    try:
        df = yf.download(
            tickers=symbol,
            start=start_date,
            end=end_date,
            progress=False
        )

        if df.empty:
            print(f"Keine Daten für {symbol} erhalten.")
            return None

        df.columns = df.columns.droplevel(0)
        df = df.reset_index()
        df = df.iloc[:, [0, 1]]
        out_file = f"data/yf_close/{symbol}.csv"
        df.to_csv(out_file)
        duration = time.time() - start_time
        print(f"dowloadtime for {symbol}:{duration:.2f} seconds")

        return df

    except requests.exceptions.RequestException as e:
        print(f"Netzwerkfehler beim Download von {symbol}: {e}")
        return None


# symb_list = symb_list[605:]  # select here subset of list if needed
for s in symb_list:
    download_symbol(symbol=s, start_date="2009-01-01", end_date="2024-01-01")
    time.sleep(10)

# concat all close data
df_list = []
expected_row_count = 0

for f in file_list:
    df = pd.read_csv(f)
    expected_row_count = expected_row_count + len(df)
    df_list.append(df)

all_sec_concat = pd.concat(df_list)

all_sec_concat.to_csv("data/all_sec_concat.csv",
                      index=False)

print(f"expected row count: {expected_row_count}")
print(f"actual row count: {len(all_sec_concat)}")

sec = pd.read_csv("data/all_sec_concat.csv")
sec_filingdate_symb = sec[["transactionDate",
                           "issuer.tradingSymbol"]].drop_duplicates()
sec_filingdate_symb["transactionDate"] = pd.to_datetime(
    sec_filingdate_symb["transactionDate"])

mypath = "data/yf_close"
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
files = [f.replace(".csv", "") for f in files]

all_symbol_dfs = []

for s in files:

    path = f"data/yf_close/{s}.csv"
    if not os.path.exists(path):
        continue

    close = pd.read_csv(path)

    if s not in close.columns:
        continue

    close = close[["Date", s]].copy()
    close["Date"] = pd.to_datetime(close["Date"])
    close = close.sort_values("Date").set_index("Date")

    full_idx = pd.date_range(close.index.min(), close.index.max(), freq="D")
    close = close.reindex(full_idx)

    observed = close[s].notna().sum()
    expected = len(close)

    coverage = observed / expected

    if coverage < 0.65:
        print(f"Skipping {s} — coverage {coverage:.2%}")
        continue

    close[s] = close[s].ffill()

    filings = sec_filingdate_symb[
        sec_filingdate_symb["issuer.tradingSymbol"] == s
    ].copy()

    if filings.empty:
        continue

    filings["transactionDate"] = pd.to_datetime(filings["transactionDate"])

    result_rows = []

    for _, row in filings.iterrows():

        filing_date = row["transactionDate"]

        if filing_date > close.index.max():
            continue

        target_dates = pd.date_range(
            filing_date,
            filing_date + pd.Timedelta(days=749),
            freq="D"
        )

        tmp = close.reindex(target_dates)

        tmp[s] = tmp[s].ffill()

        tmp["days_since_filing"] = (tmp.index - filing_date).days

        wide_row = tmp.set_index("days_since_filing")[s].to_dict()

        wide_row["issuer.tradingSymbol"] = s
        wide_row["transactionDate"] = filing_date

        result_rows.append(wide_row)

    if not result_rows:
        continue

    symbol_df = pd.DataFrame(result_rows)

    day_cols = sorted([c for c in symbol_df.columns if isinstance(c, int)])

    symbol_df = symbol_df[
        ["issuer.tradingSymbol", "transactionDate"] + day_cols
    ]

    all_symbol_dfs.append(symbol_df)

close_relative_to_filing = pd.concat(all_symbol_dfs, ignore_index=True)

close_relative_to_filing.to_csv("data/close_relative_to_filing.csv", index=False)

for _row in close_relative_to_filing:
    close_relative_to_filing_targets = close_relative_to_filing_targets[]
    (close_relative_to_filing["0"] <
    close_relative_to_filing["5"])
