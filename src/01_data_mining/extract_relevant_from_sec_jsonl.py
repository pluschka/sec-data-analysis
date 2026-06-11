import pandas as pd


def extract_relevant_from_jsonl(yyyy_mm='2018-01'):
    # load jsonl from zip file
    insider_data = pd.read_json(f'data/{yyyy_mm}.jsonl.gz',
                                lines=True,
                                compression='gzip')

    # choose relevant columns
    insider_data = insider_data[['issuer',
                                 'reportingOwner',
                                 'nonDerivativeTable']]

    # extract relevant info from dicts of columns 'issuer' and 'reportingOwner'
    def extract(df, row, value):
        df = df.copy()
        new_row_name = f'{row}.{value}'
        df[new_row_name] = df[row].apply(
            lambda x: x[value] if isinstance(x, dict) else None
        )
        return df

    insider_data = extract(insider_data, 'issuer', 'name')
    insider_data = extract(insider_data, 'issuer', 'tradingSymbol')
    insider_data = extract(insider_data, 'reportingOwner', 'name')
    insider_data = extract(insider_data, 'reportingOwner', 'relationship')
    insider_data = extract(insider_data, 'reportingOwner.relationship',
                                         'isDirector')
    insider_data = extract(insider_data, 'reportingOwner.relationship',
                                         'isOfficer')
    insider_data = extract(insider_data, 'reportingOwner.relationship',
                                         'isTenPercentOwner')
    insider_data = extract(insider_data, 'reportingOwner.relationship',
                                         'isOther')

    # extract relevant info from list of dicts in the column
    # 'nonDerivativeTable'
    tx_list = insider_data['nonDerivativeTable'].apply(
        lambda x: x.get('transactions', []) if isinstance(x, dict) else []
    )

    tx_exp = tx_list.explode()

    non_deriv_df = pd.json_normalize(tx_exp).set_index(tx_exp.index)

    # join tables
    insider_data = insider_data.merge(non_deriv_df,
                                      left_index=True, right_index=True,
                                      how='left', validate='one_to_many')

    # filter relevant rows

    # drop NAs
    insider_data = insider_data.dropna(subset=['issuer.tradingSymbol'])
    insider_data = insider_data.drop(
        insider_data[insider_data['issuer.tradingSymbol']
                               .str
                               .lower()
                               .isin(['na', 'n/a', 'none'])].index)

    # P = Open market or private purchase of non-derivative or derivative
    # securities
    insider_data = insider_data[insider_data['coding.code'] == 'P']

    # exclude some form 5 for consistency
    insider_data = insider_data[insider_data['coding.formType'] == '4']

    # exclude rows with 0 shares
    insider_data = insider_data[insider_data['amounts.shares'] > 0]

    # A = acquired, D = disposed
    insider_data = insider_data[
                   insider_data['amounts.acquiredDisposedCode'] == 'A']

    # Exclude stock grants/awards and very small penny stocks >= 2$
    insider_data = insider_data[insider_data['amounts.pricePerShare'] >= 2]

    # making sure there are no date outliers
    start = pd.to_datetime(yyyy_mm, format='%Y-%m')
    min_transactionDate = start + pd.DateOffset(months=-3)
    insider_data['transactionDate'] = pd.to_datetime(
                                        insider_data['transactionDate'])
    insider_data = insider_data[
                   insider_data['transactionDate'] >= min_transactionDate]

    # normalize ticker
    insider_data['issuer.tradingSymbol'] = (
        insider_data['issuer.tradingSymbol']
        .astype(str)
        .str.upper()
        .str.strip()
        .dropna()
    )

    # select relevant columns
    insider_data = insider_data[[
        'issuer.name',
        'issuer.tradingSymbol',
        'reportingOwner.name',
        'transactionDate',
        'amounts.shares',
        'amounts.pricePerShare',
        'postTransactionAmounts.sharesOwnedFollowingTransaction',
        'ownershipNature.directOrIndirectOwnership',
        'reportingOwner.relationship.isDirector',
        'reportingOwner.relationship.isOfficer',
        'reportingOwner.relationship.isTenPercentOwner',
        'reportingOwner.relationship.isOther']]

    # save data
    insider_data.to_csv(f'data/relevant_{yyyy_mm}.csv',
                        header=True,
                        index=False)

    return insider_data
