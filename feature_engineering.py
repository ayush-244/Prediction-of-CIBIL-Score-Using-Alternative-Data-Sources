import pandas as pd

def engineer_features(df):
    df['monthly_txn'] = df['total_txn_amount'] / df['active_months']
    df['income_to_spend_ratio'] = df['monthly_income'] / (df['monthly_txn'] + 1)
    df['repayment_behavior'] = df['on_time_payments'] / (df['loan_count'] + 1)
    df.fillna(0, inplace=True)
    return df