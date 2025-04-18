import pandas as pd
import numpy as np
import random

# Seed for reproducibility
np.random.seed(42)

# 1. Generate Lookup Tables
def generate_lookup_tables():
    currencies = ['USD', 'EUR', 'JPY', 'GBP', 'INR']
    accounts = ['ACC1001', 'ACC1002', 'ACC1003', 'ACC1004']
    lookup_currency = pd.DataFrame({'CurrencyCode': currencies, 'ExchangeRate': np.random.uniform(0.5, 1.5, len(currencies))})
    lookup_accounts = pd.DataFrame({'AccountID': accounts, 'AccountType': ['Trading', 'Savings', 'Checking', 'Investment']})

    return lookup_currency, lookup_accounts

# 2. Generate Validation Errors
def generate_validation_errors(num_errors=10):
    trade_ids = np.random.randint(10000, 99999, num_errors)
    account_ids = np.random.choice(['ACC1001', 'ACC1002', 'ACC1003', 'ACC1004'], num_errors)
    error_types = ['Missing Quantity', 'Invalid Currency', 'Incorrect Account Type', 'Negative Amount']

    errors = pd.DataFrame({
        'ErrorID': [f"ERR{str(i+1).zfill(4)}" for i in range(num_errors)],
        'TradeID': trade_ids,
        'AccountID': account_ids,
        'ErrorType': np.random.choice(error_types, num_errors)
    })

    return errors

# 3. Generate Adjustments based on Errors and Lookup Tables
def generate_adjustments(errors_df, lookup_currency, lookup_accounts):
    adjustments = []

    for _, error in errors_df.iterrows():
        sql_statements = []
        if error['ErrorType'] == 'Missing Quantity':
            qty = random.randint(10, 1000)
            stmt = f"UPDATE Trades SET Quantity={qty} WHERE TradeID={error['TradeID']};"
            sql_statements.append(stmt)

        elif error['ErrorType'] == 'Invalid Currency':
            valid_currency = lookup_currency.sample(1)['CurrencyCode'].iloc[0]
            stmt = f"UPDATE Trades SET Currency='{valid_currency}' WHERE TradeID={error['TradeID']};"
            sql_statements.append(stmt)

        elif error['ErrorType'] == 'Incorrect Account Type':
            valid_account = lookup_accounts.sample(1)['AccountType'].iloc[0]
            stmt = f"UPDATE Accounts SET AccountType='{valid_account}' WHERE AccountID='{error['AccountID']}';"
            sql_statements.append(stmt)

        elif error['ErrorType'] == 'Negative Amount':
            amt = random.uniform(100, 5000)
            stmt = f"UPDATE Trades SET Amount={amt:.2f} WHERE TradeID={error['TradeID']};"
            sql_statements.append(stmt)

        # Simulate multi-statement adjustment (insert/delete)
        if random.choice([True, False]):
            sql_statements.append(f"INSERT INTO AdjustmentLog(ErrorID, AdjustedBy) VALUES('{error['ErrorID']}', 'User1');")

        adjustments.append({
            'ErrorID': error['ErrorID'],
            'SQLAdjustment': " ".join(sql_statements)
        })

    adjustments_df = pd.DataFrame(adjustments)
    return adjustments_df

# 4. Generate All Test Data
def generate_test_data():
    lookup_currency, lookup_accounts = generate_lookup_tables()
    validation_errors = generate_validation_errors(num_errors=10000)
    adjustments = generate_adjustments(validation_errors, lookup_currency, lookup_accounts)

    return validation_errors, lookup_currency, lookup_accounts, adjustments

# Generate and display data
validation_errors_df, lookup_currency_df, lookup_accounts_df, adjustments_df = generate_test_data()

print("=== Lookup Currency Table ===")
print(lookup_currency_df.head(), "\n")

print("=== Lookup Accounts Table ===")
print(lookup_accounts_df.head(), "\n")

print("=== Validation Errors ===")
print(validation_errors_df.head(), "\n")

print("=== Generated Adjustments ===")
print(adjustments_df.head(), "\n")

# Optionally save as CSV
validation_errors_df.to_csv('data/validation_errors.csv', index=False)
lookup_currency_df.to_csv('data/lookup_currency.csv', index=False)
lookup_accounts_df.to_csv('data/lookup_accounts.csv', index=False)
adjustments_df.to_csv('data/generated_adjustments.csv', index=False)
