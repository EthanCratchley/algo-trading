import pandas as pd
import os

# List of CSV files and corresponding ticker symbols
files_and_tickers = [
    ('2strategy_performance_NVDA.csv', 'NVDA'),
    ('2strategy_performance_TSLA.csv', 'TSLA'),
    ('2strategy_performance_AAPL.csv', 'AAPL'),
    ('2strategy_performance_ACMR.csv', 'ACMR'),
    ('2strategy_performance_ARM.csv', 'ARM'),
    ('2strategy_performance_CRM.csv', 'CRM'),
    ('2strategy_performance_DBX.csv', 'DBX'),
    ('2strategy_performance_MSFT.csv', 'MSFT'),
    ('2strategy_performance_PANW.csv', 'PANW'),
    ('2strategy_performance_SMCI.csv', 'SMCI'),
]

# Get the current directory where the script is located
current_directory = os.path.dirname(os.path.abspath(__file__))

# Initialize an empty DataFrame to store the results
combined_results = pd.DataFrame()

# Process each CSV file
for csv_file, ticker in files_and_tickers:
    # Construct the full path to the CSV file
    csv_file_path = os.path.join(current_directory, csv_file)
    
    try:
        # Load the data from the CSV file
        df = pd.read_csv(csv_file_path)
        
        # Check if required columns exist
        if 'Annualized Return' not in df.columns or 'Sharpe Ratio' not in df.columns:
            print(f"Skipping {csv_file} as it does not contain required columns.")
            continue
        
        # Check if the DataFrame is empty
        if df.empty:
            print(f"Skipping {csv_file} as it is empty.")
            continue
        
        # Find the row with the highest annual return
        best_annual_return_row = df.loc[df['Annualized Return'].idxmax()]
        
        # Find the row with the highest Sharpe ratio
        best_sharpe_ratio_row = df.loc[df['Sharpe Ratio'].idxmax()]
        
        # Combine these rows into a new DataFrame, avoiding duplicate if the same
        if best_annual_return_row.equals(best_sharpe_ratio_row):
            best_rows = pd.DataFrame([best_annual_return_row])
        else:
            best_rows = pd.DataFrame([best_annual_return_row, best_sharpe_ratio_row])
        
        # Add the ticker symbol to each row
        best_rows.insert(0, 'Ticker', ticker)
        
        # Append the best rows to the combined results
        combined_results = pd.concat([combined_results, best_rows], ignore_index=True)
    
    except Exception as e:
        print(f"An error occurred while processing {csv_file}: {e}")

# Construct the full path to the new CSV file
best_csv_file_path = os.path.join(current_directory, '2best.csv')

# Write the combined results to the new CSV file
combined_results.to_csv(best_csv_file_path, index=False)

print("The best strategy statistics from all tickers have been saved to '2best.csv'")
