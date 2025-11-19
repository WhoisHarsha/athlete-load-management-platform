import sqlite3
import pandas as pd

print("="*80)
print("EXPORTING SQLITE DATA TO CSV")
print("="*80)

# Connect to SQLite database
conn = sqlite3.connect('D:/athlete-load-management-platform/data/raw/athlete_performance.db')

# List of tables to export
tables = [
    'training_sessions',
    'performance_metrics', 
    'load_calculations'
]

# Export each table
for table in tables:
    try:
        df = pd.read_sql(f"SELECT * FROM {table}", conn)
        csv_filename = f'{table}.csv'
        df.to_csv(csv_filename, index=False)
        print(f"✓ Exported {table}: {len(df)} rows → {csv_filename}")
    except Exception as e:
        print(f"✗ Error exporting {table}: {e}")

conn.close()

print("\n" + "="*80)
print("EXPORT COMPLETE!")
print("="*80)