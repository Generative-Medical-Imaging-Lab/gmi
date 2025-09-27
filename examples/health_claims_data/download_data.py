import kagglehub
import os 

kagglehub.login()

example_dir = os.path.dirname(os.path.abspath(__file__))

# Download latest version
path = kagglehub.dataset_download("dlr47685/health-data")

print(f"Dataset downloaded to: {path}")

# get the only xlsx file in the directory
for file in os.listdir(path):
    if file.endswith(".xlsx"):
        data_path = os.path.join(path, file)
        break
print(f"Data file path: {data_path}")

import pandas as pd
data = pd.read_excel(data_path)
print(data.head())
print(data.columns)
print(f"Number of rows: {len(data)}")
print(f"Number of columns: {len(data.columns)}")
# save the data to the example directory
data.to_csv(os.path.join(example_dir, "health_data.csv"), index=False)
print(f"Data saved to: {os.path.join(example_dir, 'health_data.csv')}")




