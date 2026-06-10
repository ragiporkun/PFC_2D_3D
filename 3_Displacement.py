import pandas as pd
import numpy as np

# Load the CSV produced by script 1
file_path = "local_maxima_with_avg_angles_floored_1000.csv"
df = pd.read_csv(file_path)

# Compute R0 = 4 * pi / sqrt(3)
R0 = 4 * np.pi / np.sqrt(3)

# Add new column disp_field = avg_dist - R0
if "avg_dist" in df.columns:
    df["disp_field"] = df["avg_dist"] - R0

# Save the updated dataframe to a new CSV
output_path = "local_maxima_with_avg_angles_floored_1000_with_disp.csv"
df.to_csv(output_path, index=False)
print(f"Saved {len(df)} records with disp_field to {output_path}")
