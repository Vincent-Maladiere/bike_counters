import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Define the path to the data file
data_path = os.path.join(os.path.dirname(__file__), 'data', 'train.parquet')
import seaborn as sns


df = pd.read_parquet(data_path)
print(df.head())

y = df['log_bike_count']
X = df.drop(columns=['log_bike_count'])
