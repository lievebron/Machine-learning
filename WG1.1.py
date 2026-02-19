# Machine-learning
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy import stats
import numpy as np

# Print nr of datasets
df = pd.read_csv('datasets.csv')
num_df = df['dataset'].nunique()
print('The number of datasets is:', num_df)

# Print nr of datasets
df = pd.read_csv('datasets.csv')
num_df = df['dataset'].nunique()    

# Print names of datasets
dataset_names = df['dataset'].unique()
print('The names of the datasets are:', dataset_names)  

# Print statistics per dataset (count, mean, variance, std dev) [hint: use groupby]
stats_per_dataset = df.groupby('dataset').agg({
    'x': ['count', 'mean', 'var', 'std'],
})
print('Statistics per dataset:')
print(stats_per_dataset)   

# Create violin plots of x-coordinates per dataset, next to each other and y-coordinates per dataset, next to each other
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)                   
sns.violinplot(x='dataset', y='x', data=df)
plt.title('Violin Plot of x-coordinates per Dataset')
plt.subplot(1, 2, 2)
sns.violinplot(x='dataset', y='y', data=df)
plt.title('Violin Plot of y-coordinates per Dataset')
plt.tight_layout()
plt.show()





