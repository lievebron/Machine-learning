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

# Detemine and print correlation between x and y for each dataset
correlations = df.groupby('dataset').apply(lambda group: group['x'].corr(group['y']))
print('Correlation between x and y for each dataset:')                      
print(correlations)

# Determine and print covariance matrix for each dataset
cov_matrices = df.groupby('dataset').apply(lambda group: np.cov(group['x'], group['y']))
print('Covariance matrix for each dataset:')
for dataset, cov_matrix in cov_matrices.items():
    print(f'Dataset: {dataset}')
    print(cov_matrix)
    print()

# Determine linear regression between x and y for each dataset, and print slope,intercept and r-value for each dataset (hint use scipy.stats.linregress)
regressions = df.groupby('dataset').apply(lambda group: stats.linregress(group['x'], group['y']))
print('Linear regression for each dataset:')
for dataset, regression in regressions.items():
    print(f'Dataset: {dataset}')
    print(f'Slope: {regression.slope}, Intercept: {regression.intercept}, R-value: {regression.rvalue}')
    print()     


# Create scatterplots for all datasets (hint: use FacetGrid and map_dataframe)
g = sns.FacetGrid(df, col='dataset', col_wrap=3, height=4)
g.map_dataframe(sns.scatterplot, x='x', y='y')
g.set_axis_labels('x', 'y')
g.set_titles('Dataset: {col_name}')
plt.tight_layout()
plt.show()

# Create scatterplots including the regression line for all datasets (hint: use lmplot)
g = sns.lmplot(x='x', y='y', col='dataset', col_wrap=3, height=4, data=df)
g.set_axis_labels('x', 'y')
g.set_titles('Dataset: {col_name}')
plt.tight_layout()
plt.show()

