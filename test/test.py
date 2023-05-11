import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import numpy as np
import os

print(os.getcwd())

df = pd.read_csv('test/summary.csv')

sns.set_theme(style="darkgrid")

fig, axes = plt.subplots(1, 3)
fig.suptitle('Barplots of the methods execution time, iterations and relative error')
fig.set_size_inches(20, 6)

ax1 = sns.barplot(data=df, ax=axes[0], x='Method', y='Time', hue='Tolerance', 
                  errorbar=None, palette="crest")
ax1.set_yscale('log')

ax2 = sns.barplot(data=df, ax=axes[1], x='Method', y='Iterations', hue='Tolerance', 
                  errorbar=None, palette="crest")
ax2.set_yscale('log')

ax3 = sns.barplot(data=df, ax=axes[2], x='Method', y='Relative error', hue='Tolerance', 
                  errorbar=None, palette="crest")
ax3.set_yscale('log')


plt.tight_layout()
plt.savefig('test/time-iterations-error_barplots.png')


methods = df['Method'].unique()

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle('Barplots of the methods iterations for different densities')
for i, method in enumerate(methods):
    grouped = df.loc[df['Method'] == method]
    axes[i//2, i%2].set_title(method)
    barplot = sns.barplot(data=grouped, ax=axes[i//2, i%2], x='Density', 
                      y='Iterations', errorbar=None, palette="viridis", width=0.6)
    barplot.set_yscale('log')
    ticks = barplot.get_xticklabels()
    ticks = [t.get_text()[:6] for t in ticks]
    barplot.set_xticklabels(ticks)

plt.tight_layout()
plt.savefig('test/density-iterations_barplots.png')


fig, axes = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle('Barplots of the methods execution time for different densities')
for i, method in enumerate(methods):
    grouped = df.loc[df['Method'] == method]
    axes[i//2, i%2].set_title(method)
    barplot = sns.barplot(data=grouped, ax=axes[i//2, i%2], x='Density', 
                      y='Time', errorbar=None, palette="viridis", width=0.6)
    barplot.set_yscale('log')
    ticks = barplot.get_xticklabels()
    ticks = [t.get_text()[:6] for t in ticks]
    barplot.set_xticklabels(ticks)

plt.tight_layout()
plt.savefig('test/density-time_barplots.png')

sns.set_theme(style="white")

fig, axes = plt.subplots(2, 2, figsize=(30, 20))
fig.suptitle('Correlation matrix', fontsize=40)
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.set(font_scale=1.6)

for i, method in enumerate(methods):
    grouped = df.loc[df['Method'] == method]
    axes[i//2, i%2].set_title(method, fontsize=30)
    corr = grouped[['Size', 'Density', 'Tolerance', 'Relative error', 'Time', 'Iterations']].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), 1)
    heatmap = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0, square=True, 
                          linewidths=.5, cbar_kws={"shrink": .5}, annot=True, ax=axes[i//2, i%2])
    heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize = 20, rotation=45, horizontalalignment='right')
    heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize = 20, rotation=0, horizontalalignment='right')

plt.tight_layout()
plt.savefig('test/heatmaps.png')



df_m = df[['Matrix', 'Method', 'Time', 'Iterations', 'Relative error', 'Tolerance']]
df_info_matrix = df[['Matrix', 'Size', 'Density']].drop_duplicates().reset_index(drop=True)

# group df by matrix and split it into 4 dataframes except 
df_grouped = df_m.groupby('Matrix')
df_grouped_list = [df_grouped.get_group(x) for x in df_grouped.groups]

#df_density_iter = pd.DataFrame(columns=['Method', 'Density', 'Iterations'])

corr_method = df.groupby('Method')[['Density', 'Iterations']].corr()
print(corr_method)

#df_grouped_method = df_density_iter.groupby('Method')



