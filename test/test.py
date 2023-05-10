import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import numpy as np
import os

print(os.getcwd())

df = pd.read_csv('test/summary.csv')

sns.set_theme(style="darkgrid")

fig, axes = plt.subplots(2, 2)
fig.suptitle('Barplots of the methods execution time, iterations and relative error')
fig.set_size_inches(15, 15)

ax1 = sns.barplot(data=df, ax=axes[0, 0], x='Method', y='Time', hue='Tolerance', errorbar=None, palette="crest")
ax1.set_yscale('log')

ax2 = sns.barplot(data=df, ax=axes[0, 1], x='Method', y='Iterations', hue='Tolerance', errorbar=None, palette="crest")
ax2.set_yscale('log')

ax3 = sns.barplot(data=df, ax=axes[1, 0], x='Method', y='Relative error', hue='Tolerance', errorbar=None, palette="crest")
ax3.set_yscale('log')

ax4 = sns.barplot(data=df.astype({'Density': float}), ax=axes[1, 1], x='Density', y='Iterations', hue='Method', errorbar=None, palette="viridis")
ax4.set_yscale('log')

ticks = ax4.get_xticklabels()
ticks = [t.get_text()[:6] for t in ticks]
ax4.set_xticklabels(ticks)

plt.tight_layout()
plt.savefig('test/barplot.png')

sns.set_theme(style="white")
df_corr = df[['Size', 'Density', 'Tolerance', 'Relative error', 'Time', 'Iterations']]
corr = df_corr.corr()
mask = np.triu(np.ones_like(corr, dtype=bool), 1)
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
plt.savefig('test/heatmap.png')