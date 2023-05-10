import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

print(os.getcwd())

df = pd.read_csv('test/summary.csv')

sns.set_theme(style="darkgrid")
sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)

sns.barplot(data=df, x='Method', y='Time', hue='Tolerance', errorbar=None, palette="crest")
plt.yscale('log')
plt.tight_layout()
plt.savefig('test/barplot_method-time.png')

plt.clf()
sns.barplot(data=df, x='Method', y='Iterations', hue='Tolerance', errorbar=None, palette="crest")
plt.yscale('linear')
plt.tight_layout()
plt.savefig('test/barplot_method-iterations.png')

plt.clf()
sns.barplot(data=df, x='Method', y='Relative error', hue='Tolerance', errorbar=None, palette="crest")
plt.yscale('log')
plt.tight_layout()
plt.savefig('test/barplot_method-relative_error.png')

plt.clf()
sns.barplot(data=df.astype({'Density': float}), x='Density', y='Iterations', hue='Method', errorbar=None, palette="crest")
plt.yscale('log')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('test/barplot_density-iterations.png')

sns.set_theme(style="white")
df_corr = df[['Size', 'Density', 'Tolerance', 'Relative error', 'Time', 'Iterations']]
corr = df_corr.corr()
mask = np.triu(np.ones_like(corr, dtype=bool), 1)
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
plt.savefig('test/heatmap.png')