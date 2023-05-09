import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


df = pd.read_csv('summary.csv')

sns.set_theme(style="darkgrid")

sns.barplot(data=df, x='Method', y='Time', hue='Tolerance', errorbar=None)
plt.yscale('log')
plt.tight_layout()
plt.savefig('prova1.png')

plt.clf()
sns.barplot(data=df, x='Method', y='Iterations', hue='Tolerance', errorbar=None)
plt.yscale('linear')
plt.tight_layout()
plt.savefig('prova2.png')

plt.clf()
sns.barplot(data=df, x='Method', y='Relative error', hue='Tolerance', errorbar=None)
plt.tight_layout()
plt.savefig('prova3.png')

plt.clf()
sns.barplot(data=df.astype({'Density': float}), x='Density', y='Iterations', hue='Method', errorbar=None)
plt.yscale('log')
plt.xticks(rotation=90)
#plt.gca().set_xticklabels([f'{tick:.4f}' for tick in plt.gca().get_xticks()])
plt.tight_layout()
plt.savefig('prova4.png')

sns.set_theme(style="white")
df_corr = df[['Size', 'Density', 'Tolerance', 'Relative error', 'Time', 'Iterations']]
corr = df_corr.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.savefig('prova5.png')