import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from linearsolver.utils.constants import RESULTS_DIR    


def analize_matrix(A):
    shape = A.tocsr().get_shape()
    nonzero = A.tocsr().count_nonzero()
    size = shape[0] * shape[1] 
    density = nonzero / size
    return size, density


def build_result_df(A, res, name, tol, method):
    size, density = analize_matrix(A)
    data = {'Matrix': [name],
            'Size': [size],
            'Density': [density],
            'Tolerance': [tol],
            'Method': [method],
            'Relative error': [res.relative_error],
            'Time': [res.time],
            'Iterations': [res.iterations]
            }
    
    df = pd.DataFrame(data)
    return df


def export_results(df, path):

    print("\n" + "Exporting results...")

    output_file = 'summary.csv'
    output_dir = Path(path + RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_dir / output_file)

    compare_results(df, path)

    print("\n" + "Results exported!" + "\n")
    


def init_ls_df():
    return pd.DataFrame(columns=["Matrix", "Size", "Density", "Tolerance", "Method", "Relative error", "Time", "Iterations"])


def compare_results(df, path):
    
    sns.set_theme(style="darkgrid")

    sns.barplot(data=df, x='Method', y='Time', hue='Tolerance', errorbar=None, palette="crest")
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(path + RESULTS_DIR + 'barplot_method-time.png')

    plt.clf()
    sns.barplot(data=df, x='Method', y='Iterations', hue='Tolerance', errorbar=None, palette="crest")
    plt.yscale('linear')
    plt.tight_layout()
    plt.savefig(path + RESULTS_DIR + 'barplot_method-iterations.png')

    plt.clf()
    sns.barplot(data=df, x='Method', y='Relative error', hue='Tolerance', errorbar=None, palette="crest")
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(path + RESULTS_DIR + 'barplot_method-relative_error.png')

    plt.clf()
    sns.barplot(data=df.astype({'Density': float}), x='Density', y='Iterations', hue='Method', errorbar=None, palette="crest")
    plt.yscale('log')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(path + RESULTS_DIR + 'barplot_density-iterations.png')

    sns.set_theme(style="white")
    df_corr = df[['Size', 'Density', 'Tolerance', 'Relative error', 'Time', 'Iterations']]
    corr = df_corr.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), 1)
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.savefig(path + RESULTS_DIR + 'heatmap.png')