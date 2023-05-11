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
    
    general_barplots(df, path)
    density_iterations_barplot(df, path)
    density_time_barplot(df, path)
    heatmap(df, path)


def general_barplots(df, path):

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
    plt.savefig(path + RESULTS_DIR + 'iterations-time-error_barplots.png')


def density_iterations_barplot(df, path):

    sns.set_theme(style="darkgrid")

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
    plt.savefig(path + RESULTS_DIR + 'density-iterations_barplots.png')


def density_time_barplot(df, path):

    sns.set_theme(style="darkgrid")

    methods = df['Method'].unique()

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle('Barplots of the methods execution time for different densities')
    for i, method in enumerate(methods):
        grouped = df.loc[df['Method'] == method]
        axes[i//2, i%2].set_title(method,)
        barplot = sns.barplot(data=grouped, ax=axes[i//2, i%2], x='Density', 
                        y='Time', errorbar=None, palette="viridis", width=0.6)
        barplot.set_yscale('log')
        ticks = barplot.get_xticklabels()
        ticks = [t.get_text()[:6] for t in ticks]
        barplot.set_xticklabels(ticks)
    
    plt.tight_layout()
    plt.savefig(path + RESULTS_DIR + 'density-time_barplots.png')


def heatmap(df, path):

    sns.set_theme(style="white")

    methods = df['Method'].unique()

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
    plt.savefig(path + RESULTS_DIR + 'heatmap.png')