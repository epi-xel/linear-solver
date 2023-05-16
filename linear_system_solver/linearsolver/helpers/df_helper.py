import pandas as pd
import linearsolver.utils.analize as analize

class ResultsStats:

    def __init__(self):
        self.cols = ['Matrix', 'Size', 'Density', 'Method', 'Tolerance', 'Time', 'Iterations', 'Relative error']
        self.tmp_list = []

    def get_stats_df(self):
        return pd.DataFrame(self.tmp_list, columns=self.cols)
    
    def get_stats_list(self):
        return self.tmp_list
    
    def add_stats(self, A, res, name, tol, method):
        size, density = analize.analize_matrix(A)
        self.tmp_list.append([name, size, density, method, tol, res.time, res.iterations, res.relative_error])

    def merge_stats(self, stats):
        self.tmp_list.extend(stats.get_stats_list())
