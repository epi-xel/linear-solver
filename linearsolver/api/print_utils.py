import numpy as np

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Print the stats of the solution
def print_stats(res, x_true, method, last = False):

    print(bcolors.BOLD
          + "Stats for "
          + bcolors.OKBLUE
          + method 
          + bcolors.ENDC
          + bcolors.BOLD
          + " method" 
          + bcolors.ENDC)
    print(bcolors.OKGREEN 
          + "> Relative error:  " 
          + bcolors.ENDC 
          + str(rel_error(res['solution'], x_true)))
    print(bcolors.OKGREEN 
          + "> Elapsed time:    " 
          + bcolors.ENDC 
          + str(res['time']) + " sec")
    print(bcolors.OKGREEN
          + "> Iterations:      " 
          + bcolors.ENDC 
          + str(res['iterations']))
    if not last:
        print("------------------------------------------")


# Compute relative error
def rel_error(x, x_true):
    return np.linalg.norm(x - x_true) / np.linalg.norm(x_true)