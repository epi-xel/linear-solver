import linearsolver.methods.output_colors as c
import numpy as np

# Print the stats of the solution
def print_stats(res, x_true, method, last = False):

    print(c.bcolors.BOLD
          + "Stats for "
          + c.bcolors.OKBLUE
          + method 
          + c.bcolors.ENDC
          + c.bcolors.BOLD
          + " method" 
          + c.bcolors.ENDC)
    print(c.bcolors.OKGREEN 
          + "> Relative error:  " 
          + c.bcolors.ENDC 
          + str(rel_error(res['solution'], x_true)))
    print(c.bcolors.OKGREEN 
          + "> Elapsed time:    " 
          + c.bcolors.ENDC 
          + str(res['time']) + " sec")
    print(c.bcolors.OKGREEN
          + "> Iterations:      " 
          + c.bcolors.ENDC 
          + str(res['iterations']))
    if not last:
        print("------------------------------------------")


# Compute relative error
def rel_error(x, x_true):
    return np.linalg.norm(x - x_true) / np.linalg.norm(x_true)