#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt


# output file name
output_file = "PKE_Q3_a.out"

# time step
dt = 1.0E-5
time_steps = [1e0, 1e-1,1e-2, 1e-3,1e-4,1e-5 ]
rel_diff = np.loadtxt(output_file)
log_diff = np.log(rel_diff)

# plot rho
fig, ax = plt.subplots()
ax.plot(time_steps, log_diff)
plt.xscale("log")
ax.set(xlabel='time steps', ylabel='relative error',
       title='the second-order convergence for C-N')
ax.grid()
fig.savefig("./figures/PKE_sol_powerConvergence.png")
plt.show()
