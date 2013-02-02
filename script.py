### NE 451 Final Project - Gray-Scott Reaction/Diffusion ###

# Submitted by: Derek Bennewies
# Last Edited: November 9, 2012

import main
import utils

# List the (feed, k) parameter values to try
params = [(0.002, 0.031)]

# Loop through all points
for i, v in enumerate(params):

    f, k = v
    print 'Currently running f=%s, k=%s' % (f, k)
    main.main(f, k, feedmax=0.086, kmax=0.073, grid=2048, perturb_mag=0.2, saveims=True, out_freq=250, dpi=300, timesteps=100000)
