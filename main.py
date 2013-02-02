### NE 451 Final Project - Gray-Scott Reaction/Diffusion ###

# Submitted by: Derek Bennewies
# Last Edited: November 29, 2012

import numpy as np
import random
import math
import os
import sys
import datetime
import shutil
import matplotlib.pyplot as plt
import utils
import time
import functools as ft


# Script entry point
def main(feed, k, feedmax=0, kmax=0, h=0.01, grid=256, perturb_num=1, perturb_u=0.5,
    perturb_v=0.25, perturb_mag=0.05, dt=0.5, timesteps=50000, out_freq=100, du=0.00002,
    dv=0.00001, framerate=20, laplacian=utils.lap5, visualize=False,
    integrate=utils.exp_euler, dpi=None, saveims=False):

    # Change to working directory for images and videos
    imgdir = str(datetime.datetime.today())
    os.mkdir(imgdir)
    os.chdir(imgdir)

    # Initiailize grid
    [grid_u, grid_v] = initialize(grid, perturb_num, perturb_u, perturb_v, perturb_mag)

    # Run simulation
    simulate(h, du, dv, grid, grid_u, grid_v, dt, timesteps, k, feed, out_freq, laplacian, visualize, integrate, feedmax, kmax, dpi)

    # Create movie
    namestring = make_movie(framerate, k, feed)

    # Return to original directory and delete working directory if required
    if saveims == False:
        shutil.move('%s.mp4' % namestring, os.pardir)
        shutil.rmtree(imgdir)
    os.chdir(os.pardir)


# Initialize system
def initialize(grid, perturb_num, perturb_u, perturb_v, perturb_mag):

    # Initiailize the grid with all U
    grid_u = np.ones((grid, grid))
    grid_v = np.zeros((grid, grid))

    # Apply the specified number of randomly-sized perturbations
    for i in range(0, perturb_num):

        x_start = int(math.floor(random.random() * grid * 0.9))
        y_start = int(math.floor(random.random() * grid * 0.9))
        x_end = int(math.floor(x_start + (random.random() * grid + grid) * perturb_mag))
        y_end = int(math.floor(y_start + (random.random() * grid + grid) * perturb_mag))

        # Apply perturbations
        for x in range(x_start, x_end):
            for y in range(y_start, y_end):

                # Constant perturbation
                grid_u[y, x] = perturb_u
                grid_v[y, x] = perturb_v

                # Random perturbation
                perturb = 0.01 * random.random()
                grid_u[y, x] -= perturb
                grid_v[y, x] += perturb

    # Return grids
    return (grid_u, grid_v)


# Run simulation on initialized system
def simulate(h, du, dv, grid, grid_u, grid_v, dt, timesteps, k, feed, out_freq, laplacian, visualize, integrate, feedmax, kmax, dpi):

    # If requested, initiailize live visualization
    if visualize == True:
        plt.ion()

    # Initiailize figure for movie frames
    fig, ax = plt.subplots()
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    im = ax.imshow(grid_u, interpolation='nearest')
    fig.colorbar(im)

    # Initiailize time
    start_time = time.time()

    # Calculate h^2 (for speed)
    h2 = h * h

    # If requested, reconfigure feed and k as linspace matrices
    if feedmax != 0 or kmax != 0:
        feedlin = np.linspace(feedmax, feed, num=grid)
        klin = np.linspace(k, kmax, num=grid)
        k = np.tile(klin, (grid, 1))
        feed = np.tile(feedlin[:, np.newaxis], (1, grid))

    # Time loop
    for t in range(1, timesteps):

        # Calculate Laplacians
        lap_u = laplacian(grid_u, h2)
        lap_v = laplacian(grid_v, h2)

        # Calculate central term in integration calculations (for speed)
        grid_uvv = grid_u * grid_v * grid_v

        # Calculate new U and V values using selected integration method
        u_fun = ft.partial(utils.u_fun, du=du, lap_u=lap_u, grid_uvv=grid_uvv, feed=feed)
        v_fun = ft.partial(utils.v_fun, dv=dv, lap_v=lap_v, grid_uvv=grid_uvv, feed=feed, k=k)
        grid_u += integrate(grid_u, dt, u_fun)
        grid_v += integrate(grid_v, dt, v_fun)

        # If it's the right timestep, produce output
        if t % out_freq == 0:

            # Generate the plot
            im.set_data(grid_u)
            plt.title('Timestep #%s' % t)

            # Save the figure as a png
            fig.savefig('img%06d.png' % (t / out_freq), dpi=dpi)

            # If requested, visualize the output as it's generated
            if visualize == True:
                fig.canvas.draw()

            # Calculate time remaining
            rem_time = (timesteps - t) * ((time.time() - start_time) / t)
            rem_hours = math.floor(rem_time / 3600)
            rem_mins = math.floor((rem_time % 3600) / 60)
            rem_secs = math.floor(rem_time % 60)

            # Write current progress to console
            sys.stdout.write('\r Timestep %s of %s is done, %02d:%02d:%02d remaining' % (t, timesteps, rem_hours, rem_mins, rem_secs))
            sys.stdout.flush()

    # Newline for readability
    print '\n'


# Function to generate the movie out of output images
def make_movie(framerate, k, feed):

    # The name of the movie file
    namestring = 'GSD-k=%s-feed=%s' % (k, feed)

    # Compile the movie from the images
    moviestring = 'cat *.png | ffmpeg -f image2pipe -r %s -vcodec png -i - -vcodec libx264 %s.mp4 > rubbish 2>&1' \
        % (framerate, namestring)
    os.system(moviestring)

    # Pass the filename back to main to move to original directory
    return namestring
