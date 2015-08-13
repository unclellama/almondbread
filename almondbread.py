"""
this script contains functions that can calculate an approximation to
the Mandelbrot set in various ways. the output should be identical
(for given resolutions and zoom windows), but the processing speed
varies depending on which implementation of the algorithm is used.

running it directly:
>> python almondbread.py
will run all implementations and time each one of them.

unittests checking basic functionality for most of the functions can
be found in test_almondbread.py .

the line profiler can be run on the time-intensive functions by
uncommenting the @profile lines and calling kernprof from the command
line.

"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool
from multiprocessing import cpu_count
from functools import partial

def mandelcheck(threshold,c):
    zn=0
    for i in range(threshold):
        if abs(zn) > 2 or i == threshold-1:
            return i
        else:
            zn=zn*zn+c

#@profile
def mandelcheck_vector(carr,threshold=50):
    zn=np.zeros_like(carr,dtype=np.complex_)
    iters=np.zeros_like(carr,dtype=int)
    iters[:,:]=threshold
    for i in range(threshold):
        abstest=np.absolute(zn)
        iters[np.where(abstest > 2)] = i
        carr[np.where(abstest > 2)] = 0+0*1j
        zn = np.square(zn)+carr
    return iters

#@profile
def setup_complex_arr(reco,imco):
    """
    produces a 2d np.array of complex numbers with real parts reco and
    imaginary parts imco
    inputs: equal-size arrays containing the real part and the complex part.
    """
    reals,imaginaries = np.meshgrid(reco, imco)
    return reals+1j*imaginaries

#@profile
def mandelcheck_optimized(carr,threshold=50):
    """this is an attempt to further optimize mandelcheck_vector
    while keeping it in a numpy framework. the original function
    performs a lot of unnecessary calculations by continuing to
    work with array elements that have already reached |z_n|>2.
    this new code removes already-registered array elements
    from the woring array. the code is a little more complicated
    because i now have to keep track of the coordinates of the
    remaining array elements!
    """
    zn=np.zeros_like(carr,dtype=np.complex_) # working array
    iters=np.zeros_like(carr,dtype=int) # output image
    iters[:,:]=threshold
    # make an array containing the x coord of every entry in
    # carr, and another with the y coord. for indexing iters.
    nx,ny=np.shape(carr)
    xcoords,ycoords = np.mgrid[0:nx, 0:ny]
    
    # flatten all the arrays in the same way to preserve x,y
    carr.shape=nx*ny
    zn.shape=nx*ny
    xcoords.shape=nx*ny
    ycoords.shape=nx*ny

    for i in range(threshold):
        if carr.size==0: break # if all have diverged.
        # here the diverged points are saved to the image
        diverged = abs(zn) > 2.0
        iters[xcoords[diverged],ycoords[diverged]]=i
        # here the diverged points are pruned from the arrays.
        remaining=-diverged
        zn=zn[remaining]
        carr=carr[remaining]
        xcoords=xcoords[remaining]
        ycoords=ycoords[remaining]
        # finally the mandelbrot rule is applied
        zn = np.square(zn)+carr
        
    return iters

def show_image(im,imextent,titlestring):
    """
    plots a 2D image with colorcoded intensity (orange-white!)
    """
    fig, ax = plt.subplots()
    cax = ax.imshow(im, cmap=mpl.cm.hot,extent=imextent)
    ax.set_title = titlestring
    # Add colorbar
    cbar = fig.colorbar(cax, ticks=[-1, 0, 1])
    cbar.ax.set_yticklabels(['< -1', '0', '> 1'])

#@profile
def almondbread(nreal,nimag,threshold=50,rerange=[-2,1],imrange=[-1.5,1.5],imp="naive",nproc=4):
    """
    this part of the code sets up the complex array C, calls one of the implementations
    of the Mandelbrot algorithm, and calls the plotting code when the algorithm has reached
    N_max.
    """

    reco=np.linspace(rerange[0],rerange[1],nreal)
    imco=np.linspace(imrange[0],imrange[1],nimag)
    carr=setup_complex_arr(reco,imco)

    if imp == "naive":
        iters=np.zeros_like(carr,dtype=int)       
        for n_re in range(nreal):
            for n_im in range(nimag):
                iters[n_im,n_re]=mandelcheck(threshold,carr[n_im,n_re])
        filename="mandel_naive.pdf"
        
    elif imp == "numpy":
        iters=mandelcheck_vector(carr,threshold=threshold)
        filename="mandel_vector.pdf"

    elif imp == "numpy_optimized":
        iters=mandelcheck_optimized(carr,threshold=threshold)
        filename="mandel_optimized.pdf"
        
    elif imp == "multiprocessing":
        mandelcheck_multi=partial(mandelcheck_optimized,threshold=threshold)
        csplit=np.array_split(carr,nproc)
        itersplit=pool.map(mandelcheck_multi,csplit)
        iters=np.concatenate(itersplit)
        filename="mandel_multi.pdf"

    else:
        print("That imp= keyword not supported!")
        print("...the following implementations are available: imp='naive', imp='numpy', imp='numpy_optimized'.")
            
    imextent=rerange+imrange
    titlestring = "Mandelbrot set, threshold: "+str(threshold)
    show_image(iters,imextent,titlestring)
    plt.savefig(filename)
    
if __name__=="__main__":
    nreal=5000
    nimag=5000
    nproc=8
    pool = Pool(processes=nproc)

    print("\n\nTiming the naive (loop-within-loop) implementation of the Mandelbrot algorithm\n\n")
    
    t1 = time.time()
    almondbread(nreal,nimag,imp="naive")
    print("naive implementation: {} mandelbrot prospects calculated in\
        {}s".format(nreal*nimag, time.time() - t1))

    print("\n\nTiming the NumPy vectorized implementation of the Mandelbrot algorithm\n\n")

    t2 = time.time()
    almondbread(nreal,nimag,imp="numpy")
    print("numpy implementation: {} mandelbrot prospects calculated in\
          {}s".format(nreal*nimag, time.time() - t2))

    print("\n\nTiming the optimized NumPy implementation of the Mandelbrot algorithm\n\n")

    t3 = time.time()
    almondbread(nreal,nimag,imp="numpy_optimized")
    print("numpy implementation with additional optimization: {} pix in\
          {}s".format(nreal*nimag, time.time() - t3))

    print("\n\nTiming the 2-thread multiprocessing implementation of the Mandelbrot algorithm\n\n")

    nproc=2
    pool = Pool(processes=nproc)
    t6 = time.time()
    almondbread(nreal,nimag,imp="multiprocessing",nproc=nproc)
    print("multiprocess (2 core) optimized numpy implementation: {} pix in\
     {}s".format(nreal*nimag, time.time() - t6))

    print("\n\nTiming the 4-thread multiprocessing implementation of the Mandelbrot algorithm\n\n")

    nproc=4
    pool = Pool(processes=nproc)
    t4 = time.time()
    almondbread(nreal,nimag,imp="multiprocessing",nproc=nproc)
    print("multiprocess (4 core) optimized numpy implementation: {} pix in \
    {}s".format(nreal*nimag, time.time() - t4))

    print("\n\nTiming the 8-thread multiprocessing implementation of the Mandelbrot algorithm\n\n")

    nproc=8
    pool = Pool(processes=nproc)
    t5 = time.time()
    almondbread(nreal,nimag,imp="multiprocessing",nproc=nproc)
    print("multiprocess (8 core) optimized numpy implementation: {} pix in\
     {}s".format(nreal*nimag, time.time() - t5))
