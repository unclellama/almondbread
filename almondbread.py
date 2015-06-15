"""
mandel - calculate and plot the (bounded) mandelbrot set

plan for initial, naive implementation:

- create a mesh representing the initial value c in the complex plane

- for each point in the mesh, check whether the point leaves the set
  over some maximum number of iterations

- record the number of iterations in the mesh

- produce plot

testing strategy:

- ensure the mesh creation works

- unittest for points known to be inside the set (so, points in the
  lobe)

- unittest for points known to be outside

- test the plotting by inspection!

"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time

def mandelcheck(threshold,c):
    zn=0
    for i in range(threshold):
        if abs(zn) > 2 or i == threshold-1:
            return i
        else:
            zn=zn*zn+c

def mandelcheck_vector(threshold,carr):
    zn=np.zeros_like(carr,dtype=np.complex_)
    iters=np.zeros_like(carr,dtype=int)
    iters[:,:]=threshold
    for i in range(threshold):
        abstest=np.absolute(zn)
        iters[np.where(abstest > 2)] = i
        carr[np.where(abstest > 2)] = 0+0*1j
        zn = np.square(zn)+carr
    return iters
        
def setup_mesh(xsize,ysize,xrange,yrange):
    """
    produces a 2d np.array of values and another of coordinates, given
    the specified number of pixels and xrange, yrange.
    calling seq:
    xcoords,ycoords,mesh=setup_mesh(5,5,[-2,1],[3,4])
    """
    delx=(xrange[1]-xrange[0])/float(xsize)
    dely=(yrange[1]-yrange[0])/float(ysize)
    xcoords=np.arange(xrange[0],xrange[1],delx)
    ycoords=np.arange(yrange[0],yrange[1],dely)
    mesh=np.zeros((xsize,ysize),dtype=np.int)
    return (xcoords,ycoords,mesh)

def setup_carr(reco,imco):
    """
    produces a 2d np.array of complex numbers with real parts reco and
    imaginary parts imco
    inputs: equal-size arrays containing the real part and the complex part.
    """
    reals,imaginaries = np.meshgrid(reco, imco)
    return reals+1j*imaginaries

def show_image(im,imextent,threshold):
    fig, ax = plt.subplots()
    cax = ax.imshow(im, cmap=mpl.cm.hot,extent=imextent)
    ax.set_title("Mandelbrot set, threshold: "+str(threshold))
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    cbar = fig.colorbar(cax, ticks=[-1, 0, 1])
    cbar.ax.set_yticklabels(['< -1', '0', '> 1'])
                                        
def naive_almondbread(nreal,nimag,threshold=50,rerange=[-2,1],imrange=[-1.5,1.5]):
    """basic implementation: just loops over each pixel"""

    reco,imco,iters=setup_mesh(nreal,nimag,rerange,imrange)

    for n_re in range(nreal):
        for n_im in range(nimag):
            c=reco[n_re]+imco[n_im]*1j
            iters[n_im,n_re]=mandelcheck(threshold,c)
            
    imextent=rerange+imrange
    show_image(iters,imextent,threshold)
    plt.savefig("mandel_naive.pdf")

def vector_almondbread(nreal,nimag,threshold=50,rerange=[-2,1],imrange=[-1.5,1.5]):
    """vectorized implementation using numpy"""

    reco,imco,iters=setup_mesh(nreal,nimag,rerange,imrange)
    carr=setup_carr(reco,imco)
    iters=mandelcheck_vector(threshold,carr)
    
    imextent=rerange+imrange
    show_image(iters,imextent,threshold)
    plt.savefig("mandel_vector.pdf")    
    
if __name__=="__main__":
    nreal=500
    nimag=500
    
    t1 = time.time()
    naive_almondbread(500,500)
    print("naive implementation: {} mandelbrot prospects calculated in {}s".format(nreal*nimag, time.time() - t1))

    t2 = time.time()
    vector_almondbread(500,500)
    print("numpy implementation: {} mandelbrot prospects calculated in {}s".format(nreal*nimag, time.time() - t2))
