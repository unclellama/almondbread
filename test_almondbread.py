import unittest
import almondbread as m
import numpy as np

class TestMandel(unittest.TestCase):
    def setUp(self):
        pass
        
    def test_mesh(self):
        """
        test that the mesh code returns the correct coordinates
        """
        xcoords,ycoords,mesh=m.setup_mesh(10,10,[0.,5.],[0.,5.])
        mycoords=np.array([0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5])
        self.assertTrue(np.allclose(xcoords,mycoords))
        self.assertTrue(np.allclose(ycoords,mycoords))

    def test_carr(self):
        """
        test that the complex-number grid is working as intended
        """
        xcoords,ycoords,mesh=m.setup_mesh(10,10,[0.,10.],[0.,10.])
        carr=m.setup_carr(xcoords,ycoords)
        self.assertTrue(carr[7,9]==9+1j*7)       

    def test_mandelcheck(self):
        """ test that points in the mandelbrot set don't escape the
        mandelcheck function
        """
        threshold=1000
        c=0+0*1j
        self.assertTrue(m.mandelcheck(threshold,c)==threshold-1)
        c=-0.5+0.2*1j
        self.assertTrue(m.mandelcheck(threshold,c)==threshold-1)

    def test_mandelcheck_vector(self):
        """ test that points in the mandelbrot set don't escape the
        mandelcheck vectorized function
        """
        threshold=1000
        xcoords,ycoords,mesh=m.setup_mesh(10,10,[0.,0.1],[0.,0.1])
        carr=m.setup_carr(xcoords,ycoords)
        iters=m.mandelcheck_vector(threshold,carr)
        self.assertTrue(np.amin(iters)==threshold)
        xcoords,ycoords,mesh=m.setup_mesh(10,10,[-5.0,5.0],[-5.0,5.0])
        carr=m.setup_carr(xcoords,ycoords)
        iters=m.mandelcheck_vector(threshold,carr)
        self.assertFalse(np.amin(iters)==threshold)
        pass
        
    #def test_is_not_prime(self):
        #is_prime = check_prime.check_prime(4)
        #self.assertFalse(is_prime)
    #    pass
    
    #def test_small(self):
        #self.assertRaises(ValueError,check_prime.check_prime,1)
        #self.assertRaises(ValueError,check_prime.check_prime,-1)
     #   pass
