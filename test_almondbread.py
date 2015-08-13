import unittest
import almondbread as m
import numpy as np

class TestMandel(unittest.TestCase):
    def setUp(self):
        pass

    def test_carr(self):
        """
        test that the complex-number grid is working as intended
        """
        rerange=[0.,10.]
        imrange=rerange
        nimag=11
        nreal=nimag
        reco=np.linspace(rerange[0],rerange[1],nreal)
        imco=np.linspace(imrange[0],imrange[1],nimag)
        carr=m.setup_complex_arr(reco,imco)
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
        c=-2+0.2*1j
        self.assertFalse(m.mandelcheck(threshold,c)==threshold-1)

    def test_mandelcheck_vector(self):
        """ test that points in the mandelbrot set don't escape the
        mandelcheck vectorized function
        """
        threshold=1000
        rerange=[0.,0.1]
        imrange=rerange
        nimag=100
        nreal=nimag
        reco=np.linspace(rerange[0],rerange[1],nreal)
        imco=np.linspace(imrange[0],imrange[1],nimag)
        carr=m.setup_complex_arr(reco,imco)
        iters=m.mandelcheck_vector(threshold,carr)
        self.assertTrue(np.amin(iters)==threshold)
        rerange=[0.,5]
        imrange=rerange
        reco=np.linspace(rerange[0],rerange[1],nreal)
        imco=np.linspace(imrange[0],imrange[1],nimag)
        carr=m.setup_complex_arr(reco,imco)
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
