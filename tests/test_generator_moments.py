"""Analytical conditional distributions check the incomplete-data EM core."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from generative_imputer import moments


class ConditionalMomentsTest(unittest.TestCase):
    def test_known_gaussian_regression(self):
        r,e,v,_=moments(np.array([[2.,np.nan]]),np.array([[0.,1.]]),np.array([[[4.,1.],[1.,9.]]]),np.ones((1,1)),np.array([0]))
        np.testing.assert_allclose(r,[[1.]])
        np.testing.assert_allclose(e,[[[2.,1.5]]])
        np.testing.assert_allclose(v,[[[[0.,0.],[0.,8.75]]]])

    def test_no_observations_returns_prior_and_observed_are_preserved(self):
        x=np.array([[np.nan,np.nan],[3.,4.]])
        m=np.array([[0.,1.],[2.,3.]]);c=np.tile(np.eye(2),(2,1,1));p=np.array([[.25,.75]])
        r,e,v,_=moments(x,m,c,p,np.array([0,0]))
        np.testing.assert_allclose(r[0],p[0]);np.testing.assert_allclose(e[0],m)
        np.testing.assert_allclose(e[1],[[3.,4.],[3.,4.]])
        np.testing.assert_allclose(v[1],0.)


if __name__ == "__main__":
    unittest.main()
