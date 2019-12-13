import numpy as np
import os

def dataset(filename):
    dataarray = np.genfromtxt(os.path.join(os.path.dirname(__file__),'dataset',filename), delimiter=',')
    (nins,ndim)=dataarray.shape
    X=dataarray[:,0:ndim-1]
    Y=dataarray[:,ndim-1:ndim]
    Y=Y.ravel()
    return (X,Y)
