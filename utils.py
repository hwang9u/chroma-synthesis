import numpy as np
def item2arr(a):
    return np.array([a])

def gaussian(x):
    return np.exp(-.5 * x**2)


    
def slinterp(X, F):
    '''
    simple linear interpolation
    '''
    sx = np.prod(X.shape)
    
    X1 = np.concatenate([ X[1:], item2arr(0)])
    XX = np.zeros((F, sx))
    
    for i in range(F):
        XX[i] = ((F-i)/F) * X + (i/F) * X1
    Y = XX.T.ravel()[: (sx - 1) *F+1 ]
    return Y
    
    
import numpy as np

# 2 ** oct = f / fo

def octs2hz(octs, A440 = 440.):
    '''
    octs: octaves
    '''
    hz = (A440/ (2**4) ) * (2**octs) # 27.5 * 2**octs
    return hz


# hz -> octs, fo(27.5)에 상대적으로 몇 옥타브에 해당하는가?
def hz2octs(freq, A440 = 440.):
    '''
    freq: frequency(hz)
    '''
    return np.log2( freq/ (A440/ (2**4))) #  freq / 27.5

