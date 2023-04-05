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

