import matplotlib.pyplot as plt

import librosa
from librosa.display import specshow
import numpy as np

from utils import gaussian, item2arr, slinterp
from scales import hz2octs

def fft2chromamx(nfft = 2048, nbins = 12, sr = 22050, A440 = 440., ctroct = 5, octwidth = 0):
    wts = np.zeros( (nbins, nfft) )
    
    frq = np.arange(1,nfft) / nfft * sr # frq = k /nfft * sr
    fftfrqbins =  nbins * hz2octs(frq, A440 = A440) # |fftfrqbins| = ( , nfft -1) / nbins * octs 
    
    # 0 Hz에 해당하는 bin은 2번쨰 bin의 1.5 octave 아래로 설정 (2번째 bin의 50% 아래에 위치해 있음)
    fftfrqbins = np.concatenate( [ np.array([fftfrqbins[0]]) -1.5* nbins, fftfrqbins] ) # |fftfrqbins| = ( , nfft)
    
    binwidthbins = np.concatenate( [np.maximum(1., np.diff(fftfrqbins)), item2arr(1.)])
    
    D = np.tile(fftfrqbins, (nbins, 1)) - np.tile(np.arange(nbins).reshape(-1,1), (1, nfft))
    
    nbins2 = round(nbins /2)
    
    D = (D + nbins2 + 10*nbins ) % nbins - nbins2
    
    # gaussian bumps - 2*D to make them narrower
    wts = gaussian(2*D/np.tile(binwidthbins, (nbins, 1)) )
    wts = wts/np.tile(np.sqrt( (wts**2).sum()), (nbins, 1))
    
    # remove aliasing columns
    wts[:, nfft//2 + 2: ] = 0.
    
    
    if octwidth > 0:
        wts = wts * np.tile( gaussian((fftfrqbins/nbins - ctroct)/ octwidth ), (nbins, 1)  )
    return wts




def chromagram_P(sig, sr = 22050, nfft = 2048, win_length = 1024, hop_length = 512, nbins = 12, f_ctr= 1000., f_sd = 1):
    # STFT
    S = librosa.stft(y=sig, win_length=win_length, n_fft=nfft, hop_length = hop_length, window="hann", center = True)
    # to magnitude
    M = np.abs(S) # (nfft//2 +1, ntimesteps)
    nf, nt = M.shape
    
    ctroct = np.log2(f_ctr / 27.5) # f_ctr_log = f_ctr's octave
    
    CM = fft2chromamx(nfft = nfft, nbins = nbins, sr=sr, ctroct= ctroct, octwidth= f_sd) # (nbins, nfft)
    # chop
    CM = CM[:, : nf]
     
    Mp = (M > M[ np.concatenate( [item2arr(0), np.arange(nf-1) ] ) ]) & (M >= M[ np.concatenate( [np.arange(1, nf) , item2arr(nf-1)] ) ])
    Y = CM @ (M* Mp)
    return Y



def chromasynth(C, bp = .5, sr = 22050, nocts = 7, basefrq = 27.5, f_ctr = 440., f_sd = .5):
    nchr, nt = C.shape
    if not type(bp) in (tuple, list):
        bups = 1 # upsampling factor
        framerate = bups / bp  # 1/
        ncols = nt * bups
        CMbu = C # (nchr, ncols)
    CFbu = np.tile(  basefrq * 2** (np.arange(nchr )/nchr).reshape(-1,1), (1, ncols)) # (nchr, ncols)
    
    f_bins = basefrq * 2** (np.arange(nocts * nchr )/nchr) # center frequency
    
    CF = []
    CM = []
    # gaussian weights
    f_dist = np.log2( f_bins / f_ctr) / f_sd
    f_wts = gaussian(f_dist)
    
    for oct in range(1, nocts+1):
        CF.append((2**oct) * CFbu)
        CM.append( np.diag(f_wts[ (oct-1) * nchr + np.arange(nchr) ]) @ CMbu )
    CF = np.concatenate(CF, axis = 0)
    CM = np.concatenate(CM, axis = 0)
    
    # print(CFok)
    CFok = CF[:, 0] < sr/2
    CF = CF[CFok]
    CM = CM[CFok]
    # x = synthetrax(CF, CM, SR , SUBFa= round(sr/framerate), DUR = 0 )
    return CF, CM


def synthtrax(F, M, SR = 22050, SUBF = 128, DUR = 0):
    nf, nt = F.shape
    opsamps = round(DUR * SR)
    print(opsamps / SR)
    if DUR == 0:
        opsamps = 1 + ((nt - 1) *SUBF)
    X = np.zeros(opsamps)
    for f in range(nf):
        ff = F[f]
        mm = M[f]
        mm[np.isnan(mm)] = 0
        ff[np.isnan(ff)] = 0
        
        nzv = mm.nonzero()[0]
        firstcol = nzv.min()
        lastcol = nzv.max()
        
        zz = np.arange(np.maximum( 0,firstcol-1), np.minimum(nt, lastcol+1)) ## ???
        if len(zz) > 0:
            mm = mm[zz]
            ff = ff[zz]
            nzcols = np.prod(zz.shape)
            mz = (mm ==0) # magnitude 값이 0인 지점
            mask = mz & (0 == np.concatenate([ mz[1:nzcols], item2arr(0) ]) )
            ff = ff* (1-mask) + mask * np.concatenate([ item2arr(0), ff[:nzcols-1], ] )
            ff = slinterp(ff, SUBF)
            mm = slinterp(mm, SUBF)
            
            # convert frequency to phase values
            pp = np.cumsum(2* np.pi *ff/ SR)
            
            ## cosine 함수
            xx = mm * np.cos(pp)
            base = 1+ SUBF * (zz[0] ) # ???
            # print(base)
            sizex = np.prod(xx.shape)
        
            ww = (base - 1) + np.arange(sizex)
            X[ww] = X[ww] + xx
            
    return X

        







