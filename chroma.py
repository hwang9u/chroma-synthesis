import matplotlib.pyplot as plt

import librosa
from librosa.display import specshow
import numpy as np

from utils import gaussian, slinterp
from scales import hz2octs

def fft2chromamx(n_fft = 2048, n_bins = 12, sr = 22050, A440 = 440., ctroct = 5, octwidth = 0):
    wts = np.zeros( (n_bins, n_fft) )

    frq = np.arange(1,n_fft) / n_fft * sr # frq = k /n_fft * sr
    fftfrqbins =  n_bins * hz2octs(frq, A440 = A440) # |fftfrqbins| = ( , n_fft -1) / n_bins * octs 
    fftfrqbins = np.concatenate( [ np.array([fftfrqbins[0]]) -1.5* n_bins, fftfrqbins] ) # |fftfrqbins| = ( , n_fft)    
    binwidthbins = np.append(np.maximum(1., np.diff(fftfrqbins)), 1.)

    D = np.tile(fftfrqbins, (n_bins, 1)) - np.tile(np.arange(n_bins).reshape(-1,1), (1, n_fft)) 
    n_bins2 = round(n_bins /2)
    D = (D + n_bins2 + 10*n_bins ) % n_bins - n_bins2
    
    # gaussian bumps - 2*D to make them narrower
    wts = gaussian(2*D/np.tile(binwidthbins, (n_bins, 1)) )
    wts = wts/np.tile(np.sqrt( (wts**2).sum()), (n_bins, 1))
    
    # remove aliasing columns
    wts[:, n_fft//2 + 2: ] = 0.

    if octwidth > 0:
        wts = wts * np.tile( gaussian((fftfrqbins/n_bins - ctroct)/ octwidth ), (n_bins, 1)  )
    return wts




def chromagram_P(sig, CM = None, sr = 22050, n_fft = 2048, win_length = 1024, hop_length = 512, n_bins = 12, f_ctr= 1000., f_sd = 1, A440 = 440):
    # STFT
    S = librosa.stft(y=sig, win_length=win_length, n_fft=n_fft, hop_length = hop_length, window="hann", center = True)
    # Magnitude
    M = np.abs(S) # (n_fft//2 +1, ntimesteps)
    nf, nt = M.shape 
    
    if CM is None:
        print('gen')
        ctroct = hz2octs(f_ctr, A440=A440) # center frequency -> octave
        CM = fft2chromamx(n_fft = n_fft, n_bins = n_bins, sr=sr, ctroct= ctroct, octwidth= f_sd) # (n_bins, n_fft)
        
    # chop
    CM = CM[:, : nf]

    Mp = (M > M[  np.insert(arr = np.arange(nf-1), obj=0, axis = 0, values = 0) ]) & (M >= M[ np.append( np.arange(1, nf), nf-1 ) ]) # i번째 freq에 대하여 i-1, i+1보다 모두 큰(local maxes) freq만 남김
    Y = CM @ (M* Mp)
    return Y



def chromasynth(C, bp = .5, sr = 22050, n_octs = 7, basefrq = 27.5, bups = 8):
    nchr, nt = C.shape
    
    if not type(bp) in (tuple, list):
        framerate = bups / bp 
        ncols = nt * bups # upsampling 적용된
        CMbu = np.tile(C[..., None], bups).reshape(nchr, -1) # upsampling
    else:
        framerate = 50.
        nbeats = len(bp)
        lastbeat = bp[nbeats] + (bp[nbeats] - bp[nbeats-1])
        ncols = np.round(lastbeat * framerate)
        CMbu = np.zeros( (nchr, ncols) )
        xF = np.concat( [np.zeros(nchr) , C] ,axis=1 )
        for c in range(ncols):
            CMbu[:, c] = xF[:,  np.where(c/framerate >= np.insert(arr= bp, obj = 0, values = 0, axis = 0 ))[0][-1]  ] # c/framerate >= [0,bp]를 만족하는 index 중 가장 큰 index의 값
        
    
    CFbu = np.tile(  basefrq * 2** (np.arange(nchr )/nchr).reshape(-1,1), (1, ncols)) # (nchr, ncols)
    
    f_bins = basefrq * 2** (np.arange(n_octs * nchr )/nchr) # center frequency
    CF = []
    CM = []
    # gaussian weights
    f_ctr = 440.
    f_sd = .5,
    f_dist = np.log2( f_bins / f_ctr) / f_sd
    f_wts = gaussian(f_dist)
    
    for oct in range(1, n_octs+1):
        CF.append((2**oct) * CFbu)
        CM.append( np.diag(f_wts[ (oct-1) * nchr + np.arange(nchr) ]) @ CMbu )
    CF = np.concatenate(CF, axis = 0)
    CM = np.concatenate(CM, axis = 0)
    
    CFok = CF[:, 0] < sr/2
    CF = CF[CFok]
    CM = CM[CFok]
    # x = synthtrax(CF, CM, SR =sr, SUBF= round(sr/framerate), DUR = 0 )
    return CF, CM


def synthtrax(F, M, SR = 22050, SUBF = 128, DUR = 0):
    nf, nt = F.shape
    opsamps = round(DUR * SR)
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
        
        zz = np.arange(np.maximum( 0,firstcol-1), np.minimum(nt, lastcol+1))
        if len(zz) > 0:
            mm = mm[zz]
            ff = ff[zz]
            nzcols = np.prod(zz.shape)
            mz = (mm ==0) # magnitude 값이 0인 지점
            mask = mz & (0 == np.append( mz[1 : nzcols], 0) )
            ff = ff* (1-mask) + mask *  np.insert(arr = ff[:nzcols-1], obj = 0, axis=0, values=0)
            ff = slinterp(ff, SUBF)
            mm = slinterp(mm, SUBF)
            
            # convert frequency to phase values
            pp = np.cumsum(2* np.pi *ff/ SR)
            xx = mm * np.cos(pp)
            base = 1+ SUBF * (zz[0] ) # ???
            sizex = np.prod(xx.shape)
        
            ww = (base - 1) + np.arange(sizex)
            X[ww] = X[ww] + xx
            
    return X


class Cynthesis:
    '''
    Chroma Synthesis
    '''
    def __init__(self, n_fft = 2048,  n_bins = 12, f_ctr = 1000, f_sd = 1, win_length = None, hop_length = None, sr = 22050):
        # stft params
        self.n_fft = n_fft
        self.win_length = win_length or n_fft//2
        self.hop_length = hop_length or n_fft//4
        self.sr = sr
        # chroma params
        self.n_bins = n_bins
        self.f_ctr = f_ctr
        self.f_sd = f_sd
        # mapping matrix
        self.CM = fft2chromamx(n_fft = self.n_fft, n_bins = self.n_bins, sr = self.sr, A440 = 440., ctroct = hz2octs(self.f_ctr), octwidth = self.f_sd)
    
    def __call__(self, sig, normalize = True, n_octs = 7, bups = 8, subf = None, dur = 0):
        self.chroma = chromagram_P(sig = sig, CM = self.CM, sr=self.sr, n_fft = self.n_fft, win_length = self.win_length, hop_length = self.hop_length, n_bins = self.n_bins, f_ctr = self.f_ctr, f_sd = 1, A440 = 440)
        self.bp = self.hop_length / self.sr
        CF, CM = chromasynth(C = self.chroma, bp = self.bp, sr = self.sr, n_octs = n_octs, basefrq=27.5, bups = bups)
        framerate = bups / self.bp
        subf = np.round(self.sr/framerate).astype('int16') or subf    
        cyns  = synthtrax(CF, CM, SR=self.sr, SUBF=subf, DUR = dur )
        
        if normalize:
            cyns *= np.max( np.abs(sig)) / np.max( np.abs(cyns))
        return cyns
