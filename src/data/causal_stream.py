"""Stateful 50 Hz prefix builder: no labels or future samples required online."""
from collections import deque
import numpy as np
from scipy.signal import butter,sosfilt

class CausalWindowStream:
    def __init__(self,unit='g',window=150,stride=75,ratio=.5,seq_len=5,horizon_samples=50,warmup=500):
        if unit not in ['g','ms2'] or not 0<ratio<=1 or min(window,stride,seq_len,horizon_samples)<1 or warmup<0:
            raise ValueError('Invalid stream configuration')
        self.unit,self.window,self.stride,self.seq_len=unit,window,stride,seq_len
        self.prefix=max(1,int(window*ratio));self.horizon=horizon_samples;self.warmup=warmup
        if self.prefix<2:raise ValueError('At least two observed samples required')
        self.sos=butter(5,.5,fs=50,output='sos');self.reset()
    def reset(self):
        self.zi=np.zeros((len(self.sos),2,3));self.signal=np.empty((0,6),np.float32)
        self.offset=0;self.total=0;self.next_start=self.warmup;self.context=deque(maxlen=self.seq_len)
    def push(self,raw):
        raw=np.asarray(raw,dtype=np.float64)
        if raw.ndim!=2 or raw.shape[1]!=6 or not np.isfinite(raw).all():raise ValueError('Expected finite raw samples x 6')
        if not len(raw):return []
        raw=raw.copy()
        if self.unit=='g':raw[:,:3]*=9.80665
        gravity,self.zi=sosfilt(self.sos,raw[:,:3],axis=0,zi=self.zi);raw[:,:3]-=gravity
        self.signal=np.concatenate([self.signal,raw.astype(np.float32)]);self.total+=len(raw);outputs=[]
        while self.next_start+self.prefix<=self.total:
            i=self.next_start-self.offset
            self.context.append(self.signal[i:i+self.prefix].copy())
            if len(self.context)==self.seq_len:
                target=self.next_start+int(np.ceil((self.window+self.horizon)/self.stride))*self.stride
                end=self.next_start+self.prefix
                outputs.append(dict(context=np.stack(self.context),decision_sample=end,target_start=target,horizon_seconds=(target-end)/50))
            self.next_start+=self.stride
        trim=max(0,min(self.next_start,self.total)-self.offset)
        self.signal=self.signal[trim:];self.offset+=trim
        return outputs
