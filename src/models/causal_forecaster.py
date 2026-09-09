"""Independent causal activity forecaster; does not modify the HAR recognizer."""
import numpy as np
import torch
from torch import nn


def window_features(x):
    """36 deterministic features of observed samples only. (..., time, 6)."""
    x=np.asarray(x,dtype=np.float32)
    if x.ndim<3 or x.shape[-1]!=6 or x.shape[-2]<2 or not np.isfinite(x).all():
        raise ValueError('Expected finite observed windows (..., time>=2, 6)')
    return np.concatenate([x.mean(-2),x.std(-2),x.min(-2),x.max(-2),
        np.sqrt(np.mean(x*x,axis=-2)),np.std(np.diff(x,axis=-2),axis=-2)],axis=-1)


class CausalForecaster(nn.Module):
    """GRU context plus a direct last-window branch; consumes past features only."""
    def __init__(self,n_classes=12,input_dim=36,hidden=96):
        super().__init__()
        self.input_dim,self.hidden,self.n_classes=input_dim,hidden,n_classes
        self.project=nn.Sequential(nn.Linear(input_dim,hidden),nn.LayerNorm(hidden),nn.GELU())
        self.context=nn.GRU(hidden,hidden,batch_first=True)
        self.last=nn.Linear(hidden,n_classes)
        self.temporal=nn.Sequential(nn.LayerNorm(hidden),nn.Dropout(.1),nn.Linear(hidden,n_classes))
        self.gate=nn.Linear(2*hidden,1)

    def forward(self,x):
        if x.ndim!=3 or x.shape[-1]!=self.input_dim or x.shape[1]<1:
            raise ValueError('Expected batch, context, features')
        z=self.project(x);h,_=self.context(z)
        gate=torch.sigmoid(self.gate(torch.cat([z[:,-1],h[:,-1]],dim=-1)))
        return gate*self.last(z[:,-1])+(1-gate)*self.temporal(h[:,-1])
