"""Distributional anti-collapse regularizers for SAC+JEPA latents [B, D]."""
from functools import lru_cache
import numpy as np
import torch
from torch.nn import functional as F

def visreg(z, slices=64, eps=1e-4):
    z=z.float(); mu=z.mean(0); x=z-mu; std=(x.var(0,unbiased=False)+eps).sqrt()
    dirs=F.normalize(torch.randn(z.shape[1],slices,device=z.device,dtype=z.dtype),dim=0)
    p=(x/(std.detach()+eps)@dirs).sort(0).values
    q=torch.sqrt(torch.tensor(2.,device=z.device))*torch.erfinv(2*(torch.arange(len(z),device=z.device,dtype=z.dtype)+.5)/len(z)-1)
    return mu.square().mean()+(1-std).square().mean()+(p-q[:,None]).square().mean()

def sigreg(z, slices=64):
    """Sliced Gaussian regularization used as the SIGReg ablation.

    Random one-dimensional projections of the unnormalised JEPA latent are
    matched to exact standard-normal quantiles with a sliced W2 objective.
    Unlike VISReg, this deliberately does not detach per-coordinate scale:
    SIGReg controls the latent distribution directly.
    """
    z = z.float(); b, d = z.shape
    directions = F.normalize(torch.randn(d, slices, device=z.device, dtype=z.dtype), dim=0)
    projections = (z @ directions).sort(dim=0).values
    probabilities = (torch.arange(b, device=z.device, dtype=z.dtype) + .5) / b
    gaussian_quantiles = torch.sqrt(torch.tensor(2., device=z.device, dtype=z.dtype)) * torch.erfinv(2 * probabilities - 1)
    return (projections - gaussian_quantiles[:, None]).square().mean()

def kerjepa(z, gamma=None):
    z=z.float(); b,d=z.shape; gamma=1/d if gamma is None else gamma; k=torch.exp(-gamma*torch.cdist(z,z).square())
    xx=(k.sum()-k.diagonal().sum())/(b*(b-1)); den=1+2*gamma
    xy=(den**(-d/2)*torch.exp(-gamma/den*z.square().sum(1))).mean(); yy=(1+4*gamma)**(-d/2)
    return xx-2*xy+yy

def susreg(z, slices=64, gamma=None):
    z=F.normalize(z.float(),dim=1); b,d=z.shape; a=F.normalize(torch.randn(d,slices,device=z.device),dim=0); x=z@a
    y=F.normalize(torch.randn(b,slices,d,device=z.device),dim=-1)[...,0]; gamma=10*d if gamma is None else gamma
    def k(a,b): return torch.exp(-gamma*(a.T[:,:,None]-b.T[:,None,:]).square()).mean((1,2))
    return (k(x,x)+k(y,y)-2*k(x,y)).mean()

@lru_cache(maxsize=16)
def _quad(d,n):
    x,w=np.polynomial.legendre.leggauss(n); lw=np.log(w)+(d-3)/2*np.log1p(-x*x); lw-=lw.max(); w=np.exp(lw); return x.astype('float32'),(w/w.sum()).astype('float32')

def spherical_mmd(z, quadrature=32, eps=1e-6):
    z=F.normalize(z.float(),dim=1,eps=eps); d=z.shape[1]; x,w=_quad(d,quadrature); x=torch.as_tensor(x,device=z.device); w=torch.as_tensor(w,device=z.device)
    def phi(c): return (torch.exp(-(1-c[...,None])*x.square())*w).sum(-1)
    bias=(phi(x)*w).sum(); return (phi(z@z.T).mean()-bias)/(1-bias).clamp_min(eps)

def regularizer(name,z,variance_target=.1,slices=64,gamma=None,quadrature=32):
    if name=='variance': return F.relu(variance_target-(z.var(0,unbiased=False)+1e-4).sqrt()).mean()
    if name=='sigreg': return sigreg(z,slices)
    if name=='visreg': return visreg(z,slices)
    if name=='kerjepa': return kerjepa(z,gamma)
    if name=='susreg': return susreg(z,slices,gamma)
    if name=='spherical_mmd': return spherical_mmd(z,quadrature)
    if name=='none': return z.new_zeros(())
    raise ValueError(f'unknown jepa_regularizer {name!r}')
