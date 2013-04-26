'''
Created on Apr 23, 2013

@author: ckt

'''

from numpy import dot,diag
from numpy.linalg import eig,norm
def equationSolveBisection(func, lo, hi, tol=1e-6):
    '''
    solve a one-variant equation f(x)=0, by bisection among
    func: f(x)=func(x), is an increasing function 
    lo: lower boundary
    hi: high boundary
    tol: stop when hi-lo<=tol 
    
    return value(s):
    x such that f(x)=0 
    '''
    if hi-lo <=tol:
        return (hi+lo)/2.0
    if func(hi)<0:
        hi,lo=hi+2*(hi-lo),hi
    if func(lo)>0:
        hi,lo=lo,lo-hi+lo
    if func((lo+hi)/2.0)>0:
        hi=(lo+hi)/2.0
    else:
        lo=(lo+hi)/2.0
    return equationSolveBisection(func, lo, hi, tol=1e-6)
    
def trustRegionNewtonMethod(func, grad, hessian, x0, tol=1e-6, maxIter=10000, deltaM=0.1):
    '''
    func,grad,hessian: f(x).gradf(x),hessianf(x)=func(x),grad(x),hessian(x)
    x0: initial value
    tol: stop when abs(f(x_{k+1})-f(x_{k}))<=tol 
    maxIter: maximum iteration number
    deltaM: maximum region radius
    
    return value(s):
    x:  optimized result of x
    vf: function value at x
    info:information
    
    Trust region method similiar as algorithm 4.1@ P69, 
    however, totally different from it in how to solve the local quadratic model
    here I use bisection method to find the parameter kapa(lambda in book).
    '''
    iterNum=1
    delta=deltaM
    while True:
        iterNum=iterNum+1
        vf0,gf0,H=func(x0),grad(x0),hessian(x0)
        (V,D)=eig(H)
        p0=dot(V*diag(map(lambda x:1.0/x, diag(D)))*V.H, gf0)
        if norm(p0)<delta:
            kapa=0
            p=-p0
        else:
            kapa=equationSolveBisection(lambda x: delta-norm(dot(V*diag(map(lambda y:1.0/(x+y), diag(D)))*V.H,gf0)), 0, D[0][0], tol=1e-6)
            p=-dot(V*diag(map(lambda y:1.0/(kapa+y), diag(D)))*V.H,gf0)
        x1=x0+p
        rho=(func(x1)-vf0)/(p.H*H*p+gf0.H*p)
        if rho<0.25:
            delta=delta*0.25
        else:
            if rho>0.75:
                delta=min(2*delta, deltaM)
        if rho>0.1:
            x0=x1
            if abs(func(x0)-func(x1))<=tol:
                x0,func(x0), 'msg: this is more likely to be convergent'
        if iterNum==maxIter:
            return x0,func(x0),'warning: exceeding maximum iteration number'
