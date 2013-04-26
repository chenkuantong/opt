'''
Created on Apr 23, 2013

@author: ckt
'''
import numpy as np
from numpy.linalg import eig
from numpy import diag,dot
def backTrack(func, x0, p, grad, alpha0=1.0,c=0.1,rho=0.75):
    '''
    func:a function as f(x)=func(x), give the value at x 
    x0:the value of x to find the step length
    p: the direction of line search
    grad: grad vector of func at x0
    
    return value(s):
    alpha: the found step length
    algorithm 3.1 @ P37 of numerical optimization, J.Nocedal & SJ. Wright
    '''
    alpha=alpha0
    while(func(x0+alpha*p)>func(x0)+c*alpha*np.dot(grad,p)):
        alpha=alpha*rho
    return alpha

def steepestDescent(func, x0, grad=None, tol=1e-6, maxIter=1000):
    '''
    func:if grad is None then (f(x), gf(x))=func(x) or else f(x)=func(x)
    x0: init value of x for the iteration
    grad: gf(x)=grad(x), the first order derivative function of f(x) 
    tol: stop when abs(f(x_{k+1})-f(x_{k}))<=tol 
    maxIter: maximum iteration number
    
    return value(s):
    x:  optimized result of x
    vf: function value at x
    info:information
    '''
    def f(x):
        if grad==None:
            return func(x)
        else:
            func(x),grad(x)
    vf0,gf0=f(x0)
    p=-gf0
    alpha=backTrack(lambda x:f(x)[0], x0, p, gf0)
    x1=x0+alpha*p
    vf1,gf1=f(x1)
    iterNum=1
    while np.abs(vf1-vf0)>tol:
        x0,vf0,gf0=x1,vf1,gf1
        p=-gf0
        alpha=backTrack(lambda x:f(x)[0], x0, p, gf0)
        x1=x0+alpha*p
        vf1,gf1=f(x1)
        iterNum=iterNum+1
        if iterNum==maxIter:
            return (x1, vf1, 'warning: exceeding maximum iteration number')
    return  (x1,vf1, 'msg: this is more likely to be convergent')

def newtonMethod(func, grad, hessian, x0, tol=1e-6, maxIter=1000):
    '''
    func,grad,hessian:(f(x), gf(x), hessianf(x))=func(x),grad(x),hessian(x)
    tol: stop when abs(f(x_{k+1})-f(x_{k}))<=tol 
    maxIter: maximum iteration number
    
    return value(s):
    x:  optimized result of x
    vf: function value at x
    info:information
    '''
    vf0,gf0,hf=func(x0),grad(x0),hessian(x0)
    (V,D)=eig(hf)
    invD=diag(map(lambda x: 1/(abs(x)), diag(D)))
    p=-dot(V*invD*V.H,gf0)
    alpha=backTrack(func, x0, p, gf0)
    x1=x0+alpha*p
    vf1,gf1,hf=func(x1),grad(x1),hessian(x1)
    iterNum=1
    while np.abs(vf1-vf0)>tol:
        x0,vf0,gf0=x1,vf1,gf1
        (V,D)=eig(hf)
        invD=diag(map(lambda x: 1/(abs(x)), diag(D)))
        p=-dot(V*invD*V.H,gf0)
        alpha=backTrack(func, x0, p, gf0)
        x1=x0+alpha*p
        vf1,gf1,hf=func(x1),grad(x1),hessian(x1)
        if iterNum==maxIter:
            return (x1, vf1, 'warning: exceeding maximum iteration number')
    return  (x1,vf1, 'msg: this is more likely to be convergent')
