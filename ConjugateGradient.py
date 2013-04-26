'''
Created on Apr 26, 2013

@author: ckt
'''

from numpy import dot
from numpy.linalg import norm
from scipy.optimize.linesearch import line_search_wolfe1

def linearConjugateGradient(A, x0, b, N=None, tol=1e-6):
    '''
    A,b: Ax=b, the linear system
    x0: initial value of x
    N: size of linear system
    tol:stop when abs(A(x_{k+1})-A(x_{k}))<=tol 
    
    return value:
    x, the solve of linear system
    
    the convergence analysis:
    ((\lambda_{n-k}-\lambda_{1})/(\lambda_{n-k}+\lambda_{1}))^{2}
    we can use precondition technology to get better convergence
    by multiply a new matrix P to A as a new linear system P*Ax=P*b
    no single preconditioning strategy is best for all
    '''
    r0=dot(A,x0)-b
    if N==None:
        N=A.shape[0]
    p0=-r0
    for k in range(0,N):
        if norm(r0)<tol:
            print 'stop in ', k, ' th step'
            return x0
        q=dot(A,p0)
        alpha=dot(r0.T,r0)/dot(p0.T, q)
        x0=x0+alpha*p0
        r1=r0+alpha*q
        beta=dot(r1.T, r1)/dot(r0.T, r0)
        p0=-r1+beta*p0
        r0=r1
    print 'stop in ', N, ' th step'
    return x0


def nonlinearConjugateGradient(func, grad, x0, tol=1e-6, maxIter=10000):
    '''
    func,grad: f(x),gf(x)=func(x),grad(x)
    x0: starting point of x
    tol: stop while the grad is less than tol
    maxIter: maximum iteration number
    
    return value:
    x: the optimized point of x
    
    line_search_wolfe1 is the wrap of minpack from scipy
    '''
    gf0=grad(x0)
    p0=-gf0
    iterNum=1
    while(norm(gf0)>tol):
        alpha=line_search_wolfe1(func, grad, x0, p0, c2=0.5)
        x0=x0+alpha*p0
        if iterNum==maxIter:
            return x0
        gf1=grad(x0)
        beta=(norm(gf1)/norm(gf0))**2
        p0=-gf1+beta*p0
        gf0=gf1
        iterNum=iterNum+1
    return x0
