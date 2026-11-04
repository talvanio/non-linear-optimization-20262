"""Questão 4: método de Newton; validação por gradiente nulo."""
import numpy as np
from scipy.optimize import minimize
from comum import exportar, curva

K1, K2, K3, P = 1000., 1500., 2000., 100.
H = np.array([[K2+K3, -K3], [-K3, K1+K3]])
c = np.array([0., -P])

def f(x):
    x1, x2 = x
    return .5*K2*x1**2 + .5*K3*(x2-x1)**2 + .5*K1*x2**2-P*x2

def jac(x): return H @ x + c

def hess(x): return H

def executar():
    x0=np.array([0.,0.]); hist=[f(x0)]
    sol=minimize(f,x0, method='Newton-CG', jac=jac, hess=hess,
                 callback=lambda x: hist.append(f(x)), options={'xtol':1e-12,'maxiter':100})
    x_ref=np.linalg.solve(H,-c)
    dados=exportar('questao4',sol,hist,{'metodo':'Newton (Newton-CG da SciPy)', 'x_inicial':x0.tolist(),
        'x_analitico':x_ref.tolist(), 'erro_analitico':float(np.linalg.norm(sol.x-x_ref)),
        'norma_gradiente':float(np.linalg.norm(jac(sol.x))), 'autovalores_hessiana':np.linalg.eigvalsh(H).tolist()})
    curva('questao4',hist)
    print('Q4', {k:v for k,v in dados.items() if k!='historico_f'})
    assert np.linalg.norm(sol.x-x_ref)<1e-6
if __name__=='__main__': executar()
