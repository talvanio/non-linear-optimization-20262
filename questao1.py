"""Questão 1: quase-Newton BFGS para o quociente de Rayleigh."""
import numpy as np
from scipy.optimize import minimize
from comum import exportar, curva
M=np.eye(3)
K=np.array([[2.,-1.,0.],[-1.,2.,-1.],[0.,-1.,1.]])

def f(x):
    den=x@M@x
    if den <= 1e-24: return 1e15
    return float((x@K@x)/den)

def jac(x):
    den=x@M@x
    if den <= 1e-24: return np.zeros_like(x)
    return 2*(K@x-f(x)*M@x)/den

def executar():
    x0=np.array([1.,1.,1.]); hist=[f(x0)]
    sol=minimize(f,x0,method='BFGS', jac=jac,callback=lambda x:hist.append(f(x)),options={'gtol':1e-10,'maxiter':1000})
    vals,vecs=np.linalg.eigh(K)
    v=sol.x/np.linalg.norm(sol.x)
    erro_direcao=min(np.linalg.norm(v-vecs[:,0]), np.linalg.norm(v+vecs[:,0]))
    dados=exportar('questao1',sol,hist,{'metodo':'BFGS','x_inicial':x0.tolist(),
        'frequencia_rad_s':float(np.sqrt(sol.fun)), 'menor_autovalor':float(vals[0]),
        'erro_autovalor':float(abs(sol.fun-vals[0])), 'erro_direcao':float(erro_direcao),
        'residuo_autovetor':float(np.linalg.norm(K@v-sol.fun*M@v)),
        'norma_gradiente':float(np.linalg.norm(jac(sol.x)))})
    curva('questao1',hist)
    print('Q1',{k:v for k,v in dados.items() if k!='historico_f'})
    assert abs(sol.fun-vals[0]) < 1e-7
if __name__=='__main__': executar()
