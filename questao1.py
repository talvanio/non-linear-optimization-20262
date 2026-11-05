"""Questão 1: quase-Newton BFGS para o quociente de Rayleigh."""
import numpy as np
from scipy.optimize import minimize
from comum import exportar, curva, curvas_nivel
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
    x0=np.array([1.,1.,1.]); hist=[f(x0)]; traj=[x0.copy()]
    def registrar(x): hist.append(f(x)); traj.append(x.copy())  # guarda f e o iterado (para a trajetória)
    sol=minimize(f,x0,method='BFGS', jac=jac,callback=registrar,options={'gtol':1e-10,'maxiter':1000})
    vals,vecs=np.linalg.eigh(K)
    v=sol.x/np.linalg.norm(sol.x)
    erro_direcao=min(np.linalg.norm(v-vecs[:,0]), np.linalg.norm(v+vecs[:,0]))
    dados=exportar('questao1',sol,hist,{'metodo':'BFGS','x_inicial':x0.tolist(),
        'frequencia_rad_s':float(np.sqrt(sol.fun)), 'menor_autovalor':float(vals[0]),
        'erro_autovalor':float(abs(sol.fun-vals[0])), 'erro_direcao':float(erro_direcao),
        'residuo_autovetor':float(np.linalg.norm(K@v-sol.fun*M@v)),
        'norma_gradiente':float(np.linalg.norm(jac(sol.x)))})
    curva('questao1',hist)
    # R(alpha x) = R(x): basta olhar o plano x3 = 1 (projeção x/x3).
    traj_proj=[x[:2]/x[2] for x in traj if abs(x[2])>1e-12]
    rotulos_av=[('Mínimo (λ1=%.3f)'%vals[0],'r*'),('Sela (λ2=%.3f)'%vals[1],'cX'),('Máximo (λ3=%.3f)'%vals[2],'m^')]
    autovetores={nome:(vecs[0,i]/vecs[2,i], vecs[1,i]/vecs[2,i], estilo) for i,(nome,estilo) in enumerate(rotulos_av)}
    curvas_nivel('questao1', lambda a,b: f(np.array([a,b,1.])), (-3,3), (-3,3),
        trajetoria=traj_proj, pontos=autovetores, rotulos=('x1/x3','x2/x3'),
        titulo='Quociente de Rayleigh no plano x3 = 1')
    print('Q1',{k:v for k,v in dados.items() if k!='historico_f'})
    assert abs(sol.fun-vals[0]) < 1e-7
if __name__=='__main__': executar()
