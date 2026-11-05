"""Questão 3: Nelder-Mead Simplex; parametrização positiva p=exp(z)."""
import numpy as np
from scipy.optimize import minimize
from comum import exportar, curva, curvas_nivel
A=np.array([[100000.,80000.],[70000.,56000.],[130000.,104000.],[90000.,90000.]])
ALVO=10000.

def receitas(p): return A @ (1/np.asarray(p))
def f_p(p): return np.max(np.abs(receitas(p)-ALVO)) if np.all(np.asarray(p)>0) else np.inf
def f_z(z): return f_p(np.exp(z))

def executar():
    z0=np.log([12.,25.]); hist=[f_z(z0)]; traj=[np.exp(z0)]
    def registrar(z): hist.append(f_z(z)); traj.append(np.exp(z))  # guarda f e o melhor vértice (em p)
    sol=minimize(f_z,z0,method='Nelder-Mead',callback=registrar,
        options={'maxiter':20000,'maxfev':30000,'xatol':1e-12,'fatol':1e-9, 'adaptive':False})
    p=np.exp(sol.x); receita=receitas(p)
    dados=exportar('questao3',sol,hist, {'metodo':'Nelder-Mead Simplex', 'parametrizacao':'p_j=exp(z_j)',
        'precos_otimos':p.tolist(),'precos_iniciais':np.exp(z0).tolist(),
        'receitas_cenarios':receita.tolist(),'desvios_cenarios':np.abs(receita-ALVO).tolist(),
        'pior_desvio':float(f_p(p))})
    curva('questao3',hist)
    def curva_otimos(ax):
        # Cenários 1-3 são proporcionais: o pior desvio vale 3000 sobre 100000/p1 + 80000/p2 = 10000.
        p1=np.linspace(10.05,40,400); p2=80000/(ALVO-100000/p1)
        ok=p2<=40
        ax.plot(p1[ok],p2[ok],'--',color='red',linewidth=1.5,label='Conjunto ótimo (desvio 3000)')
    curvas_nivel('questao3', lambda a,b: f_p([a,b]), (5,40), (5,40), niveis=np.linspace(3000,15000,25),
        trajetoria=traj, pontos={'Ótimo encontrado':(p[0],p[1],'r*')}, sobrepor=curva_otimos,
        rotulos=('Preço p1','Preço p2'), titulo='Objetivo minimax (pior desvio)', arquivo='contorno.png')
    print('Q3',{k:v for k,v in dados.items() if k!='historico_f'})
if __name__=='__main__': executar()
