"""Questão 3: Nelder-Mead Simplex; parametrização positiva p=exp(z)."""
import numpy as np
from scipy.optimize import minimize
from comum import exportar, curva
A=np.array([[100000.,80000.],[70000.,56000.],[130000.,104000.],[90000.,90000.]])
ALVO=10000.

def receitas(p): return A @ (1/np.asarray(p))
def f_p(p): return np.max(np.abs(receitas(p)-ALVO)) if np.all(np.asarray(p)>0) else np.inf
def f_z(z): return f_p(np.exp(z))

def executar():
    z0=np.log([12.,25.]); hist=[f_z(z0)]
    sol=minimize(f_z,z0,method='Nelder-Mead',callback=lambda z:hist.append(f_z(z)),
        options={'maxiter':20000,'maxfev':30000,'xatol':1e-12,'fatol':1e-9, 'adaptive':False})
    p=np.exp(sol.x); receita=receitas(p)
    dados=exportar('questao3',sol,hist, {'metodo':'Nelder-Mead Simplex', 'parametrizacao':'p_j=exp(z_j)',
        'precos_otimos':p.tolist(),'precos_iniciais':np.exp(z0).tolist(),
        'receitas_cenarios':receita.tolist(),'desvios_cenarios':np.abs(receita-ALVO).tolist(),
        'pior_desvio':float(f_p(p))})
    curva('questao3',hist)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    xs=np.linspace(10,40,170);ys=np.linspace(10,40,170)
    Z=np.array([[f_p([a,b]) for a in xs] for b in ys])
    fig,ax=plt.subplots(figsize=(6,5)); ct=ax.contourf(xs,ys,Z,levels=25);fig.colorbar(ct,ax=ax,label='Pior desvio')
    ax.plot(p[0],p[1],'r*',markersize=12);ax.set(xlabel='Preço p1',ylabel='Preço p2',title='Objetivo minimax')
    fig.tight_layout();fig.savefig('resultados/questao3/contorno.png',dpi=160);plt.close(fig)
    print('Q3',{k:v for k,v in dados.items() if k!='historico_f'})
if __name__=='__main__': executar()
