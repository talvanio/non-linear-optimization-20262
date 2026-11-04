"""Questão 2: Gradientes Conjugados não lineares, com gradiente analítico da TV suavizada."""
import numpy as np
from scipy.optimize import minimize, check_grad
from utils import obtem_imagem, op_A, op_AT
from comum import exportar, curva, OUT
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LAMBDA_REG=2e-2
EPSILON=1e-8

def preparar():
    return obtem_imagem()

def termos_gradiente(x,shape,b,kernel):
    X=np.asarray(x).reshape(shape)
    residuo=op_A(X,kernel)-b.reshape(shape)
    dx=np.roll(X,-1,axis=1)-X
    dy=np.roll(X,-1,axis=0)-X
    raiz=np.sqrt(dx**2+dy**2+EPSILON)
    tv=np.sum(raiz)
    fidel=np.sum(residuo**2)
    # O operador adjunto da diferença periódica é roll(q,+1)-q.
    qx=dx/raiz; qy=dy/raiz
    g_tv=np.roll(qx,1,axis=1)-qx+np.roll(qy,1,axis=0)-qy
    g=2*op_AT(residuo,kernel)+LAMBDA_REG*g_tv
    return float(fidel+LAMBDA_REG*tv),g.ravel(),float(fidel),float(tv)

def executar(maxiter=1000):
    imagem,shape,b,kernel=preparar()
    x0=imagem.ravel().copy()
    fun=lambda x: termos_gradiente(x,shape,b,kernel)[0]
    jac=lambda x: termos_gradiente(x,shape,b,kernel)[1]
    # Diferenças finitas apenas como teste de derivação, não método de otimização.
    rng=np.random.default_rng(42)
    idx=rng.choice(x0.size,10,replace=False)
    g=jac(x0); errors=[]
    for i in idx:
        h=1e-6;xp=x0.copy();xm=x0.copy();xp[i]+=h;xm[i]-=h
        errors.append(abs((fun(xp)-fun(xm))/(2*h)-g[i]))
    assert max(errors) < 1e-4, f'Erro no gradiente: {max(errors)}'
    hist=[fun(x0)]
    sol=minimize(fun,x0,method='CG',jac=jac, callback=lambda x:hist.append(fun(x)),
        options={'maxiter':maxiter, 'gtol':1e-5})
    f0,_,fid0,tv0=termos_gradiente(x0,shape,b,kernel)
    ff,gg,fidf,tvf=termos_gradiente(sol.x,shape,b,kernel)
    dados=exportar('questao2',sol,hist,{'metodo':'Gradientes Conjugados (CG)',
        'dimensao_imagem':list(shape),'maximo_iteracoes':maxiter,
        'f_inicial':f0,'fidelidade_inicial':fid0,'tv_inicial':tv0,
        'fidelidade_final':fidf,'tv_final':tvf,
        'norma_gradiente':float(np.linalg.norm(gg)), 'erro_maximo_teste_gradiente':float(max(errors))})
    curva('questao2',hist)
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    for ax,im,titulo in zip(axes,[imagem,sol.x.reshape(shape)],['Borrada e ruidosa','Reconstruída (CG)']):
        ax.imshow(im,cmap='gray',vmin=0,vmax=1);ax.set_title(titulo);ax.axis('off')
    fig.tight_layout();fig.savefig(OUT/'questao2'/'comparacao_imagens.png',dpi=220);plt.close(fig)
    print('Q2',{k:v for k,v in dados.items() if k not in ('historico_f','x_otimo')})
if __name__=='__main__': executar()
