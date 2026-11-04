"""Registro uniforme de resultados para o Trabalho Computacional I."""
import json
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent
OUT = BASE / 'resultados'
OUT.mkdir(exist_ok=True)

def exportar(nome, sol, historico, extras=None):
    pasta = OUT / nome
    pasta.mkdir(exist_ok=True)
    saida = {
        'metodo': str(extras.get('metodo', '')) if extras else '',
        'x_otimo': np.asarray(sol.x).tolist(),
        'f_otimo': float(sol.fun),
        'iteracoes': int(sol.nit),
        'avaliacoes_funcao': int(sol.nfev),
        'avaliacoes_gradiente': int(sol.njev) if getattr(sol, 'njev', None) is not None else None,
        'sucesso': bool(sol.success),
        'mensagem': str(sol.message),
        'historico_f': [float(a) for a in historico],
    }
    if extras: saida.update({k:v for k,v in extras.items() if k != 'metodo'})
    (pasta / 'resultados.json').write_text(json.dumps(saida, indent=2, ensure_ascii=False), encoding='utf-8')
    return saida

def curva(nome, valores):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(range(len(valores)), valores, '-o', markersize=2)
    ax.set(xlabel='Iteração registrada', ylabel='Função objetivo', title='Convergência — '+nome)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT/nome/'convergencia.png', dpi=170)
    plt.close(fig)
