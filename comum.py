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

def curvas_nivel(nome, func, xlim, ylim, n=200, niveis=30, trajetoria=None, pontos=None,
                 sobrepor=None, rotulos=('x1', 'x2'), titulo=None, arquivo='curvas_nivel.png'):
    """Curvas de nível de func(a, b) -> float na janela xlim x ylim.

    trajetoria: sequência (k, 2) de iterados do otimizador.
    pontos: {rótulo: (x, y, estilo_matplotlib)} para marcar ótimos/pontos estacionários.
    sobrepor: callback f(ax) para desenhos específicos de cada questão.
    niveis: número de níveis ou lista explícita de níveis.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    xs = np.linspace(*xlim, n)
    ys = np.linspace(*ylim, n)
    Z = np.array([[func(a, b) for a in xs] for b in ys])
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    cf = ax.contourf(xs, ys, Z, levels=niveis, cmap='viridis', extend='max')
    ax.contour(xs, ys, Z, levels=cf.levels, colors='k', linewidths=0.3, alpha=0.5)
    fig.colorbar(cf, ax=ax, label='Função objetivo')
    if sobrepor is not None:
        sobrepor(ax)
    if trajetoria is not None and len(trajetoria) > 0:
        t = np.asarray(trajetoria)
        ax.plot(t[:, 0], t[:, 1], '-o', color='white', markersize=3, linewidth=1, label='Trajetória')
        ax.plot(t[0, 0], t[0, 1], 's', color='orange', markersize=8, label='Ponto inicial')
    for rotulo, (px, py, estilo) in (pontos or {}).items():
        ax.plot(px, py, estilo, markersize=12, markeredgecolor='k', label=rotulo)
    ax.set(xlim=xlim, ylim=ylim, xlabel=rotulos[0], ylabel=rotulos[1], title=titulo or 'Curvas de nível — '+nome)
    ax.legend(loc='best', fontsize=8)
    fig.tight_layout()
    (OUT/nome).mkdir(exist_ok=True)
    fig.savefig(OUT/nome/arquivo, dpi=170)
    plt.close(fig)
