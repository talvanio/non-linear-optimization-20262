# Trabalho Computacional I — ELE077 (UFMG)

Implementação das quatro questões com algoritmos estudados em `unidade3.pdf`:
- `questao1.py`: método quase-Newton BFGS (quociente de Rayleigh).
- `questao2.py`: método dos gradientes conjugados não lineares (imagem 32×32).
- `questao3.py`: método Nelder–Mead Simplex (minimax não diferenciável).
- `questao4.py`: método de Newton (quadrática estritamente convexa), via `Newton-CG` com Hessiana analítica.

## Instalação e execução

```bash
python -m pip install -r requirements.txt
python questao4.py
python questao1.py
python questao3.py
python questao2.py
```

Executar os comandos a partir desta pasta. Resultados numéricos JSON e figuras PNG são gravados em `resultados/questaoN`.

O uso de `p=exp(z)` na questão 3 é apenas uma mudança de variáveis para manter os preços estritamente positivos; não é um algoritmo adicional de otimização. Na questão 1, o vetor minimizante não é único em magnitude (ou sinal); a frequência é única. Na questão 2, `EPSILON=1e-8`, `LAMBDA_REG=0.02`, convolução com contorno periódico e ruído aleatório fixo seguem o material fornecido.

`questao2.py` inclui verificação de 10 componentes do gradiente por diferenças finitas (validação, não otimizador). A rotina de restauração não impõe artificialmente limites aos pixels, pois o problema foi proposto como irrestrito. O gráfico de saída usa escala visual `[0,1]`.
