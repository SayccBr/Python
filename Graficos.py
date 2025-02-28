import numpy as np
import matplotlib.pyplot as plt

# Definindo as funções
def f(n):
    return n ** 2

def g(n):
    return 2 ** n

def h(n):
    return np.log(n)

def i(n):
    return n * np.log(n)

# Intervalo de valores para n
n = np.linspace(1, 100, 400)

# Calculando os valores das funções
f_values = f(n)
g_values = g(n)
h_values = h(n)
i_values = i(n)

# Plotando as funções
plt.figure(figsize=(10, 6))
plt.plot(n, h_values, label='h(n) = log(n)')
plt.plot(n, i_values, label='i(n) = n log(n)')
plt.plot(n, f_values, label='f(n) = n^2')
plt.plot(n, g_values, label='g(n) = 2^n')

plt.yscale('log') # Escala logarítmica para melhor visualização
plt.xlabel('n')
plt.ylabel('f(n)')
plt.title('Crescimento das funções')
plt.legend()
plt.grid(True)
plt.show()
