import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Dados
horas = np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1,1)
nota = np.array([50,55,58,62,65,68,72,75,78,82])

# Criando o modelo de regressão linear
modelo = LinearRegression()
modelo.fit(horas, nota)

# Predições
nota_predita = modelo.predict(horas)

# Calculando o MSE
mse = mean_squared_error(nota, nota_predita)

# Preparando dados para o gráfico
horas_range = np.linspace(1, 10, 100).reshape(-1,1)
nota_range_predita = modelo.predict(horas_range)

# Exibindo os coeficientes
print("Coeficiente angular (b1):", modelo.coef_[0])
print("Intercepto (b0):", modelo.intercept_)
print("MSE:", mse)

# Gráfico com MSE na legenda
plt.scatter(horas, nota, color='blue', label='Dados reais')
plt.plot(horas_range, nota_range_predita, color='red', 
         label=f'Regressão Linear (MSE={mse:.2f})')
plt.xlabel("Horas de Estudo")
plt.ylabel("Nota na Prova")
plt.title("Regressão Linear: Horas de Estudo x Nota")
plt.legend()
plt.grid(True)
plt.show()
