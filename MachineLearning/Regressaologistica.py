import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Dados
salario = np.array([2,3,4,5,6,7,8,9,10,11]).reshape(-1,1)
comprou = np.array([0,0,0,1,0,1,1,1,1,1])

# Criando o modelo de regressão logística
modelo = LogisticRegression()
modelo.fit(salario, comprou)

# Probabilidades previstas
salario_range = np.linspace(2, 11, 100).reshape(-1,1)
probabilidades = modelo.predict_proba(salario_range)[:,1]

# Exibindo os coeficientes
print("Coeficiente angular (b1):", modelo.coef_[0][0])
print("Intercepto (b0):", modelo.intercept_[0])

# Gráfico
plt.scatter(salario, comprou, color='blue', label='Dados reais')
plt.plot(salario_range, probabilidades, color='red', label='Regressão Logística')
plt.xlabel("Salário (R$ mil)")
plt.ylabel("Comprou apartamento")
plt.title("Regressão Logística: Salário x Compra de Apartamento")
plt.legend()
plt.grid(True)
plt.show()
