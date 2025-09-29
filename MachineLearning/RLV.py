import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
import warnings
warnings.filterwarnings('ignore')

print("📚 FUNDAMENTOS TEÓRICOS - DEMONSTRAÇÃO PRÁTICA")
print("=" * 55)

# ================================================================
# DATASET SUPER SIMPLES PARA DEMONSTRAÇÃO
# ================================================================

print("\n🏠 CRIANDO DATASET MÍNIMO PARA DEMONSTRAÇÃO...")

# Apenas 12 casas para facilitar o entendimento
area_m2 = np.array([50, 70, 90, 110, 130, 150, 60, 80, 100, 120, 140, 160])
preco_mil = np.array([200, 250, 300, 350, 400, 450, 220, 270, 320, 370, 420, 470])

# DataFrame para visualização
df_simples = pd.DataFrame({
    'area_m2': area_m2,
    'preco_mil': preco_mil
})

print("✅ Dataset com 12 casas criado!")
print(f"\n📊 DADOS COMPLETOS:")
print(df_simples)

# ================================================================
# PARTE 1: ENTENDENDO A REGRESSÃO LINEAR NA PRÁTICA
# ================================================================

print("\n" + "="*55)
print("📈 PARTE 1: REGRESSÃO LINEAR - PASSO A PASSO")
print("="*55)

# Visualização inicial dos dados
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(area_m2, preco_mil, color='blue', s=100, alpha=0.7)
plt.xlabel('Área (m²)')
plt.ylabel('Preço (mil R$)')
plt.title('Dados Originais')
plt.grid(True, alpha=0.3)

# Ajustando o modelo linear
X = area_m2.reshape(-1, 1)  # sklearn precisa de matriz 2D
y = preco_mil

modelo_linear = LinearRegression()
modelo_linear.fit(X, y)

# Fazendo predições
y_pred = modelo_linear.predict(X)

# Extraindo coeficientes
intercepto = modelo_linear.intercept_
coeficiente = modelo_linear.coef_[0]

print(f"\n🔍 EQUAÇÃO ENCONTRADA:")
print(f"   Preço = {intercepto:.2f} + {coeficiente:.2f} × Área")
print(f"   ou seja: Preço = {intercepto:.2f} + {coeficiente:.2f} × Área")

print(f"\n💡 INTERPRETAÇÃO:")
print(f"   • Intercepto ({intercepto:.2f}): Preço base quando área = 0")
print(f"   • Coeficiente ({coeficiente:.2f}): A cada 1m² a mais, o preço sobe R$ {coeficiente:.2f}k")

# Visualizando a linha de regressão
plt.subplot(1, 3, 2)
plt.scatter(area_m2, preco_mil, color='blue', s=100, alpha=0.7, label='Dados Reais')
plt.plot(area_m2, y_pred, color='red', linewidth=3, label='Linha de Regressão')
plt.xlabel('Área (m²)')
plt.ylabel('Preço (mil R$)')
plt.title('Regressão Linear Ajustada')
plt.legend()
plt.grid(True, alpha=0.3)

# Mostrando os resíduos (erros)
residuos = preco_mil - y_pred
plt.subplot(1, 3, 3)
plt.scatter(area_m2, residuos, color='green', s=100, alpha=0.7)
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Área (m²)')
plt.ylabel('Resíduos (Erro)')
plt.title('Análise de Resíduos')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n🎯 ANÁLISE DOS RESULTADOS:")
print(f"   • R² Score: {modelo_linear.score(X, y):.3f} (explica {modelo_linear.score(X, y)*100:.1f}% da variação)")
print(f"   • Erro médio absoluto: {np.mean(np.abs(residuos)):.2f} mil R$")

print(f"\n🔮 EXEMPLO DE PREDIÇÃO:")
nova_area = 125
preco_predito = intercepto + coeficiente * nova_area
print(f"   • Casa de {nova_area}m²: Preço predito = {preco_predito:.2f} mil R$")

# Mostrando predições para todas as casas
print(f"\n📋 COMPARAÇÃO REAL vs PREDITO:")
for i in range(len(area_m2)):
    erro = abs(preco_mil[i] - y_pred[i])
    print(f"   Casa {i+1:2d}: {area_m2[i]:3d}m² → Real: {preco_mil[i]:3d}k, Pred: {y_pred[i]:5.1f}k, Erro: {erro:4.1f}k")

# ================================================================
# PARTE 2: ENTENDENDO A REGRESSÃO LOGÍSTICA
# ================================================================

print("\n" + "="*55)
print("🏷️ PARTE 2: REGRESSÃO LOGÍSTICA - PASSO A PASSO")
print("="*55)

# Criando classificação binária baseada na mediana
mediana_preco = np.median(preco_mil)
y_binario = (preco_mil > mediana_preco).astype(int)  # 1 = cara, 0 = barata

print(f"\n🎯 TRANSFORMANDO EM PROBLEMA DE CLASSIFICAÇÃO:")
print(f"   • Limiar (mediana): {mediana_preco:.0f} mil R$")
print(f"   • Casas 'Caras' (1): {sum(y_binario)} casas")
print(f"   • Casas 'Baratas' (0): {sum(1-y_binario)} casas")

print(f"\n📋 CLASSIFICAÇÃO DAS CASAS:")
for i in range(len(area_m2)):
    classe = "Cara" if y_binario[i] == 1 else "Barata"
    print(f"   Casa {i+1:2d}: {area_m2[i]:3d}m², {preco_mil[i]:3d}k → {classe}")

# Treinando modelo logístico
modelo_logistico = LogisticRegression(random_state=42)
modelo_logistico.fit(X, y_binario)

# Predições
y_pred_classe = modelo_logistico.predict(X)
y_pred_proba = modelo_logistico.predict_proba(X)[:, 1]  # Probabilidade de ser cara

# Coeficientes da regressão logística
intercepto_log = modelo_logistico.intercept_[0]
coeficiente_log = modelo_logistico.coef_[0][0]

print(f"\n🔍 EQUAÇÃO LOGÍSTICA ENCONTRADA:")
print(f"   z = {intercepto_log:.3f} + {coeficiente_log:.5f} × Área")
print(f"   P(Casa Cara) = 1 / (1 + e^(-z))")

print(f"\n💡 INTERPRETAÇÃO:")
print(f"   • Coeficiente positivo ({coeficiente_log:.5f}): Área maior → Maior prob. de ser cara")
print(f"   • Quanto maior a área, maior a probabilidade de ser classificada como cara")

# Visualização da regressão logística
plt.figure(figsize=(15, 5))

# Dados com classificação
plt.subplot(1, 3, 1)
cores = ['red' if classe == 0 else 'blue' for classe in y_binario]
plt.scatter(area_m2, y_binario, c=cores, s=100, alpha=0.7)
plt.xlabel('Área (m²)')
plt.ylabel('Classe (0=Barata, 1=Cara)')
plt.title('Classificação das Casas')
plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
plt.grid(True, alpha=0.3)

# Função sigmóide
area_continua = np.linspace(40, 170, 100)
z_continuo = intercepto_log + coeficiente_log * area_continua
prob_continua = 1 / (1 + np.exp(-z_continuo))

plt.subplot(1, 3, 2)
plt.scatter(area_m2, y_pred_proba, c=cores, s=100, alpha=0.7, label='Dados')
plt.plot(area_continua, prob_continua, 'green', linewidth=3, label='Curva Sigmóide')
plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Limiar 50%')
plt.xlabel('Área (m²)')
plt.ylabel('P(Casa Cara)')
plt.title('Probabilidades Preditas')
plt.legend()
plt.grid(True, alpha=0.3)

# Comparação das classificações
acertos = sum(y_binario == y_pred_classe)
acuracia = acertos / len(y_binario)

plt.subplot(1, 3, 3)
resultados = ['Correto' if real == pred else 'Erro' for real, pred in zip(y_binario, y_pred_classe)]
cores_resultado = ['green' if r == 'Correto' else 'red' for r in resultados]
plt.scatter(range(len(area_m2)), y_binario, c=cores_resultado, s=100, alpha=0.7)
plt.xlabel('Índice da Casa')
plt.ylabel('Classe Real')
plt.title(f'Acertos vs Erros (Acurácia: {acuracia:.2f})')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n🎯 ANÁLISE DOS RESULTADOS LOGÍSTICOS:")
print(f"   • Acurácia: {acuracia:.3f} ({acuracia*100:.1f}% de acertos)")
print(f"   • Acertos: {acertos} de {len(y_binario)} casas")

print(f"\n📋 RESULTADOS DETALHADOS:")
for i in range(len(area_m2)):
    real_classe = "Cara" if y_binario[i] == 1 else "Barata"
    pred_classe = "Cara" if y_pred_classe[i] == 1 else "Barata"
    probabilidade = y_pred_proba[i]
    resultado = "✓" if y_binario[i] == y_pred_classe[i] else "✗"
    print(f"   Casa {i+1:2d}: {area_m2[i]:3d}m² → Real: {real_classe:6s}, Pred: {pred_classe:6s}, Prob: {probabilidade:.3f} {resultado}")

print(f"\n🔮 EXEMPLO DE NOVA PREDIÇÃO:")
nova_area_log = 125
z_novo = intercepto_log + coeficiente_log * nova_area_log
prob_novo = 1 / (1 + np.exp(-z_novo))
classe_nova = "Cara" if prob_novo > 0.5 else "Barata"
print(f"   • Casa de {nova_area_log}m²: P(Cara) = {prob_novo:.3f}, Classificação: {classe_nova}")

# ================================================================
# COMPARAÇÃO ENTRE OS DOIS MÉTODOS
# ================================================================

print("\n" + "="*55)
print("🔄 COMPARAÇÃO ENTRE REGRESSÃO LINEAR E LOGÍSTICA")
print("="*55)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Regressão Linear
ax1.scatter(area_m2, preco_mil, color='blue', s=100, alpha=0.7, label='Dados Reais')
ax1.plot(area_m2, y_pred, color='red', linewidth=3, label='Regressão Linear')
ax1.set_xlabel('Área (m²)')
ax1.set_ylabel('Preço (mil R$)')
ax1.set_title('Regressão Linear\n(Predição Contínua)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Regressão Logística
cores = ['red' if classe == 0 else 'blue' for classe in y_binario]
ax2.scatter(area_m2, y_binario, c=cores, s=100, alpha=0.7, label='Classes Reais')
ax2.plot(area_continua, prob_continua, 'green', linewidth=3, label='Regressão Logística')
ax2.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Limiar 50%')
ax2.set_xlabel('Área (m²)')
ax2.set_ylabel('P(Casa Cara)')
ax2.set_title('Regressão Logística\n(Classificação)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n🎯 PRINCIPAIS DIFERENÇAS:")
print(f"   • LINEAR: Prediz valores contínuos (preço exato)")
print(f"   • LOGÍSTICA: Prediz probabilidades e classes (cara/barata)")
print(f"   • LINEAR: Pode dar qualquer valor")
print(f"   • LOGÍSTICA: Sempre entre 0 e 1 (probabilidades)")

print(f"\n💡 QUANDO USAR CADA UM:")
print(f"   • REGRESSÃO LINEAR:")
print(f"     ✓ Quando você quer prever um valor numérico")
print(f"     ✓ Ex: preço, temperatura, vendas, peso")
print(f"   • REGRESSÃO LOGÍSTICA:")
print(f"     ✓ Quando você quer classificar em categorias")
print(f"     ✓ Ex: spam/não-spam, aprovado/reprovado, doente/saudável")

# ================================================================
# CONCEITOS MATEMÁTICOS IMPORTANTES
# ================================================================

print(f"\n" + "="*55)
print("🧮 CONCEITOS MATEMÁTICOS FUNDAMENTAIS")
print("="*55)

print(f"\n📐 MÉTODO DOS MÍNIMOS QUADRADOS (Regressão Linear):")
print(f"   • Objetivo: Minimizar Σ(yi - ŷi)²")
print(f"   • Em português: Minimizar a soma dos quadrados dos erros")
print(f"   • No nosso exemplo: Σ dos erros² = {sum(residuos**2):.2f}")

print(f"\n📊 FUNÇÃO SIGMÓIDE (Regressão Logística):")
print(f"   • Fórmula: σ(z) = 1 / (1 + e^(-z))")
print(f"   • Mapeia qualquer valor real para [0,1]")
print(f"   • Forma de 'S': cresce suavemente de 0 para 1")

print(f"\n🎯 MÉTRICAS DE AVALIAÇÃO:")
print(f"   • REGRESSÃO LINEAR:")
print(f"     - R²: {modelo_linear.score(X, y):.3f} (% da variação explicada)")
print(f"     - MSE: {np.mean(residuos**2):.2f} (erro quadrático médio)")
print(f"     - RMSE: {np.sqrt(np.mean(residuos**2)):.2f} (erro na unidade original)")
print(f"   • REGRESSÃO LOGÍSTICA:")
print(f"     - Acurácia: {acuracia:.3f} (% de acertos)")
print(f"     - Log-likelihood: Mede a 'probabilidade' dos dados dado o modelo")

print(f"\n✅ CONCEITOS FUNDAMENTAIS DEMONSTRADOS COM SUCESSO!")
print(f"   Com apenas 12 casas, conseguimos entender:")
print(f"   • Como funciona a regressão linear")
print(f"   • Como funciona a regressão logística") 
print(f"   • As diferenças práticas entre elas")
print(f"   • Como interpretar os resultados")