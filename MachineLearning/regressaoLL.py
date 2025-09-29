import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
mean_squared_error, mean_absolute_error, r2_score,
accuracy_score, precision_score, recall_score, f1_score,
log_loss, roc_auc_score, roc_curve
)

# Dataset

data = {
"Aluno": list(range(1, 21)),
"Horas": [1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,
6.0,6.5,7.0,7.5,8.0,8.5,9.0,9.5,10.0,10.5],
"Nota": [28,35,42,48,52,55,58,62,65,68,
70,72,75,78,80,83,85,88,90,93],
"Status": [0,0,0,0,0,0,0,1,1,1,
1,1,1,1,1,1,1,1,1,1]
}
df = pd.DataFrame(data)

X = df[["Horas"]].values
y_reg = df["Nota"].values
y_clf = df["Status"].values

# --- Regressão Linear ---

lin_reg = LinearRegression()
lin_reg.fit(X, y_reg)
y_pred_lin = lin_reg.predict(X)

# Métricas regressão linear

MSE = mean_squared_error(y_reg, y_pred_lin)
RMSE = np.sqrt(MSE)
MAE = mean_absolute_error(y_reg, y_pred_lin)
R2 = r2_score(y_reg, y_pred_lin)
SSE = np.sum((y_reg - y_pred_lin)**2)

print("=== Regressão Linear ===")
print(f"Coeficientes: a (intercept) = {lin_reg.intercept_:.4f}, b (slope) = {lin_reg.coef_[0]:.4f}")
print(f"MSE: {MSE:.4f}")
print(f"RMSE: {RMSE:.4f}")
print(f"MAE: {MAE:.4f}")
print(f"R²: {R2:.4f}")
print(f"SSE: {SSE:.4f}")
print()

# Gráfico 1: pontos e reta prevista

plt.figure()
plt.scatter(X.flatten(), y_reg, label="Dados (nota)")
xs = np.linspace(X.min(), X.max(), 200).reshape(-1,1)
ys_line = lin_reg.predict(xs)
plt.plot(xs.flatten(), ys_line.flatten(), label="Reta prevista (LinearRegression)", color="red")
plt.xlabel("Horas de Estudo")
plt.ylabel("Nota")
plt.title("Regressão Linear — Nota vs Horas")
plt.legend()
plt.grid(True)
plt.show()

# Gráfico 2: resíduos

res = y_reg - y_pred_lin
plt.figure()
plt.scatter(X.flatten(), res, label="Resíduos (y - y_pred)")
plt.axhline(0, linestyle="--", label="Linha zero", color="red")
plt.xlabel("Horas de Estudo")
plt.ylabel("Resíduo")
plt.title("Resíduos da Regressão Linear")
plt.legend()
plt.grid(True)
plt.show()

# --- Regressão Logística ---

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X, y_clf)
y_pred_log = log_reg.predict(X)
y_proba_log = log_reg.predict_proba(X)[:,1]

# Métricas regressão logística

acc = accuracy_score(y_clf, y_pred_log)
prec = precision_score(y_clf, y_pred_log)
rec = recall_score(y_clf, y_pred_log)
f1 = f1_score(y_clf, y_pred_log)
ll = log_loss(y_clf, y_proba_log)
auc = roc_auc_score(y_clf, y_proba_log)

print("=== Regressão Logística ===")
print(f"Coeficientes: a (intercept) = {log_reg.intercept_[0]:.4f}, b (slope) = {log_reg.coef_[0][0]:.4f}")
print(f"Acurácia: {acc:.4f}")
print(f"Precisão: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Log Loss: {ll:.4f}")
print(f"AUC: {auc:.4f}")
print()

# Gráfico 3: curva sigmoide

plt.figure()
plt.scatter(X.flatten(), y_clf, label="Status real (0/1)")
xs = np.linspace(X.min(), X.max(), 300).reshape(-1,1)
ys = log_reg.predict_proba(xs)[:,1]
plt.plot(xs.flatten(), ys, label="Probabilidade prevista (sigmóide)", color="red")
plt.axhline(0.5, linestyle="--", label="Limiar 0.5", color="green")
plt.xlabel("Horas de Estudo")
plt.ylabel("Probabilidade de Aprovação")
plt.title("Regressão Logística — Probabilidade vs Horas")
plt.legend()
plt.grid(True)
plt.show()

# Gráfico 4: Curva ROC

fpr, tpr, _ = roc_curve(y_clf, y_proba_log)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})", color="blue")
plt.plot([0,1],[0,1], linestyle="--", label="Aleatório", color="red")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("Curva ROC — Regressão Logística")
plt.legend()
plt.grid(True)
plt.show()