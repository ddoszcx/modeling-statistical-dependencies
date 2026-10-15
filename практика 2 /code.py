import math
import pandas as pd
from sklearn.linear_model import LinearRegression

def mse(y_true, y_pred):
    n = len(y_true)
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / n

def rmse(y_true, y_pred):
    return math.sqrt(mse(y_true, y_pred))

def mae(y_true, y_pred):
    n = len(y_true)
    return sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / n

def r2_score(y_true, y_pred):
    n = len(y_true)
    y_mean = sum(y_true) / n
    ss_tot = sum((yt - y_mean) ** 2 for yt in y_true)
    ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

def msle(y_true, y_pred):
    n = len(y_true)
    return sum((math.log(1 + yt) - math.log(1 + yp)) ** 2 for yt, yp in zip(y_true, y_pred)) / n

def rmsle(y_true, y_pred):
    return math.sqrt(msle(y_true, y_pred))

def aic(y_true, y_pred, k):
    n = len(y_true)
    mse_val = mse(y_true, y_pred)
    return n * math.log(mse_val) + 2 * k

def bic(y_true, y_pred, k):
    n = len(y_true)
    mse_val = mse(y_true, y_pred)
    return n * math.log(mse_val) + k * math.log(n)

df = pd.read_excel('семинар_4.xlsx')

X = df[['x1', 'x2', 'x3', 'x4']]
y_true = df['y'].tolist()

model = LinearRegression()
model.fit(X, df['y'])

y_pred = model.predict(X).tolist()
k = X.shape[1] + 1

print("\n--- Результаты расчёта метрик качества ---")
print(f"MSE:   {mse(y_true, y_pred)}")
print(f"RMSE:  {rmse(y_true, y_pred)}")
print(f"MAE:   {mae(y_true, y_pred)}")
print(f"R^2:   {r2_score(y_true, y_pred)}")
print(f"MSLE:  {msle(y_true, y_pred)}")
print(f"RMSLE: {rmsle(y_true, y_pred)}")
print(f"AIC:   {aic(y_true, y_pred, k)}")
print(f"BIC:   {bic(y_true, y_pred, k)}")
