import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib


df = pd.read_csv("Student_Performance.csv")


df = pd.get_dummies(df, drop_first=True)

X = df.drop("Performance Index", axis=1)
y = df["Performance Index"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, pred))
print("R2:", r2_score(y_test, pred))


joblib.dump(model, "student_model.pkl")


plt.figure()
plt.scatter(y_test, pred)
plt.plot([y_test.min(), y_test.max()],
         [pred.min(), pred.max()], 'r--')
plt.xlabel("Real Performance Index")
plt.ylabel("Predicted Performance Index")
plt.title("Real vs Predicted Values")

plt.show()