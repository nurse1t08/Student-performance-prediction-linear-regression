from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

app = Flask(__name__)

# Загружаем модель
model = joblib.load("student_model.pkl")

# Загружаем данные
df = pd.read_csv("Student_Performance.csv")

# Берём 10 первых строк для отображения
samples = df.head(10).values.tolist()
df_processed = pd.get_dummies(df, drop_first=True)
final_df = df_processed.drop("Performance Index", axis=1).head(10).values.tolist()

# Получаем заголовки колонок
sample_columns = df.columns.tolist()  # заголовки для samples
final_columns = df_processed.drop("Performance Index", axis=1).head(10).columns.tolist()  # заголовки для final_df

# Для метрик
X = df_processed.drop("Performance Index", axis=1)
y = df_processed["Performance Index"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

# Строим график
plt.figure()
plt.scatter(y_test, pred)
plt.plot([y_test.min(), y_test.max()],
         [pred.min(), pred.max()], 'r--')
plt.xlabel("Real")
plt.ylabel("Predicted")
plt.title("Train Graph")
plt.savefig("static/graph.png")
plt.close()

@app.route("/")
def home():
    return render_template(
        "index.html",
        samples=samples,
        final_df=final_df,
        sample_columns=sample_columns,
        final_columns=final_columns,
        mae=round(mae,2),
        r2=round(r2,2),
        result=None
    )

@app.route("/predict", methods=["POST"])
def predict():
    hours = float(request.form["hours"])
    previous = float(request.form["previous"])
    sleep = float(request.form["sleep"])
    papers = float(request.form["papers"])
    extra = int(request.form["extra"])

    X_input = np.array([[hours, previous, sleep, papers, extra]])
    prediction = model.predict(X_input)

    return render_template(
        "index.html",
        result=round(prediction[0],2),
        samples=samples,
        final_df=final_df,
        sample_columns=sample_columns,
        final_columns=final_columns,
        mae=round(mae,2),
        r2=round(r2,2)
    )

if __name__ == "__main__":
    app.run(debug=True)