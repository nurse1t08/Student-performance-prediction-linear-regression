import joblib
import pandas as pd

model = joblib.load("student_model.pkl")

print("Введите данные студента")

def input_number(prompt):
    while True:
        try:
            value = float(input(prompt))
            return value
        except ValueError:
            print("Ошибка! Введите цифру!")

hours = input_number("Сколько часов студент учился (Hours Studied): ")
previous = input_number("Предыдущие оценки (Previous Scores): ")
sleep = input_number("Сколько часов сна (Sleep Hours): ")
papers = input_number("Сколько тренировочных тестов решено: ")

while True:
    extra = input("Есть ли внеклассные занятия (yes/no): ").strip().lower()
    if extra in ["yes", "no"]:
        extra = 1 if extra == "yes" else 0
        break
    else:
        print("Ошибка! Введите 'yes' или 'no'.")

# ✅ создаём DataFrame ТУТ (после ввода)
X = pd.DataFrame([[hours, previous, sleep, papers, extra]],
columns=[
    'Hours Studied',
    'Previous Scores',
    'Sleep Hours',
    'Sample Question Papers Practiced',
    'Extracurricular Activities_Yes'
])

prediction = model.predict(X)

print("Предсказанный Performance Index:", prediction[0])