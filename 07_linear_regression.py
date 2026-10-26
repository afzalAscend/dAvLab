import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

data = {
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Attendance": [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
    "Assignment_Marks": [10, 12, 15, 16, 18, 20, 22, 24, 26, 28],
    "Marks": [35, 40, 45, 50, 55, 60, 65, 70, 80, 90]
}

df = pd.DataFrame(data)
print("Dataset:")
print(df)

X_simple = df[["Study_Hours"]]
X_multiple = df[["Study_Hours", "Attendance", "Assignment_Marks"]]
y = df["Marks"]

X_train_s, X_test_s, y_train, y_test = train_test_split(
    X_simple, y, test_size=0.2, random_state=42
)

X_train_m, X_test_m, _, _ = train_test_split(
    X_multiple, y, test_size=0.2, random_state=42
)

simple_model = LinearRegression()
simple_model.fit(X_train_s, y_train)
simple_predictions = simple_model.predict(X_test_s)

print("\nSimple Linear Regression")
print("Intercept:", simple_model.intercept_)
print("Coefficient:", simple_model.coef_[0])
print("R2 Score:", r2_score(y_test, simple_predictions))
print("RMSE:", np.sqrt(mean_squared_error(y_test, simple_predictions)))

multiple_model = LinearRegression()
multiple_model.fit(X_train_m, y_train)
multiple_predictions = multiple_model.predict(X_test_m)

print("\nMultiple Linear Regression")
print("Intercept:", multiple_model.intercept_)
print("Coefficients:", multiple_model.coef_)
print("R2 Score:", r2_score(y_test, multiple_predictions))
print("RMSE:", np.sqrt(mean_squared_error(y_test, multiple_predictions)))

new_student_simple = pd.DataFrame({"Study_Hours": [6]})
new_student_multiple = pd.DataFrame({
    "Study_Hours": [6],
    "Attendance": [80],
    "Assignment_Marks": [22]
})

print("\nPredicted marks using Simple Regression:",
      simple_model.predict(new_student_simple)[0])
print("Predicted marks using Multiple Regression:",
      multiple_model.predict(new_student_multiple)[0])

plt.scatter(df["Study_Hours"], df["Marks"], color="blue", label="Actual Marks")
plt.plot(df["Study_Hours"], simple_model.predict(X_simple),
         color="red", label="Regression Line")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()