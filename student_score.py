import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor  
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


data = {
    'StudyHours': [2, 3, 4, 5, 6, 7, 8, 9, 10, 12],  # Hours studied
    'AssignmentsCompleted': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Number of assignments completed
    'Score': [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]  # Student score
}
df = pd.DataFrame(data)
print("Sample Data:")
print(df.head())

X = df[['StudyHours', 'AssignmentsCompleted']]  # Features: study hours, assignments
y = df['Score']  # Score to predict

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = KNeighborsRegressor(n_neighbors=3)  # Using 3 nearest neighbors
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")

print("\nScore predictions on test set:")
results = pd.DataFrame({'Actual Score': y_test, 'Predicted Score': y_pred})
print(results)

plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual Score')
plt.plot(range(len(y_pred)), y_pred, color='red', label='Predicted Score')
plt.xlabel('Sample')
plt.ylabel('Score')
plt.title('Comparison of Actual vs Predicted Scores (KNN)')
plt.legend()
plt.show()

import joblib
joblib.dump(model, 'student_score_knn_model.pkl')
print("Model saved to 'student_score_knn_model.pkl'")