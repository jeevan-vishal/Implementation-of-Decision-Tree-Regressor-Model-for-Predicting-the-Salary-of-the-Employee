# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm 

1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. calculate Mean square error,data prediction and r2.

## Program:

```

Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Jeevan Vishal.G.D
RegisterNumber: 212224240062

```
```
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt


df = pd.read_csv(r"C:\Users\admin\Downloads\Salary.csv")


print("Salary Data:")

print(df)
print()


X = df[['Level']]  
y = df['Salary']    


model = DecisionTreeRegressor()
model.fit(X, y)


predictions = model.predict(X)


print("Actual vs Predicted Salaries:")
for i in range(len(df)):
    print(f"Level {df.iloc[i]['Level']}: Actual=${df.iloc[i]['Salary']}, Predicted=${int(predictions[i])}")


accuracy = model.score(X, y)
print(f"\nModel Accuracy (R² Score): {accuracy:.2f}")


new_level = [[6.5]]
predicted_salary = model.predict(new_level)
print(f"\nPredicted Salary for Level 6.5: ${int(predicted_salary[0])}")


plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='blue', label='Actual Data', s=100)
plt.plot(X, predictions, color='red', label='Decision Tree Predictions', linewidth=2)
plt.xlabel('Level')
plt.ylabel('Salary ($)')
plt.title('Salary Prediction using Decision Tree')
plt.legend()
plt.grid(True)
plt.show()
```

## Output:

Code :

<img width="942" height="805" alt="image" src="https://github.com/user-attachments/assets/db920e42-a7cd-4482-80f7-7d533daf6911" />


Salary Data :

<img width="367" height="271" alt="image" src="https://github.com/user-attachments/assets/9715257c-01cc-468f-bc43-02e2738bded5" />


Actual vs Predicted :

<img width="492" height="262" alt="image" src="https://github.com/user-attachments/assets/1f0d444e-097f-48ae-87e3-cd19d73e3d15" />


Accuracy :

<img width="393" height="78" alt="image" src="https://github.com/user-attachments/assets/2b50a5d7-de05-4efa-a8f2-076fe51ab688" />


Decision Tree :

<img width="941" height="591" alt="image" src="https://github.com/user-attachments/assets/9a25dd50-c6ed-47d3-8310-2b41d4b124a5" />



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
