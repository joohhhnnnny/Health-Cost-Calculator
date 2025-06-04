import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Sample dataset (you can replace this with your actual CSV or DataFrame)
data = pd.DataFrame({
    'age': [18, 25, 35, 45, 50, 60],
    'bmi': [22.0, 27.5, 30.1, 25.3, 28.5, 33.0],
    'children': [0, 1, 2, 1, 2, 3],
    'charges': [1200, 2500, 3200, 4000, 5000, 6200]
})

X = data[['age', 'bmi', 'children']]
y = data['charges']

# Split and scale the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Expose variables
train_dataset = X_train_scaled
test_dataset = X_test_scaled
train_labels = y_train
test_labels = y_test
