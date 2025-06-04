# main.py

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from hcc import data

np.random.seed(42)
tf.random.set_seed(42)

df = pd.DataFrame(data)
categorical_cols = ['sex', 'smoker', 'region']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

x = df.drop('expenses', axis=1)
y = df['expenses']

train_dataset, test_dataset, train_labels, test_labels = train_test_split(
    x, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
train_dataset = scaler.fit_transform(train_dataset)
test_dataset = scaler.transform(test_dataset)

# Model with Dropout and EarlyStopping
def build_model():
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(train_dataset.shape[1],)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
    return model

model = build_model()

# Add EarlyStopping
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    train_dataset,
    train_labels,
    validation_split=0.2,
    epochs=200,
    batch_size=8,
    verbose=1,
    callbacks=[early_stop]
)

loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=2)
print(f"Mean Absolute Error on test set: {mae:.2f}")

if mae < 3500:
    print("Challenge passed! Your model predicts healthcare costs within $3500 accuracy.")
else:
    print("Model needs improvement. Try tuning or adding more data.")

test_predictions = model.predict(test_dataset).flatten()

plt.figure(figsize=(8, 6))
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Expenses')
plt.ylabel('Predicted Expenses')
plt.title('True vs Predicted Healthcare Expenses')
plt.plot([min(test_labels), max(test_labels)], [min(test_labels), max(test_labels)], 'r')
plt.show()
