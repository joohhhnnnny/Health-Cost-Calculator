from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from hcc import train_dataset, test_dataset, train_labels, test_labels

# Initialize and train model
model = LinearRegression()
model.fit(train_dataset, train_labels)

# Predict on test set
predictions = model.predict(test_dataset)

# Evaluate the model
mae = mean_absolute_error(test_labels, predictions)
mse = mean_squared_error(test_labels, predictions)
r2 = r2_score(test_labels, predictions)

print(f"\nMean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.4f}\n")

# Show predictions
for i in range(len(predictions)):
    print(f"Predicted: ${predictions[i]:,.2f} | Actual: ${test_labels.iloc[i]:,.2f}")
