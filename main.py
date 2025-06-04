from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from hcc import train_dataset, test_dataset, train_labels, test_labels
import matplotlib.pyplot as plt

# Initialize and train model
model = LinearRegression()
model.fit(train_dataset, train_labels)

# Predict on test set
test_predictions = model.predict(test_dataset)

# Evaluate the model
mae = mean_absolute_error(test_labels, test_predictions)
mse = mean_squared_error(test_labels, test_predictions)
r2 = r2_score(test_labels, test_predictions)

# Print evaluation results
print(f"\nMean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.4f}\n")

# Challenge logic
print("Testing set Mean Abs Error: {:5.2f} expenses".format(mae))
if mae < 3500:
    print("You passed the challenge. Great job!")
else:
    print("The Mean Abs Error must be less than 3500. Keep trying.")

# Print individual predictions vs actual
for i in range(len(test_predictions)):
    print(f"Predicted: ${test_predictions[i]:,.2f} | Actual: ${test_labels.iloc[i]:,.2f}")

# Plot predictions
a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True values (expenses)')
plt.ylabel('Predictions (expenses)')
lims = [0, 50000]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.show()

