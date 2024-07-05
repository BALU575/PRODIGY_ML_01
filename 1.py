import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Step 2: Generate a synthetic dataset
np.random.seed(42)
num_samples = 100

square_footage = np.random.randint(500, 5000, num_samples)
num_bedrooms = np.random.randint(1, 6, num_samples)
num_bathrooms = np.random.randint(1, 4, num_samples)
prices = (square_footage * 300) + (num_bedrooms * 10000) + (num_bathrooms * 5000) + np.random.randint(-10000, 10000, num_samples)

data = pd.DataFrame({
    'SquareFootage': square_footage,
    'NumBedrooms': num_bedrooms,
    'NumBathrooms': num_bathrooms,
    'Price': prices
})

# Step 3: Preprocess Data
X = data[['SquareFootage', 'NumBedrooms', 'NumBathrooms']]
y = data['Price']

# Step 4: Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Evaluate the Model
y_pred = model.predict(X_test)

# Calculate mean squared error and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Step 7: Make Predictions
# Example prediction
example_house = np.array([[2500, 3, 2]])
predicted_price = model.predict(example_house)

print(f'Predicted price for house with 2500 square feet, 3 bedrooms, and 2 bathrooms: ${predicted_price[0]:.2f}')


