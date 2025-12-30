import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the data
df = pd.read_csv('dynamic_pricing (2).csv')

# Select features for the model
features = ['Number_of_Riders', 'Number_of_Drivers', 'Number_of_Past_Rides', 'Average_Ratings', 'Expected_Ride_Duration']
X = df[features]
y = df['Historical_Cost_of_Ride']

# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = r2_score(y_test, y_pred)

# Function to predict price
def predict_price(riders, drivers, past_rides, rating, duration):
    input_data = [[riders, drivers, past_rides, rating, duration]]
    return model.predict(input_data)[0]

# Print accuracy when model is loaded
print(f"Model Accuracy: {accuracy:.4f}")