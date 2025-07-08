import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load dataset
df = pd.read_csv("flight_price_dataset.csv")
X = df[['Passenger_Traffic', 'No_of_Flights', 'No_of_Seats']]  # Removed 'No_of_Bookings'
y = df['Flight_Price']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved without 'No_of_Bookings'")
