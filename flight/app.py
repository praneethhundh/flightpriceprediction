import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load dataset directly
df = pd.read_csv("flight_price_dataset.csv")

# Train model directly inside app
X = df[['Passenger_Traffic', 'No_of_Flights', 'No_of_Seats']]
y = df['Flight_Price']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

st.title("✈️ Flight Price Prediction")

# User input
passenger_traffic = st.number_input("Passenger Traffic", min_value=0)
no_of_flights = st.number_input("Number of Flights", min_value=0)
no_of_seats = st.number_input("Number of Seats", min_value=0)

# Prediction
if st.button("Predict Price"):
    input_data = pd.DataFrame([[passenger_traffic, no_of_flights, no_of_seats]],
                               columns=['Passenger_Traffic', 'No_of_Flights', 'No_of_Seats'])
    predicted_price = model.predict(input_data)[0]
    st.success(f"Predicted Flight Price: ₹ {int(predicted_price)}")
