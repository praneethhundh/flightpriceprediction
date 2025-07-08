import streamlit as st
import pandas as pd
import pickle

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

st.title("✈️ Flight Price Prediction")

# Input fields (removed No_of_Bookings)
passenger_traffic = st.number_input("Passenger Traffic", min_value=0)
no_of_flights = st.number_input("Number of Flights", min_value=0)
no_of_seats = st.number_input("Number of Seats", min_value=0)

if st.button("Predict Price"):
    input_df = pd.DataFrame([[passenger_traffic, no_of_flights, no_of_seats]],
                            columns=['Passenger_Traffic', 'No_of_Flights', 'No_of_Seats'])
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Price: ₹ {int(prediction)}")
