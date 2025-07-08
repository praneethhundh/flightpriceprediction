import streamlit as st
import pandas as pd
import pickle
import os

# 🔄 Auto-train model if not available
if not os.path.exists("model.pkl"):
    from train_model import train_model
    train_model()

# ✅ Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("✈️ Flight Price Prediction")

passenger_traffic = st.number_input("Passenger Traffic", min_value=0)
no_of_flights = st.number_input("Number of Flights", min_value=0)
no_of_seats = st.number_input("Number of Seats", min_value=0)

if st.button("Predict Price"):
    df_input = pd.DataFrame([[passenger_traffic, no_of_flights, no_of_seats]],
                            columns=['Passenger_Traffic', 'No_of_Flights', 'No_of_Seats'])
    prediction = model.predict(df_input)[0]
    st.success(f"Predicted Flight Price: ₹ {int(prediction)}")
