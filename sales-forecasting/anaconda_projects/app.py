
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load('model.pkl')

# Load data
df = pd.read_csv('sales_data.csv')
df['Order Date'] = pd.to_datetime(df['Order Date'], format='%m/%d/%y')

# Preprocess
monthly_sales = df.groupby(df['Order Date'].dt.to_period('M'))['Sales'].sum().reset_index()
monthly_sales['Date'] = monthly_sales['Order Date'].dt.to_timestamp()
monthly_sales['Previous_Sales'] = monthly_sales['Sales'].shift(1)
monthly_sales = monthly_sales.dropna()

# Predict
latest = monthly_sales['Previous_Sales'].iloc[-1]
pred = model.predict([[latest]])[0]

# Streamlit app layout
st.title("📈 Sales Forecasting App")
st.write("Predicting next month's sales based on past data.")

# Display prediction
st.metric("📊 Predicted Next Month's Sales", f"${pred:,.2f}")

# Plot
st.subheader("📉 Monthly Sales Trend")
fig, ax = plt.subplots()
ax.plot(monthly_sales['Date'], monthly_sales['Sales'], label='Actual')
ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.set_title("Monthly Sales")
st.pyplot(fig)
