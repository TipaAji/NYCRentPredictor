import joblib
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

data = pd.read_csv('streeteasy.csv')

model = joblib.load('random_forest_model.pkl')
model_metrics = joblib.load('model_metrics.pkl')

st.title("NYC Rent Prediction")

st.write("## Model Accuracy Metrics")
st.write(f"Prediction Accuracy: {model_metrics['scores']}")
st.write(f"Mean Absolute Error: {model_metrics['mae']}")

neighborhood = st.selectbox("Neighborhood", options=sorted(data['neighborhood'].unique()))
bedrooms = st.slider("Bedrooms", min_value=0, max_value=5, step=1, value=1)
bathrooms = st.slider("Bathrooms", min_value=0, max_value=5, step=1, value=1)
size = st.number_input("Size in Square Feet", min_value=100, max_value=5000, step=10)
min_to_subway = st.number_input("Minutes to Subway", min_value=0, max_value=60, step=1)
floor = st.slider("Floor Level", min_value=0, max_value=50, step=1)
building_age = st.slider("Building Age (years)", min_value=0, max_value=150, step=1)
has_roofdeck = st.checkbox("Has Roofdeck")
has_washer_dryer = st.checkbox("Has Washer/Dryer")
has_doorman = st.checkbox("Has Doorman")
has_elevator = st.checkbox("Has Elevator")
has_dishwasher = st.checkbox("Has Dishwasher")
has_patio = st.checkbox("Has Patio")
has_gym = st.checkbox("Has Gym")

if st.button("Predict Rent"):
    user_data = pd.DataFrame({
        'neighborhood': [neighborhood],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'size_sqft': [size],
        'min_to_subway': [min_to_subway],
        'floor': [floor],
        'building_age_yrs': [building_age],
        'has_roofdeck': [int(has_roofdeck)],
        'has_washer_dryer': [int(has_washer_dryer)],
        'has_doorman': [int(has_doorman)],
        'has_elevator': [int(has_elevator)],
        'has_dishwasher': [int(has_dishwasher)],
        'has_patio': [int(has_patio)],
        'has_gym': [int(has_gym)]
    })

    predicted_rent = model.predict(user_data)[0]

    st.write(f"### Estimated Rent Price: ${predicted_rent:.2f} per month")

st.write("### Data Visualizations")
sample_data = data.sample(n=100, random_state=42)

# Average Rent by Neighborhood
fig, ax = plt.subplots(figsize=(15, 8))
avg_rent_by_neighborhood = sample_data.groupby('neighborhood')['rent'].mean().sort_values()
avg_rent_by_neighborhood.plot(kind='bar', ax=ax, color='skyblue')
ax.set_title("Average Rent by Neighborhood")
ax.set_xlabel("Neighborhood")
ax.set_ylabel("Average Rent ($)")
plt.xticks(rotation=45)
st.pyplot(fig)

# Scatter Plot: Rental Size vs Rent
fig, ax = plt.subplots()
ax.scatter(sample_data['size_sqft'], sample_data['rent'], alpha=0.5, color='coral')
ax.set_title("Rental Size vs Rent")
ax.set_xlabel("Size (sqft)")
ax.set_ylabel("Rent ($)")
st.pyplot(fig)

# Histogram: Distribution of Rent
fig, ax = plt.subplots()
sample_data['rent'].plot(kind='hist', bins=30, color='purple', ax=ax)
ax.set_title("Distribution of Rent")
ax.set_xlabel("Rent ($)")
st.pyplot(fig)