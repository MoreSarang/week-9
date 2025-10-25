import streamlit as st
import pandas as pd
import numpy as np
from apputil import GroupEstimate


st.write(
'''
This app demonstrates the GroupEstimate class which predicts values 
based on categorical groupings.
''')

# Sample data
st.write("### Sample Coffee Review Data")

# Create example dataset
sample_data = {
    'loc_country': ['Guatemala', 'Guatemala', 'Mexico', 'Mexico', 
                    'Ethiopia', 'Ethiopia', 'Guatemala', 'Mexico'],
    'roast': ['Light', 'Light', 'Medium', 'Medium', 
              'Light', 'Dark', 'Light', 'Medium'],
    'rating': [88.0, 89.0, 91.0, 90.5, 92.0, 93.0, 88.2, 91.5]
}

df_raw = pd.DataFrame(sample_data)
st.dataframe(df_raw)

# User inputs
st.write("### Model Configuration")

estimate_type = st.selectbox(
    "Select estimate type:",
    ["mean", "median"]
)

# Fit the model
X = df_raw[["loc_country", "roast"]]
y = df_raw["rating"]

gm = GroupEstimate(estimate=estimate_type)
gm.fit(X, y)

st.success(f"Model fitted using {estimate_type} estimation!")

# Prediction interface
st.write("### Make Predictions")

country = st.selectbox(
    "Select country:",
    ["Guatemala", "Mexico", "Ethiopia", "Canada"]
)

roast = st.selectbox(
    "Select roast type:",
    ["Light", "Medium", "Dark"]
)

if st.button("Predict Rating"):
    X_new = [[country, roast]]
    prediction = gm.predict(X_new)
    
    if np.isnan(prediction[0]):
        st.error(f"No data available for {country} - {roast} combination")
    else:
        st.success(f"Predicted rating: {prediction[0]:.2f}")

# Batch prediction example
st.write("### Batch Prediction Example")

batch_input = [
    ["Guatemala", "Light"],
    ["Mexico", "Medium"],
    ["Canada", "Dark"]
]

if st.button("Run Batch Prediction"):
    predictions = gm.predict(batch_input)
    
    results_df = pd.DataFrame({
        'Country': [x[0] for x in batch_input],
        'Roast': [x[1] for x in batch_input],
        'Predicted Rating': predictions
    })
    
    st.dataframe(results_df)
