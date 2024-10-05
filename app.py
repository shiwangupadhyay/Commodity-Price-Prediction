import streamlit as st
import pandas as pd
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Load your trained model (pipeline with OneHotEncoder and LinearRegression)
model = pickle.load(open('LinearRegressionModel.sav', 'rb'))

# Load dataset for dropdown options (this dataset is only for fetching dropdown options)
df = pd.read_csv('Cleaned Car.csv')

# Sidebar for car price prediction
st.sidebar.header('Car Price Prediction')
st.title('Car Price Prediction App')
# st.write(model)  This will print out the components of the pipeline

# Car features input
company = st.selectbox('Select Car Company', df['company'].unique())
model_name = st.selectbox('Select Car Model', df[df['company'] == company]['name'].unique())
year = st.selectbox('Select Year', sorted(df['year'].unique(), reverse=True))
fuel_type = st.selectbox('Select Fuel Type', df['fuel_type'].unique())
kms_driven = st.number_input('Enter Kilometers Driven', min_value=0, step=1)

# Predict button
if st.button('Predict Car Price'):
    # Prepare input as a DataFrame (this format matches the training data)
    year = int(year)
    kms_driven = int(kms_driven)
    input_data = pd.DataFrame({
        'name': [model_name],
        'company': [company],
        'year': [year],
        'kms_driven': [kms_driven],
        'fuel_type': [fuel_type]
    })
    
    st.write("Input Data for Prediction:")
    st.write(input_data)  # Show the input data to check formatting
    
    # Manually transform input data
    try:
        preprocessor = ColumnTransformer(transformers=[
            ('cat', OneHotEncoder(categories=model.named_steps['columntransformer'].transformers_[0][1].categories_), 
             ['name', 'company', 'fuel_type']),
            ('num', 'passthrough', ['year', 'kms_driven'])
        ])
        transformed_input = preprocessor.fit_transform(input_data)
        prediction = model.named_steps['linearregression'].predict(transformed_input)
        st.success(f'The predicted price for the car is: ₹{prediction[0]:,.2f}')
    except Exception as e:
        st.error(f'Error in prediction: {e}')
