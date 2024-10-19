import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Load your trained models
car_model = pickle.load(open('CarLinearRegressionModel.sav', 'rb'))
laptop_model = pickle.load(open('pipe.pkl', 'rb'))  # Assuming this is your laptop model

# Load datasets for dropdown options
car_df = pd.read_csv('Cleaned Car.csv')
laptop_df = pickle.load(open('df.pkl', 'rb'))  # Assuming this is the DataFrame for laptop options

# Sidebar for commodity selection
st.sidebar.header('Commodity Price Prediction')
commodity = st.sidebar.selectbox('Select Commodity', ['Car', 'Laptop'])

if commodity == 'Car':
    st.title('Car Price Prediction')

    # Car features input
    company = st.selectbox('Select Car Company', car_df['company'].unique())
    model_name = st.selectbox('Select Car Model', car_df[car_df['company'] == company]['name'].unique())
    year = st.selectbox('Select Year', sorted(car_df['year'].unique(), reverse=True))
    fuel_type = st.selectbox('Select Fuel Type', car_df['fuel_type'].unique())
    kms_driven = st.number_input('Enter Kilometers Driven', min_value=0, step=1)

    # Predict button for car price
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

        # Manually transform input data
        try:
            preprocessor = ColumnTransformer(transformers=[
                ('cat', OneHotEncoder(categories=car_model.named_steps['columntransformer'].transformers_[0][1].categories_), 
                 ['name', 'company', 'fuel_type']),
                ('num', 'passthrough', ['year', 'kms_driven'])
            ])
            transformed_input = preprocessor.fit_transform(input_data)
            prediction = car_model.named_steps['linearregression'].predict(transformed_input)
            st.success(f'The predicted price for the car is: ₹{prediction[0]:,.2f}')
        except Exception as e:
            st.error(f'Error in prediction: {e}')

elif commodity == 'Laptop':
    st.title('Laptop Price Prediction')

  # Laptop features input
    company = st.selectbox('Brand', laptop_df['Company'].unique())
    type = st.selectbox('Type', laptop_df['TypeName'].unique())
    ram = st.selectbox('RAM(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
    weight = st.number_input('Weight of the Laptop')
    touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
    ips = st.selectbox('IPS', ['No', 'Yes'])
    screen_size = st.slider('Screen size in inches', 10.0, 18.0, 13.0)
    resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160',
                                                    '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])
    cpu = st.selectbox('CPU', laptop_df['Cpu brand'].unique())
    hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])
    ssd = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])
    gpu = st.selectbox('GPU', laptop_df['Gpu brand'].unique())
    os = st.selectbox('OS', laptop_df['os'].unique())

    # Predict button for laptop price
    if st.button('Predict Laptop Price'):
        # Prepare PPI and other features
        touchscreen = 1 if touchscreen == 'Yes' else 0
        ips = 1 if ips == 'Yes' else 0
        X_res = int(resolution.split('x')[0])
        Y_res = int(resolution.split('x')[1])
        ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

        # Prepare input as a DataFrame
        query = pd.DataFrame({
            'Company': [company],
            'TypeName': [type],
            'Ram': [ram],
            'Weight': [weight],
            'Touchscreen': [touchscreen],
            'Ips': [ips],
            'ppi': [ppi],  # Make sure this is 'ppi' and not 'PPI'
            'Cpu brand': [cpu],
            'HDD': [hdd],
            'SSD': [ssd],
            'Gpu brand': [gpu],
            'os': [os]
        })

        try:
            # Use the model to predict
            predicted_price = laptop_model.predict(query)
            st.success(f"The predicted price of this laptop configuration is ₹{int(np.exp(predicted_price[0])):,}")
        except Exception as e:
            st.error(f"Prediction error: {e}")
