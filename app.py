import streamlit as st
import pickle
import numpy as np

regressor = pickle.load(open('regressor.pkl', 'rb'))
laptop = pickle.load(open('laptoporig.pkl', 'rb'))

st.title("Laptop price predictor")

Company = st.selectbox('Brand', laptop['Company'].unique())
TypeName = st.selectbox('Type', laptop['TypeName'].unique())
Inches = st.selectbox('Inches', laptop['Inches'].unique())
Cpu = st.selectbox('Cpu', laptop['Cpu'].unique())
Memory = st.selectbox('Memory', laptop['Memory'].unique())
Gpu = st.selectbox('Gpu', laptop['Gpu'].unique())
OpSys = st.selectbox('OpSys', laptop['OpSys'].unique())
Weight = st.selectbox('Weight', laptop['Weight'].unique())
Ram = st.selectbox('RAM(in GB)', [2, 4, 6, 8, 10, 12, 16, 24, 32, 64])
Touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
Ips = st.selectbox('Ips', laptop['Ips'].unique())
selected_features = ['Company', 'TypeName', 'Inches', 'Cpu', 'Ram', 'Memory', 'Gpu', 'OpSys', 'Weight', 'Touchscreen',
                     'Ips']

# Create a subset of the DataFrame with only the selected columns
input_data = laptop[selected_features]

# Now, you can use the selected input data for prediction
if st.button('Predict Price'):
    if Touchscreen == 'Yes':
        Touchscreen = 1
    else:
        Touchscreen = 0

    # You should pass only the selected features for prediction
    query = np.array([Company, TypeName, Inches, Cpu, Ram, Memory, Gpu, OpSys, Weight, Touchscreen, Ips])
    query = query.reshape(1, 11)

    predicted_price = regressor.predict(query)

    print("query:", query)

    # Display the predicted price
    st.title(f'Predicted Price: {predicted_price[0]}')
