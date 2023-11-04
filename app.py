import streamlit as st
import pickle
import numpy as np

pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df1.pkl', 'rb'))

st.title("Laptop price predictor")

Company = st.selectbox('Brand', df['Company'].unique())
TypeName = st.selectbox('Type', df['TypeName'].unique())
Inches = st.selectbox('Inches', df['Inches'].unique())
Ram = st.selectbox('RAM(in GB)', [2, 4, 6, 8, 10, 12, 16, 24, 32, 64])
Weight = st.selectbox('Weight', df['Weight'].unique())
Touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
Ips = st.selectbox('Ips	', df['Ips'].unique())
gpu = st.selectbox('Gpu', df['gpu'].unique())
cpu = st.selectbox('Cpu', df['cpu'].unique())
HDD	 = st.selectbox('HDD', df['HDD'].unique())
SSD = st.selectbox('SSD', df['SSD'].unique())
Hybrid =  st.selectbox('Hybrid',df['Hybrid'].unique())
Flash_Storage = st.selectbox('Flash_Storage', df['Flash_Storage'].unique())
os = st.selectbox('os', df['os'].unique())
selected_features =['Company', 'TypeName', 'Inches', 'Ram', 'Weight', 'Touchscreen', 'Ips', 'gpu', 'cpu', 'HDD', 'SSD', 'Hybrid', 'Flash_Storage', 'os']

# Create a subset of the DataFrame with only the selected columns
input_data = df[selected_features]

# Now, you can use the selected input data for prediction
if st.button('Predict Price'):
    if Touchscreen == 'Yes':
        Touchscreen = 1
    else:
        Touchscreen = 0

        if Ips == 'Yes':
            Ips = 1
        else:
            Ips = 0

            # You should pass only the selected features for prediction
    query = np.array([Company, TypeName, Inches, Ram, Weight, Touchscreen, Ips, gpu, cpu, HDD, SSD, Hybrid, Flash_Storage, os])
    query = query.reshape(1, 14)

    predicted_price =np.exp(pipe.predict(query))

    print("query:", query)

    # Display the predicted price
    st.title(f'Predicted Price: {predicted_price[0]}')
