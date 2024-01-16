
import pickle
import streamlit as st

# Loading the model
winequality_model = pickle.load(open('C:/Users/pc/Downloads/Wine prediction/winequality_model.sav', 'rb'))

# page title
st.title('Wine Quality Prediction App')

# getting the input data from the user
col1, col2, col3 = st.columns(3)

with col1:
    FixedAcidity = st.text_input('Fixed Acidity')

with col2:
    VolatileAcidity = st.text_input('Volatile Acidity')

with col3:
    CitricAcidity = st.text_input('Citric Acidity')

with col1:
    ResidualSugar = st.text_input('Residual Sugar Level')

with col2:
    FreeSulfurDioxide = st.text_input('Free Sulfur Dioxide')

with col3:
    Density = st.text_input('Density')

with col1:
    pH = st.text_input('pH Value')

with col2:
    Sulphate = st.text_input('Sulphate')

with col3:
    Alcohol = st.text_input('Alcohol')
    
with col1:
    Chloride = st.text_input('Chloride')

with col2:
    TotalSulfurDioxide = st.text_input('Total Sulfur Dioxide')

# code for Prediction
wine_quality = ''

# creating a button for Prediction
if st.button('Wine Quality Prediction Result'):
    wine_prediction = winequality_model.predict([[FixedAcidity, VolatileAcidity, CitricAcidity, ResidualSugar, FreeSulfurDioxide, Density, Sulphate, Alcohol, Chloride, TotalSulfurDioxide]])

    if wine_prediction[0] == 1:
        wine_quality = 'Good Quality Wine'
    else:
        wine_quality = 'Bad Quality Wine'

st.success(wine_quality)
