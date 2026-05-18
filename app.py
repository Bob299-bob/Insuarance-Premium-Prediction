import streamlit as st
import numpy as np
import joblib
model=joblib.load('model.pkl')
encoder=joblib.load('Hot.pkl')

result=st.empty()
st.title('Insurance Charges Prediction')
age=st.number_input('Enter Your Age..')
sex=st.selectbox('Select Gender',['male','female'])
bmi=st.number_input('Enter Your Body mass index')
child=st.number_input('No. of Childrens')
smoker=st.radio('Are you smoker ?',['yes','no'])
region=st.selectbox('Enter your region',['southwest','northwest','southeast','northeast'])
prev_data=[age,bmi,child]
if(st.button('Predict')):
    data=[sex,smoker,region]
    encode_data=encoder.transform([data])
    prev_data = np.array(prev_data).reshape(1, -1)
    encode_data = np.array(encode_data).reshape(1, -1)
    final_data = np.concatenate((prev_data, encode_data), axis=1)
    Y_pred=model.predict(final_data)
    st.session_state.page='pred'
    result.success(f"Estimated Charge is : {Y_pred[0]:.2f}")
st.markdown("""
        <style>
            .stApp{
                background-color:#0f172a;
            }
        </style>
""",unsafe_allow_html=True
)