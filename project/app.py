import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
import datetime as dt

Starting = '2009-01-01'
Ending = dt.datetime.now()

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'MSFT')
df = data.DataReader(user_input, 'yahoo', Starting, Ending)


st.subheader('Data of last 10 days')
st.write(df.tail(10))


st.subheader('Price vs Time')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)



training=pd.DataFrame(df['Close'][0:int(len(df)*0.80)])
testing=pd.DataFrame(df['Close'][int(len(df)*0.80):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

training_array=scaler.fit_transform(training)

pred_days=100



x_train=[]
y_train=[]

for i in range(pred_days,training_array.shape[0]):
    x_train.append(training_array[i-pred_days:i])
    y_train.append(training_array[i,0])
    
x_train,y_train=np.array(x_train),np.array(y_train)



model=load_model('C:/Users/Subhojit/Desktop/project/keras_model.h5')

past100Days_data = training.tail(100)
final = past100Days_data.append(testing, ignore_index=True)
Input = scaler.fit_transform(final)

x_test = []
y_test = []

for i in range(pred_days, Input.shape[0]):
    x_test.append(Input[i-pred_days: i])
    y_test.append(Input[i, 0])
    
x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)

scaler=scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted*scale_factor
y_test = y_test * scale_factor


st.subheader('Predicted Value')
fig_pred=plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig_pred)
