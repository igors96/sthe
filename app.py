import streamlit as st
import pandas as pd
import xgboost as xg
import pickle
from htbuilder import div, big, h2, styles
from htbuilder.units import rem

# load saved model
with open('xgb_regressor_model.pkl' , 'rb') as f:
   xgb_r = pickle.load(f)

# Getting heat exchanger data

def get_heat_exchanger_data():

    st.sidebar.markdown("# Insert heat exchanger parameters:")
    flow_hot = st.sidebar.slider('Flow of hot fluid (kg/s)', 1.0, 3.0, 2.0, 0.1)
    flow_cold = st.sidebar.slider('Flow of cold fluid (kg/s)', 1.0, 3.0, 2.0, 0.1)
    temp_hot = st.sidebar.slider('Temperature of hot fluid (ºC)', 40, 80, 40, 1)
    temp_cold = st.sidebar.slider('Temperature of cold fluid (ºC)', 5, 30, 5, 1)
    shell_passes = st.sidebar.slider('Shell passes', 1, 2, 1, 1)
    fouling_factor_tube = st.sidebar.slider('Fouling factor *10\u207B\u2074(tube, W/m2*K)', 0.9, 2.0, 1.5, 0.1)
    fouling_factor_shell = st.sidebar.slider('Fouling factor *10\u207B\u2074(shell, W/m2*K)', 0.9, 2.0, 1.5, 0.1)


    heat_exchanger_data = {'Flow_hot (kg/s)' : flow_hot,
                            'Flow_cold (kg/s)' : flow_cold,
                            'Temperature_hot (ºC)' : temp_hot,
                            'Temperature_cold (ºC)' : temp_cold,
                            'Number of passes' : shell_passes,
                            'Fouling Factor Tube (W/m2.K)' : fouling_factor_tube * 0.0001,
                            'Fouling Factor Shell (W/m2.K)' : fouling_factor_shell * 0.0001}
                        
    features = pd.DataFrame(heat_exchanger_data, index = [0])

    return features

heat_exchanger_input = get_heat_exchanger_data()

# Making predictions
prediction = xgb_r.predict(heat_exchanger_input)
rounded = list(map('{:.2f}%'.format, prediction))

# Creating parameters to use after
color = '#FF4D4C'
title = 'HEAT EXCHANGER THERMAL EFFICIENCY'
value = rounded

st.markdown('This application is very simple. The intention here is to provide the calculus of thermal efficiency of the shell and tube heat exchanger. Just input seven parameters in the left side: flows and temperatures of the hot and cold fluid, shell passes and the shell and tube fouling factors. Automatically will appear, below, the value of the thermal efficiency.')

st.markdown(
        div(
            style=styles(
                text_align="left",
                color = color,
                padding=(rem(1), 0, rem(1), 0),
            )
        )(
            h2(style=styles(font_size=rem(2), font_weight=800, padding=0))(title),
            big(style=styles(font_size=rem(4), font_weight=800, line_height=1.5))(
                value
            ),
        ),
        unsafe_allow_html=True,
    )
