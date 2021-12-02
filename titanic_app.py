import streamlit as st
import pandas as pd
import numpy as np
import pickle as pk
import os

# components
from config import config
from preprocessing.data_management import load_pipeline

def main():
    """Simple EDA App"""
    st.title("Titanic web App with streamlit")
    st.subheader("Fill the form below please: ")
    st.write("At its time, the Titanic was both a source a great achievment and a tragedy. Today,\
                we could still learn from it.")

    # importing the sample data:
    df = pd.read_csv(config.TRAIN_FILE)

    # Age
    min_age = df['age'].min()
    max_age = df['age'].max()
    age = st.number_input('select your age', min_value=min_age, max_value=max_age)
    
    st.write(f'the min age is {min_age}')
    st.write(f'the min age is {max_age}')

    # selecting pclass
    col1, col2 = st.columns(2)

    with col1:
        pclass = st.selectbox("Select your class", (1, 2, 3))
        #msg = 'You selected the {} class'
        #if pclass == 1:
        #    st.write(msg.format('First'))
        #elif pclass == 2:
        #    st.write(msg.format('middle'))
        #else:
        #    st.write(msg.format('Third'))

    # Select your sex and title:
    with col2:
        sex = st.selectbox("Select your gender", ('male', 'female'))
        if sex =='female':
            title = 'Miss'
            #st.write('You are a Miss')
        else:
            title = 'Mr'
            #st.write('You are a Mr')
    
    # sibsp and parch
    col3, col4 = st.columns(2)

    with col3:
        st.write('\n')
        sibsp = st.slider('How many brothers and sisters on the boat?', min_value=0, max_value=10, key=1)
    with col4:
        #st.write('\n')
        parch = st.slider('How many parents (fathers and mothers) have you on the boat?', min_value=0, max_value=10, key=2)
    
    # fare and cabin
    col5, col6 = st.columns(2)

    with col5:
        #st.write('\n')
        min_fare = df.loc[df['pclass']==pclass,'fare'].min()
        max_fare = df.loc[df['pclass']==pclass,'fare'].max()

        fare = st.number_input('select the fare you can afford', 
                                min_value=float(min_fare),
                                max_value=float(max_fare), 
                                key = 1) # number_inpu identifier

        st.write(f'min fare associated with your class: {min_fare}')
        st.write(f'min fare associated with your class: {max_fare}')

    with col6:
        #st.write('\n')
        cabin_list = df['cabin'].dropna().unique().tolist()
        cabin = st.selectbox('Select your cabin', cabin_list)

    # Embarked
    dport_list = ['Cherbourg', 'Queenstown', 'Southampton']
    dport = st.selectbox('From which port do you go?', dport_list)
    if dport == dport_list[0]:
        embarked = 'C'
    elif dport == dport_list[1]:
        embarked = 'Q'
    else:
        embarked = 'S'
    
    # recap
    st.subheader('So here what you entered:')
    _input = np.array([pclass, sex, age, sibsp, parch,fare, cabin, embarked, title])
    input_df = pd.DataFrame(_input.reshape(1,-1), columns=config.KEEP_FEATURES)
    st.dataframe(input_df)

    # make a prediction
    # Set the saved model directory
    model_dir = config.SAVED_MODEL_PATH

    # list all model names
    last_model_name = os.listdir(model_dir)

    # Set the model path
    pipeline_file_name = os.path.join(model_dir, last_model_name[-1])

    # Load the model
    _titanic_pipe = load_pipeline(pipeline_file_name)

    if _titanic_pipe:
        st.subheader("So would you survive?")
        if _titanic_pipe.predict(input_df):
            st.success('yes!! absolutely!!!')
        else:
            st.warning("unfortunetely you didn't, sorry")


if __name__ == '__main__':
    main()