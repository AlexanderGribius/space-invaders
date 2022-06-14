import streamlit as st
from tensorflow import keras
import tensorflow as tf
import numpy as np

import gym
# from gym import make
from gym.wrappers.atari_preprocessing import AtariPreprocessing
from gym.wrappers.frame_stack import FrameStack


MODEL = r'some/path'

### Helper functions

def instantiate_environmnent():
    env = gym.make("SpaceInvadersNoFrameskip-v4")
    env = AtariPreprocessing(env, grayscale_newaxis=True, frame_skip=4)
    env = FrameStack(env, 4)

    return env

def load_model(model=MODEL):
    return keras.models.load_models(model)

### End Helper Funtions

# Instantiate

env = instantiate_environmnent()
model = load_model()

st.markdown('''# BIGCHAMP-900''')

# import time

# with st.empty():
#      for seconds in range(60):
#          st.write(f"⏳ {seconds} seconds have passed")
#          time.sleep(1)
#      st.write("✔️ 1 minute over!")

episodes = st.number_input('Repeat how many times?')

start_model = st.button('Play!')



for episode in range(1, episodes+1):
    state = np.asarray(env.reset()).reshape(84, 84, 4)
    done = False
    score = 0

    while not done:
        batch_state = tf.expand_dims(state, 0)
        action = np.argmax(model.predict(batch_state)[0])
        state, reward, done, info = env.step(action)
        state_image = np.asarray(state)
        with st.empty():
            st.image(state_image)
        state = np.asarray(state).reshape(84, 84, 4)
        score += reward
