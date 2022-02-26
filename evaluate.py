from tkinter import E
from env import Env
from trainer import PPO

env = Env()
model = PPO.load('./train/best_model_1000000')
state = env.reset()
while True: 
    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
    env.render()