import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from matplotlib import pyplot as plt


class Env():
    def __init__(self) -> None:
        self.init()

    def init(self):
        # 1. Create the base environment
        env = gym_super_mario_bros.make('SuperMarioBros-v0')
        # 2. Simplify the controls 
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        # 3. Grayscale
        env = GrayScaleObservation(env, keep_dim=True)
        # 4. Wrap inside the Dummy Environment
        env = DummyVecEnv([lambda: env])
        # 5. Stack the frames
        env = VecFrameStack(env, 4, channels_order='last')
        self.env = env
    
    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self):
        return self.env.render()
        
if __name__ == "__main__":
    env = Env()
    state = env.reset()
    state, reward, done, info = env.step([5])
    plt.figure(figsize=(20,16))
    for idx in range(state.shape[3]):
        plt.subplot(1,4,idx+1)
        plt.imshow(state[0][:,:,idx])
    plt.show()