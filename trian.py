import numpy as np
import cv2
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import time
import os
from matplotlib import style
style.use('ggplot')

from environment.pg import envCube

#train
'''
import gymnasium as gym
env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset(seed=42)
for _ in range(1000):
   action = env.action_space.sample()  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()

env.close()
'''
# 初始化环境参数
SIZE = 10
EPISODES = 100000
SHOW_EVERY = 3000

epsilon = 0.6 #选择随机概率
EPS_DECAY = 0.9998 #概率折扣

DISCOUNT = 0.95 #折扣 gamma
LEARNING_RATE = 0.3 #alpha

q_table_name = f'Q_table\qtable_1719578112.pickle'

env = envCube()
obs = env.reset()
q_table = env.init_q_table()

# train
episode_rewards = [] # 奖励序列
for episode in range(EPISODES): 
    done = False   

    # 显示图像 
    if(episode % SHOW_EVERY == 0):
        print(f'episode #{episode}, epsilon:{epsilon}')
        flag_show = False
        if(episode >= SHOW_EVERY):
            print(f'mean reward:{np.mean(episode_rewards[-SHOW_EVERY:])}')
            flag_show = True
    else:
        flag_show = False
    
    episode_reward = 0
    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs]) #选择Q值最高动作
        else:
            action = np.random.randint(0, env.ACTION_SPACE_VALUES) #随机选择一个动作
            
        newObs, reward, done = env.step(action)
    
        # Update the Q_table----------------------
        current_q = q_table[obs][action]    # 当前动作、状态对应Q_value
        max_future_q = np.max(q_table[obs]) # 新状态最大Q_value

        if (reward == env.FOOD_REWARD):
            new_q = env.FOOD_REWARD
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT + max_future_q) 

        q_table[obs][action] = new_q
        obs = newObs
        # ---------------------------------
        
        if flag_show:
            env.render()
  
        episode_reward += reward
        
    if done:
        obs = env.reset()       
    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY
    
cv2.destroyAllWindows()
#画曲线
moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode = 'valid')
print(len(moving_avg))
plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.xlabel('episode #')
plt.ylabel(f'mean {SHOW_EVERY} reward')
plt.show()

if not os.path.exists('Q_table'):
    os.makedirs('Q_table')
with open(f'Q_table\qtable_pro_{int(time.time())}.pickle', 'wb') as f:
    pickle.dump(q_table, f)

