import numpy as np
import cv2
from PIL import Image
import pickle

# 方格类：可实例化为玩家、食物、敌人
class Cube:
    def __init__(self, size):
        self.size = size
        self.x = np.random.randint(0, self.size)
        self.y = np.random.randint(0, self.size)
        
    def __str__(self):
        return f'{self.x},{self.y}'
    
    def __sub__(self, other):
        return (self.x-other.x,self.y-other.y)
    
    def __eq__(self, other):
        return (self - other == (0, 0))

    # 1:向左或向下，2：不动，3：向右或向上
    def action(self, choise):
        if choise == 0:
            self.move(x=2, y=2)
        elif choise == 1:
            self.move(x=3, y=2)
        elif choise == 2:
            self.move(x=2, y=3)
        elif choise == 3:
            self.move(x=2, y=1)
        elif choise == 4:
            self.move(x=1, y=2)
        elif choise == 5:
            self.move(x=3, y=1)
        elif choise == 6:
            self.move(x=3, y=3)
        elif choise == 7:
            self.move(x=1, y=3)
        elif choise == 8:
            self.move(x=1, y=1)  
    
    def move(self, x=False, y=False):
        if not x:
            self.x = self.x + np.random.randint(-1, 2)
        else:
            self.x = self.x + x - 2
        if not y:
            self.y = self.y + np.random.randint(-1, 2) 
        else:
            self.y = self.y + y -2
               
        #边界情况
        if self.x < 0:
            self.x = 0
        if self.x >= self.size:
            self.x = self.size-1
        if self.y < 0:
            self.y = 0
        if self.y >= self.size:
            self.y = self.size-1


# 环境类 
class envCube:
    SIZE = 10

    ACTION_SPACE_VALUES = 9 # action操作数
    IMAGE_SHAPE = (SIZE, SIZE, 3) # 画面大小
    RETURN_IMAGE = False # 是否返回图像为observation

    FOOD_REWARD = 25 #食物奖励
    MOVE_PENALITY = -1 #移动惩罚
    ENEMY_PENALITY = -300 #敌人惩罚

    
    BGR = { 1:(255, 0, 0), # blue
            2:(0, 255, 0), # green
            3:(0, 0, 255)} # red
        
    PLAYER_N = 1
    FOOD_N = 2
    ENEMY_N = 3

    # 重置环境
    def reset(self):
        self.player = Cube(self.SIZE)
        self.food = Cube(self.SIZE)
        while (self.player == self.food):
            self.food = Cube(self.SIZE)

        self.enemy = Cube(self.SIZE)
        while (self.player == self.enemy or self.food == self.enemy):
            self.enemy = Cube(self.SIZE)
        
        if self.RETURN_IMAGE:
            observation = np.array(self.get_image())
        else:
            observation = (self.player - self.food) + (self.player - self.enemy)

        # 局数重置
        self.episode_step = 0

        return observation
    
    '''
    new_observation: 新状态
    reward: 当前步骤的奖励
    done: 游戏是否结束
    '''
    def step(self, action):
        self.episode_step += 1

        self.player.action(action)
        self.food.move()
        self.enemy.move()

        if self.RETURN_IMAGE:
            new_observation = np.array(self.get_image())
        else:
            new_observation = (self.player - self.food) + (self.player - self.enemy)

        if (self.player == self.food):
            reward = self.FOOD_REWARD
        elif (self.player == self.enemy):
            reward = self.ENEMY_PENALITY
        else:
            reward = self.MOVE_PENALITY
        
        done = False
        if (self.player == self.food or self.player == self.enemy or self.episode_step >= 200):
            done = True

        return new_observation, reward, done
    

    '''
    获取画图数组
    '''
    def get_image(self):
        env_image = np.zeros(self.IMAGE_SHAPE, dtype=np.uint8)
        env_image[self.food.x][self.food.y] = self.BGR[self.FOOD_N]
        env_image[self.player.x][self.player.y] = self.BGR[self.PLAYER_N]
        env_image[self.enemy.x][self.enemy.y] = self.BGR[self.ENEMY_N]
        image = Image.fromarray(env_image, 'RGB')
        return image

    '''
    画图
    '''
    def render(self):
        img = self.get_image()
        img = img.resize((800, 800))
        cv2.imshow('', np.array(img))
        cv2.waitKey(1)

    '''
    q_table
    '''
    def init_q_table(self, q_table_name=None):
        # 初始化Q——table
        if q_table_name is None:
            q_table = {}

            for x1 in range(-self.SIZE+1, self.SIZE):
                for y1 in range(-self.SIZE+1, self.SIZE):
                    for x2 in range(-self.SIZE+1, self.SIZE):
                        for y2 in range(-self.SIZE+1, self.SIZE):
                            q_table[x1, y1, x2, y2] = [np.random.uniform(-5, 0) for i in range(self.ACTION_SPACE_VALUES)]                        
        else:
            with open(q_table_name, 'rb') as f:
                q_table = pickle.load(f) 
        return q_table