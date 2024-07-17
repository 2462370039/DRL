from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

def build_model(status, nb_actions): 
    # Next, we build a very simple model.
    model = Sequential()
    model.add(Flatten(input_shape=(2,) + status)) # status: env.observation_space.shape
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    # 输出层
    model.add(Dense(nb_actions, activation='linear')) # nb_actions: env.action_space.n
    return model


def build_agent(model, nb_actions):
    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=50000, window_length=2) # window_length 窗口值，应该与model输入层的（1,)对应，即输入样本数对应
    policy = BoltzmannQPolicy() # 玻尔兹曼策略
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1000, 
                   target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    return dqn