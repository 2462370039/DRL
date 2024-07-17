from env import envCube
from dqn_agent import build_model, build_agent

# Get the environment and extract the number of actions.
env = envCube()

model = build_model(env.OBSERVATION_SPACE_VALUES, env.ACTION_SPACE_VALUES)
print(model.summary())

dqn = build_agent(model, env.ACTION_SPACE_VALUES)
dqn.fit(env, nb_steps=100000, visualize=False, verbose=1)

# After training is done, we save the final weights.
dqn.save_weights('dqn_n1_weights.h5f', overwrite=True)
