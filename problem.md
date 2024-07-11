#### DQN > Predator Game test.ipynb
##### 错误1
```
ERROR:root:Internal Python error in the inspect module.
Below is the traceback from this internal error.

Traceback (most recent call last):
  File "e:\Anaconda3\envs\py36\lib\site-packages\IPython\core\interactiveshell.py", line 2862, in run_code
    exec(code_obj, self.user_global_ns, self.user_ns)
  File "<ipython-input-8-7021b3f18ae7>", line 1, in <module>
    dqn = build_agent(model, env.ACTION_SPACE_VALUES)
  File "<ipython-input-6-a63b2792ba16>", line 7, in build_agent
    target_model_update=1e-2, policy=policy)
  File "e:\Anaconda3\envs\py36\lib\site-packages\rl\agents\dqn.py", line 108, in __init__
    if hasattr(model.output, '__len__') and len(model.output) > 1:
  File "e:\Anaconda3\envs\py36\lib\site-packages\tensorflow_core\python\framework\ops.py", line 733, in __len__
    "shape information.".format(self.name))
TypeError: len is not well defined for symbolic Tensors. (dense_3/BiasAdd:0) Please call `x.shape` rather than `len(x)` for shape information.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "e:\Anaconda3\envs\py36\lib\site-packages\IPython\core\interactiveshell.py", line 1806, in showtraceback
    stb = value._render_traceback_()
AttributeError: 'TypeError' object has no attribute '_render_traceback_'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "e:\Anaconda3\envs\py36\lib\site-packages\IPython\core\ultratb.py", line 1090, in get_records
    return _fixed_getinnerframes(etb, number_of_lines_of_context, tb_offset)
...
  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
ModuleNotFoundError: No module named 'tensorflow_core.estimator'
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
```
解决：
[TypeError: len is not well defined for symbolic Tensors.](https://github.com/keras-rl/keras-rl/issues/348)
```python
dqn.py 中第108行:
if hasattr(model.output, '__len__') and len(model.output) > 1:
修改成：
if hasattr(model.output, '__shape__') and len(model.output) > 1:
```

##### 错误2
```
  in fit observation, r, done, info = env.step(action)
ValueError: not enough values to unpack (expected 4, got 3)
```
解决：
```python
# 返回值增加info
info = {'prob': 1.0}
return new_observation, reward, done, info
```

##### 错误3
```
Traceback (most recent call last):
  File "e:\Anaconda3\envs\py36\lib\site-packages\IPython\core\interactiveshell.py", line 2862, in run_code
    exec(code_obj, self.user_global_ns, self.user_ns)
  File "<ipython-input-9-71e8f9239596>", line 1, in <module>
    dqn.fit(env, nb_steps=10000, visualize=True, verbose=2)
  File "e:\Anaconda3\envs\py36\lib\site-packages\rl\core.py", line 187, in fit
    callbacks.on_action_end(action)
  File "e:\Anaconda3\envs\py36\lib\site-packages\rl\callbacks.py", line 101, in on_action_end
    callback.on_action_end(action, logs=logs)
  File "e:\Anaconda3\envs\py36\lib\site-packages\rl\callbacks.py", line 366, in on_action_end
    self.env.render(mode='human')
TypeError: render() got an unexpected keyword argument 'mode'
```
解决：
```python
# render增加参数mode
def render(self, mode='human'):
```

```
During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "e:\Anaconda3\envs\py36\lib\site-packages\IPython\core\interactiveshell.py", line 1806, in showtraceback
    stb = value._render_traceback_()
AttributeError: 'TypeError' object has no attribute '_render_traceback_'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "e:\Anaconda3\envs\py36\lib\site-packages\IPython\core\ultratb.py", line 1090, in get_records
    return _fixed_getinnerframes(etb, number_of_lines_of_context, tb_offset)
...
  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
ModuleNotFoundError: No module named 'tensorflow_core.estimator'
```


```
# tensorflow
AttributeError                            Traceback (most recent call last)
Cell In[8], line 1
----> 1 dqn = build_agent(model, env.ACTION_SPACE_VALUES)

Cell In[6], line 6
      4 memory = SequentialMemory(limit=50000, window_length=1) # window_length 窗口值，应该与model输入层的（1,)对应，即输入样本数对应
      5 policy = BoltzmannQPolicy() # 玻尔兹曼策略
----> 6 dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1000, 
      7                target_model_update=1e-2, policy=policy)
      8 dqn.compile(Adam(lr=1e-3), metrics=['mae'])
      9 return dqn

File e:\Anaconda3\envs\tensorflow\lib\site-packages\rl\agents\dqn.py:110, in DQNAgent.__init__(self, model, policy, test_policy, enable_double_dqn, enable_dueling_network, dueling_type, *args, **kwargs)
    108 if hasattr(model.output, '__shape__') and len(model.output) > 1:
    109     raise ValueError('Model "{}" has more than one output. DQN expects a model that has a single output.'.format(model))
--> 110 if model.output._keras_shape != (None, self.nb_actions):
    111     raise ValueError('Model output "{}" has invalid shape. DQN expects a model that has one dimension for each action, in this case {}.'.format(model.output, self.nb_actions))
    113 # Parameters.

AttributeError: 'Tensor' object has no attribute '_keras_shape'
```

```
# tensorflow1.14
Traceback (most recent call last):
  File "F:\Coding\paperCoding\test\t.py", line 254, in <module>
    dqn.fit(env, nb_steps=100000, visualize=False, verbose=1)
  File "E:\Anaconda3\envs\tensorflow1.14\lib\site-packages\rl\core.py", line 194, in fit
    metrics = self.backward(reward, terminal=done)
  File "E:\Anaconda3\envs\tensorflow1.14\lib\site-packages\rl\agents\dqn.py", line 284, in backward
    q_values = self.model.predict_on_batch(state1_batch)
  File "E:\Anaconda3\envs\tensorflow1.14\lib\site-packages\keras\engine\training.py", line 1580, in predict_on_batch
    outputs = self.predict_function(ins)
  File "E:\Anaconda3\envs\tensorflow1.14\lib\site-packages\tensorflow\python\keras\backend.py", line 3292, in __call__
    run_metadata=self.run_metadata)
  File "E:\Anaconda3\envs\tensorflow1.14\lib\site-packages\tensorflow\python\client\session.py", line 1458, in __call__
    run_metadata_ptr)
tensorflow.python.framework.errors_impl.InternalError: 2 root error(s) found.
  (0) Internal: Blas GEMM launch failed : a.shape=(32, 4), b.shape=(4, 32), m=32, n=32, k=4
	 [[{{node dense_1/MatMul}}]]
	 [[dense_3/BiasAdd/_109]]
  (1) Internal: Blas GEMM launch failed : a.shape=(32, 4), b.shape=(4, 32), m=32, n=32, k=4
	 [[{{node dense_1/MatMul}}]]
0 successful operations.
0 derived errors ignored.
```