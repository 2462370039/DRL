## DQN > Predator Game test.ipynb
### 错误1：TypeError: len is not well defined for symbolic Tensors.
<details> 
<summary><font size="4" color="orange">报错原文</font></summary> 
<pre><code class="language-cpp">
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
</code>
</pre> </details>


解决：[参考链接](https://github.com/keras-rl/keras-rl/issues/348)
```python
dqn.py 中第108行:
if hasattr(model.output, '__len__') and len(model.output) > 1:
修改成：
if hasattr(model.output, '__shape__') and len(model.output) > 1:
```

### 错误2：ValueError: not enough values to unpack (expected 4, got 3)
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

### 错误3：TypeError: render() got an unexpected keyword argument 'mode'

<details> 
<summary><font size="4" color="orange">报错原文</font></summary> 
<pre><code class="language-cpp">
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
</code>
</pre> </details>

解决：
```python
# render增加参数mode
def render(self, mode='human'):
```

### 错误4：ModuleNotFoundError: No module named 'tensorflow_core.estimator'
<details> 
<summary><font size="4" color="orange">报错原文</font></summary> 
<pre><code class="language-cpp">
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
</code>
</pre> </details>

解决
```
重新安装tensorflow-estimator，保证与tensorflow版本一致
```

### 错误5：AttributeError: Tensor.op is meaningless when eager execution is enabled.
<details> 
<summary><font size="4" color="orange">报错原文</font></summary> 
<pre><code class="language-cpp">
Traceback (most recent call last):
  File "f:/Coding/paperCoding/RL/DQN/dqn_cartpole.py", line 50, in <module>
    dqn.fit(env, nb_steps=50000, visualize=True, verbose=2)
  File "E:\Anaconda3\envs\tf2.1\lib\site-packages\rl\core.py", line 194, in fit
    metrics = self.backward(reward, terminal=done)
  File "E:\Anaconda3\envs\tf2.1\lib\site-packages\rl\agents\dqn.py", line 325, in backward
    metrics = self.trainable_model.train_on_batch(ins + [targets, masks], [dummy_targets, targets])
  File "E:\Anaconda3\envs\tf2.1\lib\site-packages\keras\engine\training.py", line 1513, in train_on_batch
    self._make_train_function()
  File "E:\Anaconda3\envs\tf2.1\lib\site-packages\keras\engine\training.py", line 333, in _make_train_function
    **self._function_kwargs)
  File "E:\Anaconda3\envs\tf2.1\lib\site-packages\keras\backend\tensorflow_backend.py", line 3009, in function
    **kwargs)
  File "E:\Anaconda3\envs\tf2.1\lib\site-packages\tensorflow_core\python\keras\backend.py", line 3760, in function
    return EagerExecutionFunction(inputs, outputs, updates=updates, name=name)
  File "E:\Anaconda3\envs\tf2.1\lib\site-packages\tensorflow_core\python\keras\backend.py", line 3657, in __init__
    base_graph=source_graph)
  File "E:\Anaconda3\envs\tf2.1\lib\site-packages\tensorflow_core\python\eager\lift_to_graph.py", line 260, in lift_to_graph
    add_sources=add_sources))
  File "E:\Anaconda3\envs\tf2.1\lib\site-packages\tensorflow_core\python\ops\op_selector.py", line 393, in map_subgraph
    ops_to_visit = [_as_operation(init_tensor)]
  File "E:\Anaconda3\envs\tf2.1\lib\site-packages\tensorflow_core\python\ops\op_selector.py", line 320, in _as_operation
    return op_or_tensor.op
  File "E:\Anaconda3\envs\tf2.1\lib\site-packages\tensorflow_core\python\framework\ops.py", line 1094, in op
    "Tensor.op is meaningless when eager execution is enabled.")
AttributeError: Tensor.op is meaningless when eager execution is enabled.
</code>
</pre> </details>

解决
```python
# 添加
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
```

### 错误6：AttributeError: 'str' object has no attribute 'decode'

<details> 
<summary><font size="4" color="orange">报错原文</font></summary> 
<pre><code class="language-cpp">
Traceback (most recent call last):
  File "e:\Coding\RL\DQN\dqn_train.py", line 17, in <module>
    dqn.load_weights('dqn_n1_weights.h5f')
  File "D:\Software\Coding\anaconda3\envs\tf2\lib\site-packages\rl\agents\dqn.py", line 209, in load_weights
    self.model.load_weights(filepath)
  File "D:\Software\Coding\anaconda3\envs\tf2\lib\site-packages\keras\engine\saving.py", line 492, in load_wrapper       
    return load_function(*args, **kwargs)
  File "D:\Software\Coding\anaconda3\envs\tf2\lib\site-packages\keras\engine\network.py", line 1230, in load_weights     
    f, self.layers, reshape=reshape)
  File "D:\Software\Coding\anaconda3\envs\tf2\lib\site-packages\keras\engine\saving.py", line 1183, in load_weights_from_hdf5_group
    original_keras_version = f.attrs['keras_version'].decode('utf8')
AttributeError: 'str' object has no attribute 'decode'
</code>
</pre> </details>

解决：[参考链接](https://github.com/keras-team/keras/issues/14294 )
```shell
# h5py降级为2.10.0
pip install h5py==2.10.0
```
