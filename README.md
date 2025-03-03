<p align="center">
 <img width="80%" src="https://raw.githubusercontent.com/MichaelTMatthews/Craftax/main/images/logo.png" />
</p>

# ⛏️ Multi-Agent Craftax
Multi-Agent Craftax is an RL environment written entirely in <a href="https://github.com/google/jax">JAX</a>.  It is an extension of the recent hit, Craftax which reimplements and significantly extends the game mechanics of <a href="https://github.com/danijar/crafter">Crafter</a>, taking inspiration from roguelike games such as <a href="https://github.com/facebookresearch/nle">NetHack</a>. All the flair of Craftax, but now for the multi-agent setting!

Get ready to run 2 billion environment interactions in less than an hour!

Craftax conforms to the <a href="https://github.com/FLAIROx/JaxMARL/tree/main">JaxMARL</a> interface, allowing easy integration with existing JAX-based multi-agent RL baselines.

<p align="middle">
  <img src="https://raw.githubusercontent.com/MichaelTMatthews/Craftax/main/images/archery.gif" width="200" />
  <img src="https://raw.githubusercontent.com/MichaelTMatthews/Craftax/main/images/building.gif" width="200" /> 
  <img src="https://raw.githubusercontent.com/MichaelTMatthews/Craftax/main/images/dungeon_crawling.gif" width="200" />
</p>
<p align="middle">
  <img src="https://raw.githubusercontent.com/MichaelTMatthews/Craftax/main/images/farming.gif" width="200" />
  <img src="https://raw.githubusercontent.com/MichaelTMatthews/Craftax/main/images/magic.gif" width="200" /> 
  <img src="https://raw.githubusercontent.com/MichaelTMatthews/Craftax/main/images/mining.gif" width="200" />
</p>

# 📜 Basic Usage
Craftax conforms to the gymnax interface:
```python
rng = jax.random.PRNGKey(0)
rng, _rng = jax.random.split(rng)
rngs = jax.random.split(_rng, 3)

# Create environment
env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
env_params = env.default_params

# Get an initial state and observation
obs, state = env.reset(rngs[0], env_params)

# Pick random action
action = env.action_space(env_params).sample(rngs[1])

# Step environment
obs, state, reward, done, info = env.step(rngs[2], state, action, env_params)
```
```
import jax
from craftax.craftax_env import make_craftax_env_from_name

key = jax.random.PRNGKey(0)
key, key_reset, key_act, key_step = jax.random.split(key, 4)

# Initialise environment.
env = make_craftax_env_from_name('Craftax-MARL-Symbolic-v1')

# Reset the environment.
obs, state = env.reset(key_reset)

# Sample random actions.
key_act = jax.random.split(key_act, env.num_agents)
actions = {agent: env.action_space(agent).sample(key_act[i]) for i, agent in enumerate(env.agents)}

# Perform the step transition.
obs, state, reward, done, infos = env.step(key_step, state, actions)
```

## GPU-Enabled JAX
By default, both of the above methods will install JAX on the CPU.  If you want to run JAX on a GPU/TPU, you'll need to install the correct wheel for your system from <a href="https://github.com/google/jax?tab=readme-ov-file#installation">JAX</a>.
For NVIDIA GPU the command is:
```
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

