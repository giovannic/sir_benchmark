import sys
import csv
from typing import Any, Tuple
from dataclasses import dataclass
from jax import numpy as jnp, random, vmap, jit, default_device, devices
from jax.lax import scan
from jaxtyping import Array
from jax.tree_util import register_pytree_node

# Read in command line arguments
N = int(sys.argv[1])
parameters = sys.argv[2]
output_file = sys.argv[3]
use_gpu = int(sys.argv[4])

# Add some static args
dt = .1
timesteps = int(100. / dt)

def bernoulli(key: Any, rate: Array, shape: Tuple) -> Array:
    return random.bernoulli(key, rate, shape)

@dataclass
class State:
    susceptible: Array
    infected: Array
    recovered: Array

register_pytree_node(
    State,
    lambda tree: ((tree.susceptible, tree.infected, tree.recovered), None),
    lambda _, args: State(*args)
)

@dataclass
class Observation:
    n_susceptible: Array
    n_infected: Array
    n_recovered: Array

register_pytree_node(
    Observation,
    lambda tree: ((tree.n_susceptible, tree.n_infected, tree.n_recovered), None),
    lambda _, args: Observation(*args)
)

def init(infected: float, n: int) -> State:
    n_infected = jnp.array(n * infected, int)
    susceptible = (jnp.arange(n) < n_infected).astype(int)
    return State(
        susceptible,
        1 - susceptible,
        jnp.zeros((n,))
    )

def step(
    key: Any,
    beta: float,
    gamma: float,
    state: State
    ) -> State:

    # calculate force of infection
    foi = beta * jnp.sum(state.infected) / N

    # sample infections
    key, key_i = random.split(key)
    new_infections = bernoulli(key_i, state.susceptible * foi, (N,))

    # sample recoveries
    key, key_i = random.split(key)
    new_recoveries = bernoulli(key_i, state.infected * gamma, (N,))

    # make new state
    return State(
        state.susceptible - new_infections,
        state.infected - new_recoveries + new_infections,
        state.recovered + new_recoveries
    )

def observe(state: State) -> Observation:
    return Observation(
        state.susceptible.sum(),
        state.infected.sum(),
        state.recovered.sum()
    )

def _scan_step(
        key: Any,
        beta: float,
        gamma: float,
        state: State
    ) -> Tuple[State, Observation]:
    new_state = step(key, beta, gamma, state)
    return new_state, observe(new_state)

def run(
    key: Any,
    infected: float,
    beta: float,
    gamma: float,
    ) -> Observation:
    state = init(infected, N)
    _, obs = scan(
        f = lambda s, k: _scan_step(k, beta, gamma, s),
        init = state,
        xs = random.split(key, timesteps),
        length = timesteps
    )
    return obs

if __name__ == '__main__':

    # Read in parameters into jnp arrays
    with open(parameters, 'r') as f:
        reader = csv.reader(f)
        R0, I0, gamma = zip(*[
            (float(row[0]), float(row[1]), float(row[2]))
            for row in reader
        ])
        R0 = jnp.round(N * jnp.array(R0))
        I0 = jnp.round(N * jnp.array(I0))
        gamma = jnp.array(gamma)

    outputs = []

    if use_gpu:
        device = devices("gpu")[0]
    else:
        device = devices("cpu")[0]

    with default_device(device):
        outputs = jit(vmap(
            run,
            in_axes=(0, 0, 0, 0)
        ))(
            random.split(random.PRNGKey(0), len(I0)),
            I0,
            R0,
            gamma,
        )

    # Reshape outputs and write to file
    outputs = jnp.stack([
        outputs.n_susceptible.reshape(-1),
        outputs.n_infected.reshape(-1),
        outputs.n_recovered.reshape(-1)
    ]).T
    with open(output_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Susceptible', 'Infected', 'Removed'])
        writer.writerows(outputs)