import jax
import jax.numpy as jnp
from jax import Array as Tensor
from typing import Callable, Tuple, Any, Union, Dict
from jax.random import PRNGKey

def create_jax_function(func: Callable) -> Callable:
    jitted_func = jax.jit(func)

    vectorized_func = jax.jit(jax.vmap(func))
    
    def wrapper(x):
        # Handle both single inputs and batches
        if len(x.shape) == 1:
            return jitted_func(x)
        return vectorized_func(x)
    
    return wrapper


@create_jax_function
def rosenbrock(x):
    """
    The Rosenbrock function (banana function)
    Minimum at (1,1,...,1)
    """
    return jnp.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

@create_jax_function
def rastrigin(x):
    """
    The Rastrigin function
    Minimum at (0,0,...,0)
    """
    n = len(x)
    return 10 * n + jnp.sum(x**2 - 10 * jnp.cos(2 * jnp.pi * x))

@create_jax_function
def ackley(x):
    """
    The Ackley function
    Minimum at (0,0,...,0)
    """
    n = len(x)
    sum_sq = jnp.sum(x**2)
    sum_cos = jnp.sum(jnp.cos(2 * jnp.pi * x))
    return (-20 * jnp.exp(-0.2 * jnp.sqrt(sum_sq/n)) 
            - jnp.exp(sum_cos/n) + 20 + jnp.e)


def generate_data(
        reg_fn       : Callable,
        key          : PRNGKey,
        noise_var    : float,
        min_x        : float,
        max_x        : float,
        num_examples : int,
        d            : int = 10
) -> Tuple[Dict['str',Tensor],Tensor]:
    
    key_x, key_y, key_e = jax.random.split(key,3)
    X = jax.random.uniform(
        key    = key_x,
        shape  = (num_examples,d),
        minval = min_x,
        maxval = max_x
    )
    eps = jax.random.normal(key_e,(num_examples,1))
    Y = reg_fn(X)
    Z = Y + noise_var * eps

    return {'x' : X, 'y' : Z}, Y

