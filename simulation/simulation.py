import jax
import jax.numpy as jnp
from jax import Array as Tensor
from typing import Callable, Tuple, Any, Union, Dict
from jax.random import PRNGKey
from data import rosenbrock,rastrigin,ackley,generate_data
from dataclasses import dataclass

Params = Dict[str, Union[Tensor, Dict[str,Any]]]
Dataset = Dict[str,Tensor]
# implement the model

def sigmoid(x):
    return 1 / (1 + jnp.exp(-1 * x))

@dataclass
class ParalellDense:
    kn  : int
    L   : int
    r   : int
    d   : int
    n   : int
    c1  : float
    c2  : float
    tau : float
    
    def init(self, key : PRNGKey) -> Params:
        # fix the key mess
        keys = jax.random.split(key,self.L+2)
        
        maxval = self.c2 * jnp.log(n)*jnp.pow(n,self.tau)
        in_proj = jax.random.uniform(
            key,
            (self.kn,self.d,self.r),
            minval = -1* maxval,
            maxval = maxval
        )
        
        layers = {
            'layer' + str(i) : jax.random.uniform(
                key,
                (self.kn,self.r,self.r),
                minval = -1*self.c1,
                maxval = self.c2
            )
            for i in range(self.L-1)
        }
        
        out_proj = jax.random.uniform(
                key,
                (self.kn,self.r,self.r),
                minval = -1*self.c1,
                maxval = -1*self.c2
            )
        
        weighting = jnp.zeros((self.kn,1))

        return {
            'in_proj'   : in_proj,
            'layers'    : layers,
            'out_proj'  : out_proj,
            'weighting' : weighting
        }

    def __call__(self, w : Params, x : Tensor) -> Tensor:
        x = jnp.einsum('bi,kir->bkr', x, w['in_proj'])
        x = sigmoid(x)
        for mat in w['layers'].values():
            x = jnp.einsum('bkr,krj->bkj', x, mat)
            x = sigmoid(x)
        x = jnp.einsum('bkr,kr->bk', x, w['out_proj'])
        x = sigmoid(x)
        x = jnp.einsum('bk,kj->bj',x, w['weighting'])
        return x
            

def get_model(
        n    : int,
        c1   : float,
        c2   : float,
        c3   : float,
        c4   : float,
        kn   : int,
        q    : int,
        d    : int,
        beta : float,
        p    : float
) -> ParalellDense:
    
    beta = jnp.log(n)*c3
    L    = jnp.ceil(jnp.log(q+d))+1
    r    = 2*(jnp.ceil(2p+d)**2)
    tau  = 1 / (2p+d)
    lam  = c5 / (n*kn**3)
    tn   = jnp.ceil(c6*(kn**3) / beta)
    train_params = {'lambda' : lam, 'beta' beta, 'tn' : tn}
    return ParalellDense(kn,L,r,d,n,c1,c2,tau), train_params



    
key = jax.random.key(seed = 213487)
key_data, key_model = jax.random.split(key)
d = 15
kn = 10
L = 15
r = 20
model = ParalellDense(kn,L,r,d)
data,Y = generate_data(
    rosenbrock,
    key_data,
    0.05,
    0.0,
    1,
    30000,
    d
)
params = model.init(key)
print(model(params,data['x']))




# least squares regression
@jax.jit
def loss_fn(params,x,y):
    return jnp.average(((y-model(params,x))**2))
    
@jax.jit
def apply_grads(params: Params, grads: Any, learning_rate: float) -> Any:
    return jax.tree.map(
        lambda p, g: p - learning_rate * g,
        params,
        grads
    )

def train_step(
    loss_fn: Callable,
    params: Any,
    batch: Tuple,
    learning_rate: float
) -> Tuple[Any, float]:
    
    loss, grads = jax.value_and_grad(loss_fn)(params,batch['x'],batch['y'])
    params = apply_grads(params, grads, learning_rate)
    
    return params, loss

def train_loop(
    loss_fn: Callable,
    params: Params,
    data: Dict[str, Tensor],
    num_epochs: int,
    learning_rate: float
) -> Params:
    
    for epoch in range(num_epochs):
        # Perform single training step
        params, loss = train_step(loss_fn, params, data, learning_rate)
        print(f"Epoch: {epoch+1}, Loss: {loss}")
    
    return params

train_loop(loss_fn,params,data,200,0.01)
