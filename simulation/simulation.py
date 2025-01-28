import jax
import jax.numpy as jnp


# implement the model

def ParalellDense:
    def __init__(self, kn : int, L : int, r : int, d : int) -> None:
        self.kn = kn
        self.L = L
        self.r = r
        self.d = d

    def init(self, key : PRNGKey) -> Params:
        keys = jax.random.split(key,len(layers))
        in_proj = jax.random.normal(keys, (kn,self.r,self.d))
        layers = {
            'layer' + str(i) : jax.random.normal(keys, (kn,self.r,self.r))
            for i in range(self.L-1)
        }
        out_proj = jax.random.normal(keys[], (kn,self.r))
        weighting = jax.random.normal(keys[0],(kn,1))

        return {
            'in_proj'   : in_proj,
            'layers'    : layers,
            'out_proj'  : out_proj,
            'weighting' : weigthing
        }

    def __call__(self, w : Params, x : Tensor) -> Tensor:
        x = jnp.einsum('', x, w['in_proj'])
        for mat in w['layers'].values():
            x = jnp.einsum('', x, mat)
        x = jnp.einsum('', x, w['out_proj'])
        x = jnp.einsum(''x, w['weighting'])
        return x
            

            
