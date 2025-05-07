import os
os.environ["JAX_ENABLE_X64"] = "True"

import jax
import math
import jax.numpy as jnp
from jax.experimental import jet
jax.config.read("jax_enable_x64")


gamma_40 = jnp.asarray(13.0, dtype=jnp.float64) / jnp.asarray(192.0, dtype=jnp.float64)
gamma_31 = jnp.asarray(-1.0, dtype=jnp.float64) / jnp.asarray(3.0, dtype=jnp.float64)
gamma_22 = jnp.asarray(5.0, dtype=jnp.float64) / jnp.asarray(8.0, dtype=jnp.float64)

def f(x, y):
    return x**4 * y**4


primals = jnp.asarray([1.4, 0.6], dtype=jnp.float64)

# biharmonic operator of f at (1.4, 0.6)
result = 2.0 * 144.0 * primals[0]**2.0 * primals[1]**2 + 24.0 * primals[1]**4.0 + 24.0 * primals[0]**4.0



### compute the jets we need for the sum ####
j1 = jet.jet(f, primals, ((4.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)))[1][3]
j2 = jet.jet(f, primals, ((0.0, 0.0, 0.0, 0.0), (4.0, 0.0, 0.0, 0.0)))[1][3]
j3 = jet.jet(f, primals, ((1.0, 0.0, 0.0, 0.0), (3.0, 0.0, 0.0, 0.0)))[1][3]
j4 = jet.jet(f, primals, ((3.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)))[1][3]
j5 = jet.jet(f, primals, ((2.0, 0.0, 0.0, 0.0), (2.0, 0.0, 0.0, 0.0)))[1][3]


### compute TTC
# first term: (2*D*gamma_((2, 2), (4, 0)) + 2 * gamma_((2,2)(3,1)) + gamma_((2, 2), (2, 2) 
sum1 = (4.0 * gamma_40 +  2.0 * gamma_31 + gamma_22) * (j1 + j2)

# second term: (gamma_((2,2)(3,1)))
sum2 = 2.0 * gamma_31 * (j3 + j4)

# third term: (gamma_((2, 2), (2, 2))
sum3 = 2.0 * gamma_22 * (j5)

# Final result
ttc_result = (sum1 + sum2 + sum3) / 24.0
jnp.isclose(jnp.asarray(result, dtype=jnp.float64), ttc_result)