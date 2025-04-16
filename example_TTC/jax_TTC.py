import os
os.environ["JAX_ENABLE_X64"] = "True"

import jax
import math
import jax.numpy as jnp
from jax.experimental import jet
jax.config.read("jax_enable_x64") 


def f(x, y):
    return x**4 * y**4


primals = jnp.asarray([1.4, 0.6], dtype=jnp.float64)

# biharmonic operator of f at (1.4, 0.6)
result = 2.0 * 144.0 * primals[0]**2.0 * primals[1]**2 + 24.0 * primals[1]**4.0 + 24.0 * primals[0]**4.0

# gamma_(44)
gamma_44 = 0.09375

# gamma_((2, 2), (4, 0))
gamma_40 = 0.06770833333333333


# gamma_((2, 2), (3, 1))
gamma_31 = -0.3333333333333331

# gamma_((2, 2), (2, 2))
gamma_22 = 0.6250000000000003


# Ensure all gamma constants are float64
gamma_44 = jnp.asarray(gamma_44, dtype=jnp.float64)
gamma_40 = jnp.asarray(gamma_40, dtype=jnp.float64)
gamma_31 = jnp.asarray(gamma_31, dtype=jnp.float64)
gamma_22 = jnp.asarray(gamma_22, dtype=jnp.float64)


# Precompute the coefficients with float64 and 1/4! scaling 
gamma_44_scaled = gamma_44 * jnp.array(1.0 / 24.0, dtype=jnp.float64)

# the following terms are scaled by 2 / 4!
coeff = jnp.array(1.0 / 12.0, dtype=jnp.float64)
gamma_40_scaled = gamma_40 * coeff
gamma_31_scaled = gamma_31 * coeff

# due to the symmetry in the terms gamma_22 is also scaled by 2
gamma_22_scaled = gamma_22 * coeff


### compute the jets we need for the sum ####
# required for first sum
j1 = jet.jet(f, primals, ((4.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)))[1][3]
j2 = jet.jet(f, primals, ((0.0, 0.0, 0.0, 0.0), (4.0, 0.0, 0.0, 0.0)))[1][3]

# required for second sum
j3 = jet.jet(f, primals, ((1.0, 0.0, 0.0, 0.0), (3.0, 0.0, 0.0, 0.0)))[1][3]
j4 = jet.jet(f, primals, ((3.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)))[1][3]

# required for last sum
j5 = jet.jet(f, primals, ((2.0, 0.0, 0.0, 0.0), (2.0, 0.0, 0.0, 0.0)))[1][3]


### compute TTC
# first term: (gamma_44 + 2*(D-1)*gamma_((2, 2), (4, 0))
sum1_1 = gamma_44_scaled * (j1 + j2)
sum1_2 = gamma_40_scaled * (j1 + j2)

# second term: (gamma_((2,2)(3,1)))
sum2 = gamma_31_scaled * (j3 + j4)

# third term: (gamma_((2, 2), (2, 2))
sum3 = gamma_22_scaled * j5

# Final result
ttc_result = sum1_1 + sum1_2 + sum2 + sum3
print(jnp.isclose(jnp.asarray(result, dtype=jnp.float64), ttc_result))