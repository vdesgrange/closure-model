# Dataset details

Inviscid burgers dataset using random gaussian conditions (single wave without shock)
```
inviscid_burgers_gauss_256_j173.jld2
Julia 1.7.3
256
t in [0., 6.]
x in [0., pi]
t_n, x_n = 64, 64
nu = 0.
initial conditions - random_gaussian_init
sigma = 0.25
mu = 1.
```

Inviscid burgers dataset using high dim gaussian conditions (single wave with shock)
```
inviscid_burgers_high_dim_m4_256_j173.jld2
Julia 1.7.3
256
t in [0., 6.]
x in [0., pi]
t_n, x_n = 64, 64
nu = 0.
initial conditions - high_dim 
m = 4
```

Inviscid burgers dataset using high dim gaussian conditions (multiple wave with shock)
```
inviscid_burgers_high_dim_m10_256_j173.jld2
Julia 1.7.3
256
t in [0., 2.]
x in [0., pi]
t_n, x_n = 64, 64
nu = 0.
initial conditions - high_dim 
m = 10
```

Viscous burgers dataset using high dim gaussian conditions (multiple wave with shocks)
```
viscous_burgers_high_dim_m10_256_j173.jld2
Julia 1.7.3
256
t in [0., 2.]
x in [0., pi]
t_n, x_n = 64, 64
nu = 0.04
initial conditions - high_dim 
m = 10
```

Inviscid burgers dataset using advecting shock initial condition (analytical solution from paper Malik 2020).
```
inviscid_burgers_advecting_shock_t2_4_j173.jld2
Julia 1.7.3
4
t in [0., 2.]
x in [0., 1.]
t_n, x_n = 64, 64
upscale = 64
initial conditions - advecting shock
nu = 0.001
```

POD Galerkin projection from inviscid burgers dataset 
using advecting shock initial condition (analytical solution from paper Malik 2020).
```
inviscid_burgers_advecting_shock_podgp_t2_4_j173.jld2
Julia 1.7.3
POD Galerkin projection
4
t in [0., 2.]
x in [0., 1.]
t_n, x_n = 64, 64
upscale = 64
initial conditions - advecting shock
nu = 0.001
```

```
high_dim_1k_set.jld2
Julia 1.6.3
t in [0., 1.]
x in [0., pi]
t_n, x_n = 64, 64
nu = 0.04
```

```
high_dim_1k_set_j173.jld2
Julia 1.7.3
1024
t in [0., 1.]
x in [0., pi]
t_n, x_n = 64, 64
nu = 0.04
```

```
high_dim_256_set_j173.jld2
Julia 1.7.3
256
t in [0., 1.]
x in [0., pi]
t_n, x_n = 64, 64
nu = 0.04
```

