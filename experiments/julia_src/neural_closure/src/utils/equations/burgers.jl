using JLD2

include("../processing_tools.jl")
include("../../equations/equations.jl")
include("../../equations/initial_functions.jl")

"""
  get_burgers_batch(t_max, t_min, x_max, x_min, t_n, x_n, nu, typ)

Small processing of Burgers equation snapshot generated.
"""
function get_burgers_batch(
  t_max, 
  t_min, 
  x_max, 
  x_min, 
  t_n, 
  x_n, 
  nu, 
  typ, 
  init_kwargs=(;))
  t, u_s = burgers_snapshot_generator(t_max, t_min, x_max, x_min, t_n, x_n, nu, typ, init_kwargs)
  u0 = copy(u_s[:, 1])
  return t, u0, u_s
end


"""
  burgers_snapshot_generator(t_max, t_min, x_max, x_min, t_n, x_n, nu, typ=1)

Generate a solution to Bateman-Burgers equation

# Arguments
- `t_max::Float`: t maximum value
- `t_min::Float``: t minimum value
- `x_max::Float`: x maximum value
- `x_min::Float`: x minimum value
- `t_n::Integer`: t axis discretization size
- `x_n::Integer`: x axis discretization size
- `nu::Float`: viscosity parameter
- `typ::Integer`: Initial condition to randomly generates: gaussian random, high dimensional gaussian random
- `init_kwargs`: other keywords arguments for initial functions
"""
function burgers_snapshot_generator(
  t_max,
  t_min,
  x_max,
  x_min,
  t_n,
  x_n,
  nu,
  typ,
  init_kwargs=(;))

  t = LinRange(t_min, t_max, t_n);
  x = LinRange(x_min, x_max, x_n);
  dx = round((x_max - x_min) / (x_n - 1), digits=8);

  rand_init = rand((1, 3));
  if typ > 0
    rand_init = typ;
  end

  init = Dict([
    (1, (a, b) -> InitialFunctions.random_init(a, b)),
    (2, (a, b) -> InitialFunctions.high_dim_random_init(a, b, init_kwargs...)), # a, b, m=20
    (3, (a, b) -> InitialFunctions.advecting_shock(a, b, init_kwargs...)), # a, b, nu
    (4, (a, b) -> InitialFunctions.random_gaussian_init(a, b, init_kwargs...)), # a, b, mu, sigma
  ]);

  u0 = copy(init[rand_init](t, x));
  if ((rand_init == 3) || (nu == 0.))
    t, u = Equations.get_burgers_godunov(t, dx, x_n, nu, u0[1, :])
  else
    t, u = Equations.get_burgers_fft(t, dx, x_n, nu, u0[1, :])
  end

  if sum(isfinite.(u)) != prod(size(u))
    print("u matrix is not finite.")
  end

  return t, u
end


"""
  generate_burgers_training_dataset(t_max, t_min, x_max, x_min, t_n, x_n, nu, n, typ, upscale, keep_high_dim, filename, name)

Generate a dataset of solution to heat equation.

# Arguments
- `t_max::Float`: t maximum value
- `t_min::Float``: t minimum value
- `x_max::Float`: x maximum value
- `x_min::Float`: x minimum value
- `t_n::Integer`: t axis discretization size of coarse grid
- `x_n::Integer`: x axis discretization size of coarse grid
- `nu::Float`: viscosity parameter
- `n::Integer`: number of snapshots to generates
- `typ::Integer`: Initial condition to randomly generates: gaussian random, high dimensional gaussian random, analytical frequency-based solution
- `upscale::Integer`: Size of upscaling for fine grid solution generated: (upscale * t_n, upscale * x_n)
- `keep_high_dim::Boolean`: True to save fine grid solution. False not to (memory space heavily used for high upscale)
- `filename::String`: file name saved
- `name::String`: data structure name saved
"""
function generate_burgers_training_dataset(
  n=64,
  typ=1,
  upscale=64,
  keep_high_dim=true,
  filename="burgers_training_set.jld2",
  name="training_set",
  snap_kwargs=(; t_max=1., t_min=0., x_max=pi, x_min=0., t_n=64, x_n=64, nu=0.),
  init_kwargs=(;),
  )

  train_set = [];

  for i in range(1, n, step=1)
    print("Generating snapshot ", i, "...")

    high_t, high_dim = burgers_snapshot_generator(
      snap_kwargs.t_max, 
      snap_kwargs.t_min, 
      snap_kwargs.x_max, 
      snap_kwargs.x_min, 
      snap_kwargs.t_n * upscale, 
      snap_kwargs.x_n * upscale, 
      snap_kwargs.nu, 
      typ, 
      init_kwargs);

    low_dim = ProcessingTools.downsampling(high_dim, upscale);
    low_t = LinRange(snap_kwargs.t_min, snap_kwargs.t_max, snap_kwargs.t_n);

    if keep_high_dim
      item = [low_t, low_dim, high_t, high_dim];
    else
      item = [low_t, low_dim]
    end

    push!(train_set, item);

    println("Done")
  end

  if !isempty(filename)
    JLD2.save(filename, name, train_set);
  end

  return train_set
end
