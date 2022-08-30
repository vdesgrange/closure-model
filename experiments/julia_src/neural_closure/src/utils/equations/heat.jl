using JLD2

include("../../equations/initial_functions.jl")
include("../../equations/equations.jl")
include("../processing_tools.jl")

function get_heat_batch(t_max, t_min, x_max, x_min, t_n, x_n, typ=-1, d=1.)
  """
    get_heat_batch(t_max, t_min, x_max, x_min, t_n, x_n, typ=1, ka=1.)

  Small processing of heat equation snapshot generated.
  """
  t, u_s = heat_snapshot_generator(t_max, t_min, x_max, x_min, t_n, x_n, typ, d)
  u0 = copy(u_s[:, 1])
  return t, u0, u_s
end

function heat_snapshot_generator(t_max, t_min, x_max, x_min, t_n, x_n, typ=1, ka=1.)
  """
    heat_snapshot_generator(t_max, t_min, x_max, x_min, t_n, x_n, typ=1, ka=1.)

  Generate a solution to heat equation 

  # Arguments
  - `t_max::Float`: t maximum value
  - `t_min::Float``: t minimum value
  - `x_max::Float`: x maximum value
  - `x_min::Float`: x minimum value
  - `t_n::Integer`: t axis discretization size
  - `x_n::Integer`: x axis discretization size
  - `typ::Integer`: Initial condition to randomly generates: gaussian random, high dimensional gaussian random, analytical frequency-based solution
  - `ka::Float`: diffusion parameter
  """
  dt = (t_max - t_min) / (t_n - 1);
  dx = (x_max - x_min) / (x_n - 1);

  t = LinRange(t_min, t_max, t_n);
  x = LinRange(x_min, x_max, x_n);

  rand_init = rand((1, 2, 3));
  if typ > 0
    rand_init = typ
  end

  init = Dict([
    (1, InitialFunctions.random_init),
    (2, InitialFunctions.high_dim_random_init),
    (3, (a, b) -> InitialFunctions.heat_analytical_init(a, b, 1:50, [], ka)),
    (4, (a, b) -> InitialFunctions.analytical_heat_1d(a, b, 1:50, [], ka)),
  ]);

  u0 = copy(init[rand_init](t, x));
  if (typ != 4)
    t, u = Equations.get_heat_fft(t, dx, x_n, ka, u0[1, :]);
  else
    u = u0;
  end

  if sum(isfinite.(u)) != prod(size(u))
    print("u matrix is not finite.")
    u0 = copy(InitialFunctions.heat_analytical_init(t, x, collect(range(1, 51, step=1)), [], ka))
    t, u = Equations.get_heat_fft(t, dx, x_n, ka, u0[1, :])
  end

  return t, u
end

function generate_heat_training_dataset(t_max, t_min, x_max, x_min, t_n, x_n, n=64, typ=1, ka=1., filename="heat_training_set.jld2", name="training_set", upscale=1)
  """
    heat_snapshot_generator(t_max, t_min, x_max, x_min, t_n, x_n, typ=1, ka=1.)

  Generate a dataset of solution to heat equation.

  # Arguments
  - `t_max::Float`: t maximum value
  - `t_min::Float``: t minimum value
  - `x_max::Float`: x maximum value
  - `x_min::Float`: x minimum value
  - `t_n::Integer`: t axis discretization size of coarse grid
  - `x_n::Integer`: x axis discretization size of coarse grid
  - `n::Integer`: number of snapshots to generates
  - `typ::Integer`: Initial condition to randomly generates: gaussian random, high dimensional gaussian random, analytical frequency-based solution
  - `ka::Float`: diffusion parameter
  - `filename::String`: file name saved
  - `name::String`: data structure name saved
  - `upscale::Integer`: Size of upscaling for fine grid solution generated: (upscale * t_n, upscale * x_n)
  """
  train_set = [];

  for i in range(1, n, step=1)
    print("Item", i)
    high_t, high_dim = heat_snapshot_generator(t_max, t_min, x_max, x_min, t_n * upscale, x_n * upscale, typ, ka);
    low_dim = ProcessingTools.downsampling(high_dim, upscale);
    low_t = LinRange(t_min, t_max, t_n);

    item = [low_t, low_dim, high_t, high_dim];
    push!(train_set, item);
  end

  JLD2.save(filename, name, train_set);
  return train_set
end
