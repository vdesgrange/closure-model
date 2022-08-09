module Generator

using FileIO
using JLD2
using Statistics
using Random

include("../equations/initial_functions.jl")
include("../equations/equations.jl")
include("./processing_tools.jl")

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

function get_burgers_batch(t_max, t_min, x_max, x_min, t_n, x_n, nu, typ)
  """
    get_burgers_batch(t_max, t_min, x_max, x_min, t_n, x_n, nu, typ)

  Small processing of Burgers equation snapshot generated.
  """
  t, u_s = burgers_snapshot_generator(t_max, t_min, x_max, x_min, t_n, x_n, nu, typ)
  u0 = copy(u_s[:, 1])
  return t, u0, u_s
end

function burgers_snapshot_generator(t_max, t_min, x_max, x_min, t_n, x_n, nu, typ=1)
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
  """
  dt = round((t_max - t_min) / (t_n - 1), digits=8);
  dx = round((x_max - x_min) / (x_n - 1), digits=8);

  t = LinRange(t_min, t_max, t_n);
  x = LinRange(x_min, x_max, x_n);

  rand_init = rand((1, 3));
  if typ > 0
    rand_init = typ;
  end

  init = Dict([
    (1, InitialFunctions.random_init),
    (2, (a, b) -> InitialFunctions.high_dim_random_init(a, b, 20)),
    (3, (a, b) -> InitialFunctions.advecting_shock(a, b, nu)),
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

function generate_burgers_training_dataset(t_max, t_min, x_max, x_min, t_n, x_n, nu, n=64, typ=1, upscale=64, keep_high_dim=true, filename="burgers_training_set.jld2", name="training_set")
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
  train_set = [];

  for i in range(1, n, step=1)
    print("Item", i)
    high_t, high_dim = burgers_snapshot_generator(t_max, t_min, x_max, x_min, t_n * upscale, x_n * upscale, nu, typ);
    low_dim = ProcessingTools.downsampling(high_dim, upscale);
    low_t = LinRange(t_min, t_max, t_n);

    if keep_high_dim
      item = [low_t, low_dim, high_t, high_dim];
    else
      item = [low_t, low_dim]
    end
    push!(train_set, item);
  end

  if !isempty(filename)
    JLD2.save(filename, name, train_set);
  end
  return train_set
end

function read_dataset(filepath)
  """
    read_dataset(filepath)
    Load JL2 data file.
  """
  training_set = JLD2.load(filepath)
  return training_set
end

end
