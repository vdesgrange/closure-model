module Generator

using FileIO
using JLD2
using Statistics
using Random

include("../equations/initial_functions.jl")
include("../equations/equations.jl")
include("./processing_tools.jl")

function get_heat_batch(t_max, t_min, x_max, x_min, t_n, x_n, typ=-1, d=1., k=1.)
  t, u_s = heat_snapshot_generator(t_max, t_min, x_max, x_min, t_n, x_n, typ, d, k)
  u0 = copy(u_s[:, 1])
  return t, u0, u_s
end

function heat_snapshot_generator(t_max, t_min, x_max, x_min, t_n, x_n, typ=1, ka=1., k=1.)
  dt = round((t_max - t_min) / (t_n - 1), digits=8);
  dx = round((x_max - x_min) / (x_n - 1), digits=8);

  t = LinRange(t_min, t_max, t_n);
  x = LinRange(x_min, x_max, x_n);

  rand_init = rand((1, 2, 3));
  if typ > 0
    rand_init = typ
  end

  init = Dict([
    (1, InitialFunctions.random_init),
    (2, InitialFunctions.high_dim_random_init),
    (3, (a, b) -> InitialFunctions.heat_analytical_init(a, b, collect(1:50), [], ka)),
    (4, (a, b) -> InitialFunctions.analytical_heat_1d(a, b, collect(1:50), [], ka)),
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
  t, u_s = burgers_snapshot_generator(t_max, t_min, x_max, x_min, t_n, x_n, nu, typ)
  u0 = copy(u_s[:, 1])
  return t, u0, u_s
end

function burgers_snapshot_generator(t_max, t_min, x_max, x_min, t_n, x_n, nu, typ=1)
  dt = round((t_max - t_min) / (t_n - 1), digits=8);
  dx = round((x_max - x_min) / (x_n - 1), digits=8);

  t = LinRange(t_min, t_max, t_n);
  x = LinRange(x_min, x_max, x_n);

  rand_init = rand((1, 2));
  if typ > 0
    rand_init = typ;
  end

  init = Dict([
    (1, InitialFunctions.random_init),
    (2, InitialFunctions.high_dim_random_init),
  ]);

  u0 = copy(init[rand_init](t, x));
  t, u = Equations.get_burgers_fft(t, dx, x_n, nu, u0[1, :])

  if sum(isfinite.(u)) != prod(size(u))
    print("u matrix is not finite.")
  end

  return t, u
end

function generate_heat_training_dataset(t_max, t_min, x_max, x_min, t_n, x_n, n=64, typ=1, ka=1., k=1., filename="heat_training_set.jld2", name="training_set")
  train_set = [];
  upscale = 1;

  for i in range(1, n, step=1)
    print("Item", i)
    high_t, high_dim = heat_snapshot_generator(t_max, t_min, x_max, x_min, t_n * upscale, x_n * upscale, typ, ka, k);
    low_dim = ProcessingTools.downsampling(high_dim, upscale);
    low_t = LinRange(t_min, t_max, t_n);

    item = [low_t, low_dim, high_t, high_dim];
    push!(train_set, item);
  end

  JLD2.save(filename, name, train_set);
  return train_set
end

function generate_burgers_training_dataset(t_max, t_min, x_max, x_min, t_n, x_n, nu, n=64, typ=1, upscale=64, keep_high_dim=true, filename="burgers_training_set.jld2", name="training_set")
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

  JLD2.save(filename, name, train_set);
  return train_set
end

function read_dataset(filepath)
  training_set = JLD2.load(filepath)
  return training_set
end

end
