module Generator

using Random

include("../equations/initial_functions.jl")
include("../equations/heat.jl")

function get_heat_batch(t_max, t_min, x_max, x_min, t_n, x_n, typ=-1, d=1., k=1.)
  t, u_s = heat_snapshot_generator(t_max, t_min, x_max, x_min, t_n, x_n, typ, d, k)
  u0 = copy(u_s[1, :])
  return t, u0, u_s
end

function heat_snapshot_generator(t_max, t_min, x_max, x_min, t_n, x_n, typ=-1, d=1, k=1)
  dt = round((t_max - t_min) / (t_n - 1), digits=8);
  dx = round((x_max - x_min) / (x_n - 1), digits=8);

  t = LinRange(t_min, t_max, t_n);
  x = LinRange(x_min, x_max, x_n);

  rand_init = rand((1, 2, 3));
  if typ > -1
    rand_init = typ
  end

  init = Dict([
    (1, InitialFunctions.random_init),
    (2, InitialFunctions.high_dim_random_init),
    (3, (a, b) -> InitialFunctions.heat_analytical_init(a, b, collect(range(1, 51, step=1)), [], k)),
  ]);

  u0 = copy(init[rand_init](t, x));
  t, u = get_heat_fft(t, dx, x_n, d, u0[1, :])

  if sum(isfinite.(u)) != prod(size(u))
    print("u matrix is not finite.")
    u0 = copy(heat_analytical_init(t, x, collect(range(1, 51, step=1)), [], k))
    t, u = get_heat_fft(t, dx, x_n, d, u0[1, :])
  end

  return t, u
end

function get_burgers_batch()
end

function burgers_snapshot_generator()
end

function generate_heat_training_dataset()
end

function read_dataset(filepath)
end

function process_dataset(dataset)
end

end
