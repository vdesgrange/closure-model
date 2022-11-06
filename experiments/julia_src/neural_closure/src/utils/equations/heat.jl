using JLD2

include("../../equations/initial_functions.jl")
include("../../equations/equations.jl")
include("../processing_tools.jl")

"""
get_heat_batch(t_max, t_min, x_max, x_min, t_n, x_n, typ=1, ka=1.)
Small processing of heat equation snapshot generated.
"""
function get_heat_batch(tₘₐₓ, tₘᵢₙ, xₘₐₓ, xₘᵢₙ, tₙ, xₙ, typ, init_kwargs=(;)) #(t_max, t_min, x_max, x_min, t_n, x_n, typ=-1, d=1.)
  t, u = heat_snapshot_generator(tₘₐₓ, tₘᵢₙ, xₘₐₓ, xₘᵢₙ, tₙ, xₙ, typ, init_kwargs); # (t_max, t_min, x_max, x_min, t_n, x_n, typ, d)
  u₀ = copy(u[:, 1]);
  return t, u₀, u;
end

function heat_snapshot_generator(tₘₐₓ, tₘᵢₙ, xₘₐₓ, xₘᵢₙ, tₙ, xₙ, typ=-1, init_kwargs=(;)) # (t_max, t_min, x_max, x_min, t_n, x_n, typ=1, ka=1.)
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
  κ = init_kwargs.κ;
  N = init_kwargs.N;
  t = LinRange(tₘᵢₙ, tₘₐₓ, tₙ);
  x = LinRange(xₘᵢₙ, xₘₐₓ , xₙ);
  Δx = round((xₘₐₓ - xₘᵢₙ) / (xₙ - 1), digits=8);

  rand_init = rand((1, 2, 3));
  if typ > 0
    rand_init = typ
  end

  init = Dict([
    (1, InitialFunctions.random_init),
    (2, InitialFunctions.high_dim_random_init),
    (3, (a, b) -> InitialFunctions.heat_analytical_init(a, b, 1:N, [], κ)),
    (4, (a, b) -> InitialFunctions.analytical_heat_1d(a, b, 1:N, [], κ)),
  ]);

  u₀ = copy(init[rand_init](t, x));
  if (typ != 4)
    t, u = Equations.get_heat_fd(t, Δx, u₀[1, :], κ);
    # t, u = Equations.get_heat_fft(t, Δx, xₙ, κ, u₀[1, :]);
  else
    u = u₀;
  end

  if sum(isfinite.(u)) != prod(size(u))
    print("u matrix is not finite.")
  end

  return t, u
end


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
function generate_heat_dataset(
n=64,
upscale=64,
filename="heat.jld2",
snap_kwargs=(;),
init_kwargs=(;))

# function generate_heat_training_dataset(tₘₐₓ, tₘᵢₙ, xₘₐₓ, xₘᵢₙ, tₙ, xₙ, n=64, typ=1, ka=1., filename="heat_training_set.jld2", upscale=1)
  tₘₐₓ, tₘᵢₙ, xₘₐₓ, xₘᵢₙ, tₙ, xₙ, typ = snap_kwargs;
  x = LinRange(xₘᵢₙ, xₘₐₓ , xₙ);
  Δx = round((xₘₐₓ - xₘᵢₙ) / (xₙ - 1), digits=8);

  train_set = [];
  for i in range(1, n, step=1)
    print("Generating snapshot ", i, "...")
  
    tₕ, u = heat_snapshot_generator(tₘₐₓ, tₘᵢₙ, xₘₐₓ, xₘᵢₙ, tₙ, xₙ * upscale, typ, init_kwargs); # tₙ * upscale
    û = ProcessingTools.downsampling_x(u, upscale);
    item = [tₕ, û, u, snap_kwargs, init_kwargs]

    # high_t, high_dim = heat_snapshot_generator(t_max, t_min, x_max, x_min, t_n, x_n * upscale, typ, ka);
    # low_dim = ProcessingTools.downsampling_x(high_dim, upscale);
    # low_t = LinRange(t_min, t_max, t_n);
    # item = [low_t, low_dim, high_t, high_dim];
    push!(train_set, item);
  
    println("Done")
  end

  if !isempty(filename)
    JLD2.save(filename, "training_set", train_set);
  end
  
  return train_set
end
