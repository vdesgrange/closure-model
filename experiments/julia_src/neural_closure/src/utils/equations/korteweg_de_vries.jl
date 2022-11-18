include("../processing_tools.jl")
include("../../rom/pod.jl")
include("../../equations/equations.jl")
include("../../equations/initial_functions.jl")

"""
  get_kdv_batch(t_max, t_min, x_max, x_min, t_n, x_n, nu, typ)

Small processing of original KdV equation snapshot generated.
"""
function get_kdv_batch(tₘₐₓ, tₘᵢₙ, xₘₐₓ, xₘᵢₙ, tₙ, xₙ, typ,init_kwargs=(;))
  t, u = kdv_snapshot_generator(tₘₐₓ, tₘᵢₙ, xₘₐₓ, xₘᵢₙ, tₙ, xₙ, typ, init_kwargs);
  u₀ = copy(u[:, 1]);
  return t, u₀, u
end

"""
  kdv_snapshot_generator(t_max, t_min, x_max, x_min, t_n, x_n, nu, typ=1)

Generate a solution to original Korteweg-de-Vries equation.

# Arguments
- `t_max::Float`: t maximum value
- `t_min::Float``: t minimum value
- `x_max::Float`: x maximum value
- `x_min::Float`: x minimum value
- `t_n::Integer`: t axis discretization size
- `x_n::Integer`: x axis discretization size
- `typ::Integer`: Initial condition to randomly generates: gaussian random, high dimensional gaussian random
- `init_kwargs`: other keywords arguments for initial functions
"""
function kdv_snapshot_generator(tₘₐₓ, tₘᵢₙ, xₘₐₓ, xₘᵢₙ, tₙ, xₙ, typ, init_kwargs=(;))
  t = LinRange(tₘᵢₙ, tₘₐₓ, tₙ);
  x = LinRange(xₘᵢₙ, xₘₐₓ , xₙ);
  Δx = round((xₘₐₓ - xₘᵢₙ) / (xₙ - 1), digits=8);

  rand_init = rand((1, 3));
  if typ > 0
    rand_init = typ;
  end

  init = Dict([
    (1, (a, b) -> InitialFunctions.random_init(a, b)),
    (2, (a, b) -> InitialFunctions.high_dim_random_init2(a, b, init_kwargs...)), # a, b, m=20
    (3, (a, b) -> InitialFunctions.random_gaussian_init(a, b, init_kwargs...)), # a, b, mu, sigma
  ]);

  u₀ = copy(init[rand_init](t, x));
  t, u = Equations.get_kdv_fd(t, Δx, u₀[1, :]);

  if sum(isfinite.(u)) != prod(size(u))
    print("u matrix is not finite.")
  end

  return t, u
end


"""
  generate_kdv_training_dataset(n, upscale, filename, snap_kwarg, init_kwargs)

Generate a dataset of solution to heat equation.

# Arguments
- `n::Integer`: number of snapshots to generates
- `upscale::Integer`: Size of upscaling for fine grid solution generated: (upscale * t_n, upscale * x_n)
- `filename::String`: file name saved
- `snap_kwargs`: t_max, t_mub, x_max, x_min, t_n, x_n, typ (of initial conditions)
- `init_kwargs`: other keywords arguments for initial functions
"""
function generate_kdv_dataset(
    n=64,
    upscale=64,
    filename="kdv.jld2",
    snap_kwargs=(;),
    init_kwargs=(;)
    )

    tₘₐₓ, tₘᵢₙ, xₘₐₓ, xₘᵢₙ, tₙ, xₙ, typ = snap_kwargs;
    x = LinRange(xₘᵢₙ, xₘₐₓ , xₙ);
    Δx = round((xₘₐₓ - xₘᵢₙ) / (xₙ - 1), digits=8);

    G(Δ, x) = √(6 / π) / Δ * exp(-6x^2 / Δ^2)
    Δ = 4Δx
    W = sum(G.(Δ, x .- x' .- z) for z ∈ -2:2)
    W = W ./ sum(W; dims = 2)

    train_set = [];
    for i in range(1, n, step=1)
      print("Generating snapshot ", i, "...")

      tₕ, u = kdv_snapshot_generator(tₘₐₓ, tₘᵢₙ, xₘₐₓ, xₘᵢₙ, tₙ, xₙ * upscale, typ, init_kwargs); # tₙ * upscale
      û = ProcessingTools.downsampling_x(u, upscale);

      item = [tₕ, û, snap_kwargs, init_kwargs]
      push!(train_set, item);

      println("Done")
    end

    if !isempty(filename)
      JLD2.save(filename, "training_set", train_set);
    end

    return train_set
  end
