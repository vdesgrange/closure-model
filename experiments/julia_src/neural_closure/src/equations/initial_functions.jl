module InitialFunctions

export gaussian_init, random_init, high_dim_random_init, burgers_analytical_init, heat_analytical_init

using LinearAlgebra
using SparseArrays
using OrdinaryDiffEq
using DiffEqFlux
using Random
using Distributions

function gaussian_init(t, x)
  u = zeros(Float64, size(t, 1), size(x, 1))
  u[1, :] .= exp.(-(x .- 1).^2)
  u[:, 1] .= 0
  u[:, end] .= 0
  return u
end

function gaussian2_init(t, x, σ, μ)
  u = zeros(Float64, size(t, 1), size(x, 1))
  u[1, :] .= 1 ./ (σ * sqrt.(2 * pi)) * exp.(-0.5 .* ((x .- μ) ./ σ).^2)
  u[:, 1] .= 0
  u[:, end] .= 0
  return u
end

function random_gaussian_init(t, x, σ, μ)
  d = Normal(1., 0.5)
  σ2 = σ * rand(d)
  μ2 = μ * rand(d)
  return gaussian2_init(t, x, σ2, μ2)
end

function random_init(t, x)
  u = zeros(Float64, size(t, 1), size(x, 1))
  d = Normal(0, .25)
  nu = rand(d, size(x, 1))
  u[1, :] .= sin.(x) + nu
  u[:, 1] .= 0
  u[:, end] .= 0

  return u
end

function high_dim_random_init(t, x, m=28)
  d = Normal(0., 1.)
  nu = rand(d, 2 * m)
  s = [nu[2 * k] * sin.(k * x) + nu[2 * k - 1] * cos.(k * x) for k in range(1, m, step=1)]

  u0 = zeros(Float64, size(t, 1), size(x, 1))
  u0[1, :] .= (1 / sqrt(m)) .* sum(s)
  u0[:, 1] .= 0
  u0[:, end] .= 0

  return u0
end

function high_dim_random_init2(t, x, m=28)
  d = Normal(0., 1.)
  nu = rand(d, 2 * m)
  x = (x .- x[1]) ./ (x[end] .- x[1])
  s = [nu[2 * k] * sin.(2 * pi * k * x) + nu[2 * k - 1] * cos.(2 * pi * k * x) for k in range(1, m, step=1)]

  u0 = zeros(Float64, size(t, 1), size(x, 1))
  u0[1, :] .= (1 / sqrt(m)) .* sum(s)

  return u0
end


function analytical_burgers_1d(t, x, nu)
end

function burgers_analytical_init(t, x, nu)
  u0 = zeros(Float64, size(t, 1), size(x, 1))
  # u = analytical_burgers_1d(t, x, nu)
  # u0[1, :] .= u[1, :]
  # u0[:, 1] .= 0
  # u0[:, end] .= 0
  return u0
end

function advecting_shock(t, x, nu)
  Re = 1. / nu;
  t0 = exp(Re / 8.);

  u0 = zeros(Float64, size(t, 1), size(x, 1));
  u0[1, :] .= x ./ (1 .+ sqrt(1. / t0) * exp.(Re * (x.^2 ./ 4)));
  u0[:, 1] .= 0
  u0[:, end] .= 0

  return u0
end

function analytical_heat_1d(t, x, n=[], c=[], κ=1.)
  xₘᵢₙ, xₘₐₓ = x[1], x[end];
  L = xₘₐₓ - xₘᵢₙ;
  N = size(n)[1];
  k = 1:N;

  if (size(c, 1) == 0)
    c = Array(randn(N) ./ k);  # n = range(1, n_max, step=1), 1 to avoid division by 0
  end

  aₖ(k, x) = sqrt(2 / L) * sin(π * k * (x - xₘᵢₙ) / L);
  u(t, x) = sum(cₖ * exp(- κ * π^2 * k^2 / L * t) * aₖ(k, x) for (cₖ, k) in zip(c, k))
  return Array([u(a, b) for a in t, b in x]) 
end


function heat_analytical_init(t, x, n=[], c=[], κ=1.)
  u₀ = zeros(Float64, size(t, 1), size(x, 1))
  u = analytical_heat_1d([t[1]], x, n, c, κ)
  u₀[1, :] = copy(u[1, :])
  u₀[:, 1] .= 0
  u₀[:, end] .= 0

  return u₀
end

end
