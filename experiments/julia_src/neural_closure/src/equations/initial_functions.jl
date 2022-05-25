module InitialFunctions

export random_init, high_dim_random_init, burgers_analytical_init, heat_analytical_init

using Revise
using LinearAlgebra
using SparseArrays
using OrdinaryDiffEq
using DiffEqFlux
using Random
using Distributions

function gaussian_init(t, x)
  u = zeros(Float64, size(t, 1), size(x, 1))
  u[1, :] .= exp.(-(x - 1).^2)
  u[:, 1] .= 0
  u[:, end] .= 0
  return u
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

function high_dim_random_init(t, x, m=48)
  d = Normal(0., 1.)
  nu = rand(d, 2 * m)
  s = [nu[2 * k] * sin.(k * x) + nu[2 * k - 1] * cos.(k * x) for k in range(1, m, step=1)]

  u = zeros(Float64, size(t, 1), size(x, 1))
  u[1, :] .= (1 / sqrt(m)) * sum(s)
  u[:, 1] .= 0
  u[:, end] .= 0

  return u
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

function analytical_heat_1d(t, x, n=[], c=[], k=1.)
  L = 1.

  if (size(c, 1) == 0)
    c = randn(size(n, 1)) ./ n # n = range(1, n_max, step=1), 1 to avoid division by 0
  end

  X(n, x) = sqrt(2 / L) * sin(pi * n * x / L);
  u(x, t) = sum(c * exp(-k * (pi * n / L)^2 * t) * X(n, x) for (c, n) in zip(c, n))
  return [u(b, a) for a in t, b in x];

  # a = c' .* exp.(- k * (n' * pi / L).^2 .* t) 
  # b = sqrt(2 / L) * sin.(n' .* x * pi / L)
  # return a * b'
end


function heat_analytical_init(t, x, n=[], c=[], k=1.)
  u0 = zeros(Float64, size(t, 1), size(x, 1))
  u = analytical_heat_1d([t[1]], x, n, c, k)
  u0[1, :] = copy(u[1, :])
  u0[:, 1] .= 0
  u0[:, end] .= 0

  return u0
end

end
