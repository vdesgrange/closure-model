module Objectives

using Flux
using Statistics
using LinearAlgebra

function _check_sizes(ŷ, y)
  @assert size(y) == size(ŷ)
end

function mser(ŷ, y)
  _check_sizes(ŷ, y)
  t_n = size(y)[2]
  r = collect(LinRange(0.1, 1., t_n))
  mean(abs2, r.*(ŷ .- y) .^ 2)
end

function rmse(ŷ, y)
  _check_sizes(ŷ, y)
  sqrt(mean(abs2, (ŷ .- y).^2))
end

function nre(ŷ, y)
  _check_sizes(ŷ, y)
  norm(ŷ .- y) / norm(y)
end

function Δ_loss(K, ŷ, y, Δt)
  """
    Δ_loss(K, ŷ, y, Δt)

    Derivative fitting when y is a snapshot and ∂y∂t unknown.
  """
  ∂y∂t = (y[:, 2:end] - y[:, 1:end-1]) / Δt
  return Flux.mse(K(ŷ), ∂y∂t)
end

function momentum(ŷ)
  return sum(sum(ŷ[:, 2:end, :]; dims=1) - sum(ŷ[:, 1:end-1, :]; dims=1)) + sum(sum(ŷ[:, end, :]; dims=2) - sum(ŷ[:, 1, :]; dims=2))
end

function energy(ŷ, y)
  _check_sizes(ŷ, y)
end

end
