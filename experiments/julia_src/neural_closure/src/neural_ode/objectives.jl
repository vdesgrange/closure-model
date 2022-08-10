module Objectives

using Flux
using Statistics

function _check_sizes(ŷ, y)
  @assert size(y) == size(ŷ)
end

function mser(ŷ, y)
  _check_sizes(ŷ, y)
  t_n = size(y)[2]
  r = collect(LinRange(0.1, 1., t_n))
  mean(abs2, r.*(ŷ .- y) .^ 2)
end

function Δ_loss(K, ŷ, y, Δt)
  """
    Δ_loss(K, ŷ, y, Δt)

    Derivative fitting when y is a snapshot and ∂y∂t unknown.
  """
  ∂y∂t = (y[:, 3:end] - y[:, 1:end-2]) / (2 * Δt)
  return Flux.mse(K(ŷ), ∂y∂t)
end

function energy(ŷ, y)
  _check_sizes(ŷ, y)
end

end
