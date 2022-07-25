module Objectives

using Statistics

function l2(ŷ, y)
  _check_sizes(ŷ, y)
  sqrt(sum(abs2, ŷ .- y) .^ 2)
end

function derivative_fitting(K, ŷ, y, Δt)
  """
    Derivative fitting when y is a snapshot and ∂y∂t unknown.
  """
  ∂y∂t = (y[:, 3:end] - y[:, 1:end-2]) / (2 * Δt)
  return mean((K(ŷ) .- ∂y∂t) .^ 2)
end

function energy(ŷ, y)
  _check_sizes(ŷ, y)
end

end
