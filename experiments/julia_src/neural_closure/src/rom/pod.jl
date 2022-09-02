module POD

using LinearAlgebra
using Statistics

struct Basis{T}
  modes::Matrix{T}
  coefficients::Matrix{T}
  eigenvalues::Vector{T}
end

  """
    generate_pod_basis(M, substract_mean::Bool = false)

  Generate modes of Proper orthogonal decomposition. Return basis datastructure composed of modes and coefficients.

  # Reference
  ```
  Shady E. Ahmed, Suraj Pawar, Omer San, Adil Rasheed, Traian Iliescu, and Bernd R.Noack.
  On closures for reduced order models—a spectrum of first-principle to machine-learned avenues
  ```
  """
function generate_pod_basis(M, substract_mean::Bool = false)
  n = size(M, 2)
  S = copy(M);

  sm = mean(S, dims=2);
  if substract_mean
      S .-= sm;
  end

  C = S'S;
  E = eigen(C);
  λ = E.values;
  W = E.vectors;

  idx = sortperm(abs.(λ) / n, rev=true)
  λ = λ[idx]
  W = W[:, idx]

  D = sqrt.(abs.(λ))
  θ = real(S * W) * Diagonal(1 ./ D) # Modes
  A = θ' * S # Coefficients

  return Basis(θ, A, λ), sm
end

"""
  get_energy()

Quality of information is measured using relative information, aka. energy.

# Arguments
- `λ::Vector{Number}` : eigenvalues of correlation matrix C
- `m::Integer` : number of modes
"""
function get_energy(λ::Vector{<:Real}, m::Integer)
  return sum(λ[1:m]; dims=1) / sum(λ; dims=1);
end

function get_energy2(λ::Vector{<:Real}, m::Integer)
return sum(λ[m+1:end]; dims=1) / sum(λ.^2; dims=1);
end

"""
  get_relative_projection_err(x̂, x)

Quality of information is measured using relative projection error.
Similar to energy

# Arguments
- `x̂::Matrix` : matrix of reduced snapshots
- `x::Matrix` : matrix of snapshots
"""
function get_relative_projection_err(x̂, x)
  return sum(sum(abs.(x .- x̂); dims=1).^2; dims=2) / sum(sum(abs.(x); dims=1).^2; dims=2);
end

end
