module POD

using LinearAlgebra
using Statistics

struct Basis{T}
  modes::Matrix{T}
  coefficients::Matrix{T}
end

function generate_pod_basis(S, substract_mean::Bool = false)
  n = size(S, 2)

  if substract_mean
    S .-= mean(S, dim=1);
  end

  C = S'S;
  E = eigen(C);
  λ = E.values;
  W = E.vectors;

  idx = sortperm(abs.(λ) / n, rev=true)
  λ = λ[idx]
  W = W[:, idx]

  D = sqrt.(abs.(λ))
  θ = S * W * Diagonal(1 ./ D) # Modes
  A = θ' * S # Coefficients

  return Basis(θ, A)
end

end
