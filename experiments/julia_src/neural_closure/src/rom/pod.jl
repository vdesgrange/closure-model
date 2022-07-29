module POD

using LinearAlgebra
using Statistics

struct Basis{T}
  modes::Matrix{T}
  coefficients::Matrix{T}
end

function generate_pod_basis(M, substract_mean::Bool = false)
  """
    generate_pod_basis(M, substract_mean::Bool = false)

  Generate modes of Proper orthogonal decomposition. Return basis datastructure composed of modes and coefficients.

  # Reference
  ```
  Shady E. Ahmed, Suraj Pawar, Omer San, Adil Rasheed, Traian Iliescu, and Bernd R.Noack.
  On closures for reduced order models—a spectrum of first-principle to machine-learned avenues
  ```
  """
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

  return Basis(θ, A), sm
end

end
