module Reg

using CUDA
using LinearAlgebra

function l1(w, λ=1e-3)
  λ * sum(abs.(w))
end

function l2(w, λ=1e-3)
  λ * sum(w.^2)
end

function augment(x, ϵ=1e-6)
  x + ϵ .* randn(size(x))
end

function gaussian_augment(x::AbstractArray{T, N}, ρ=1e-6) where {T,N}
  ϵ = ρ .* x
  # noise = ϵ .* randn(size(x))
  x isa CuArray ? (noise = ϵ .* CUDA.randn(T, size(x))) : (noise = ϵ .* randn(T, size(x)))
  return x + noise
end

end
