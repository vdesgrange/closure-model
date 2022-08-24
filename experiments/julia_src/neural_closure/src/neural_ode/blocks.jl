module Block

using LinearAlgebra
using Flux
using DiffEqFlux

function MomentumT(x) # t
  X = x[:, 2:end, :] .- x[:, 1:end-1, :]
  Y = x[:, end, :] .- x[:, 1, :];
  Z = reshape(Y, (size(Y)[1], 1, size(Y)[2]))
  return hcat(X, Z)
end

function MomentumX(x::AbstractArray{T, N}) where {T, N} # t
  X = x[2:end, :, :, :] .- x[1:end-1, :, :, :]
  Y = x[end, :, :] .- x[1, :, :];
  Z = reshape(Y, (1, size(Y)...))
  return vcat(X, Z)
end

function Extend(x::AbstractArray{T, N}, k::Int8) where {T, N}
  # y = deepcopy(x)
  # z = zeros(size(y)[1] + 2 * k, size(y)[2:end]...);
  # z[1+k:end-k, :, :, :] = y;
  # z[1:k, :, :, :] = y[end-k+1:end, :, :, :];
  # z[end-k+1:end, :, :, :] = y[1:k, :, :, :];
  # return deepcopy(z)
  vcat(x[end-k+1:end, :, :, :], x, x[1:k, :, :, :])
end

function Reduce(x::AbstractArray{T, N}, k::Int8) where {T, N}
  y = deepcopy(x)
  return y[k+1:end-k, :, :, :]
end

function Power2(x::AbstractArray{T, N}) where {T, N}
  Array{T}(cat(deepcopy(x), deepcopy(x.^2); dims=3))
end

end
