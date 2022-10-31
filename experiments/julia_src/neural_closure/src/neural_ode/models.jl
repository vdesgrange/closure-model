module Models

using Random
using Flux

function glorot_uniform_float64(rng::AbstractRNG, dims::Integer...; gain::Real=1)
    scale = gain * sqrt(24.0 / sum(Flux.nfan(dims...)))
    (rand(rng, Float64, dims...) .- 0.5) .* scale
end
glorot_uniform_float64(dims::Integer...; kw...) = glorot_uniform_float64(Flux.rng_from_array(), dims...; kw...)
glorot_uniform_float64(rng::AbstractRNG=Flux.rng_from_array(); init_kwargs...) = (dims...; kwargs...) -> glorot_uniform_float64(rng, dims...; init_kwargs..., kwargs...)

include("models/linear.jl");
include("models/fnn.jl");
include("models/cnn.jl");
include("models/cae.jl");
include("models/others.jl");

end
