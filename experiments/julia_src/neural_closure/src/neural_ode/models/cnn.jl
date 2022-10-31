using Flux

include("blocks.jl")

function CNN(x_n, l, k)
  hidden = []
  for i in range(1, l, step=1)
    layer = Flux.Conv((k, 1), 1 => 1, tanh; stride = 1, pad = SamePad(), bias=true, init=Flux.glorot_uniform);
    push!(hidden, layer);
  end

  return Flux.Chain(
    hidden...,
    Flux.Dense(x_n => x_n, identity; init=Flux.glorot_uniform, bias=true),
  )
end

function CNN2(k, channels)
  hidden = []

  for (c1, c2) in zip(channels[1:end-2], channels[2:end-1])
    kernel = Models.glorot_uniform_float64(k, c1, c2); # (k,) , c1 => c2
    layer = Flux.Conv(kernel, true, tanh; stride = 1, pad = SamePad());
    # layer = Flux.Conv((k,), c1 => c2, tanh; stride = 1, pad = SamePad(), bias=true, init=Flux.glorot_uniform);
    push!(hidden, layer);
  end

  # (nx, 1, nsample) -> CNN -> (nx, 1, nsample)
  kernel = Models.glorot_uniform_float64(k, channels[end-1], channels[end]);
  return Flux.Chain(
    x -> Block.Extend(x, Int8(floor(k / 2))),
    x -> Block.Power2(x),
    hidden...,
    Flux.Conv(kernel, true, identity; stride = 1, pad = SamePad()),
    # Flux.Conv((k,), channels[end-1] => channels[end], identity; stride = 1, pad = SamePad(), bias=true, init=Flux.glorot_uniform),
    x -> Block.Reduce(x, Int8(floor(k / 2))),
  )
end

function CNN3(k, channels)
  hidden = []

  for (c1, c2) in zip(channels[1:end-2], channels[2:end-1])
    layer = Flux.Conv((k,), c1 => c2, tanh; stride = 1, pad = SamePad(), bias=true, init=Flux.glorot_uniform);
    push!(hidden, layer);
  end

  # (nx, 1, nsample) -> CNN -> (nx, 1, nsample)
  return Flux.Chain(
    x -> Block.Extend(x, Int8(floor(k / 2))),
    x -> Block.Power3(x),
    hidden...,
    Flux.Conv((k,), channels[end-1] => channels[end], identity; stride = 1, pad = SamePad(), bias=true, init=Flux.glorot_uniform),
    x -> Block.Reduce(x, Int8(floor(k / 2))),
  )
end
