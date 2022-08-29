using Flux

include("blocks.jl")

function FeedForwardNetwork(x_n, l, n)
  """
    FeedForwardNetwork(x_n, l, n)

  Create a FeedForwardNetwork flux chain model. Number of layers and neurons are variables.
  Glorot uniform used for initialization

  # Arguments
  - `x_n::Integer`: input/output dimension
  - `l::Integer`: number of hidden layers
  - `n::Integer`: number of neurons in hidden layers
  """

  hidden = []
  for i in range(1, l, step=1)
    layer = Flux.Dense(n => n, tanh; init=Flux.glorot_uniform, bias=true);
    push!(hidden, layer);
  end

  return Flux.Chain(
    Flux.Dense(x_n => n, tanh; init=Flux.glorot_uniform, bias=true),
    hidden...,
    Flux.Dense(n => x_n, identity; init=Flux.glorot_uniform, bias=true),
  )
end

function FNNM(x_n, l, n)
  """
    FNNM(x_n, l, n)

  Create a FeedForwardNetwork flux chain model. Number of layers and neurons are variables.
  Glorot uniform used for initialization

  # Arguments
  - `x_n::Integer`: input/output dimension
  - `l::Integer`: number of hidden layers
  - `n::Integer`: number of neurons in hidden layers
  """

  hidden = []
  for i in range(1, l, step=1)
    layer = Flux.Dense(n => n, tanh; init=Flux.glorot_uniform, bias=true);
    push!(hidden, layer);
  end

  return Flux.Chain(
    Flux.Dense(x_n => n, tanh; init=Flux.glorot_uniform, bias=true),
    hidden...,
    Flux.Dense(n => x_n, identity; init=Flux.glorot_uniform, bias=true),
    x -> Blocks.MomentumX(x),
  )
end
