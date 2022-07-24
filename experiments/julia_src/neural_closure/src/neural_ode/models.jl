module Models

using Flux
using DiffEqFlux

function HeatModel(x_n)
  return FastChain(
    DiffEqFlux.FastDense(x_n, x_n, identity; bias=false, initW=Flux.zeros32)
  )
end

function LinearModel(x_n)
  """
    LinearModel(x_n)

  Create a linear flux chain model. Use for testing with linear heat equation.

  # Arguments
  - `x_n::Integer`: input/output dimension
  """
  return Flux.Chain(
    Flux.Dense(x_n, x_n, identity; bias=false, init=Flux.zeros32)
  );
end

function BasicAutoEncoder(x_n)
  """
    BasicAutoEncoder(x_n)

  Create a basic auto encoder flux chain model.

  # Arguments
  - `x_n::Integer`: input/output dimension
  """
  return Flux.Chain(
    Flux.Dense(x_n, round(Int64, x_n / 2), tanh; bias=true, init=Flux.kaiming_uniform),
    Flux.Dense(round(Int64, x_n / 2), round(Int64, x_n / 4), tanh; bias=true, init=Flux.kaiming_uniform),
    Flux.Dense(round(Int64, x_n / 4), round(Int64, x_n / 2), tanh; bias=true, init=Flux.kaiming_uniform),
    Flux.Dense(round(Int64, x_n / 2), x_n, identity; bias=true, init=Flux.kaiming_uniform),
  )
end

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
  for i in l
    layer = Flux.Dense(n, n, tanh; bias=true, init=Flux.glorot_uniform);
    push!(hidden, layer);
  end

  return Flux.Chain(
    Flux.Dense(x_n, n, tanh; bias=true, init=Flux.glorot_uniform),
    hidden...,
    Flux.Dense(n, x_n, identity; bias=true, init=Flux.glorot_uniform),
  )
end

end
