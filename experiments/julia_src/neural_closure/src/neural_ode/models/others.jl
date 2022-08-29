using Flux
using DiffEqFlux

function HeatModel(x_n)
  return FastChain(
    DiffEqFlux.FastDense(x_n, x_n, identity; bias=false, initW=Flux.zeros32)
  )
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

