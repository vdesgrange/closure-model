module Models

using Flux
using DiffEqFlux

function HeatModel(x_n)
  return FastChain(
    DiffEqFlux.FastDense(x_n, x_n, identity; bias=false, initW=Flux.zeros32)
  )
end

function BasicAutoEncoder(x_n)
  return FastChain(
    DiffEqFlux.FastDense(x_n, round(Int64, x_n / 2), tanh; bias=true, initW=Flux.kaiming_uniform, initb = Flux.zeros32),
    DiffEqFlux.FastDense(round(Int64, x_n / 2), round(Int64, x_n / 4), tanh; bias=true, initW=Flux.kaiming_uniform, initb = Flux.zeros32),
    DiffEqFlux.FastDense(round(Int64, x_n / 4), round(Int64, x_n / 2), tanh; bias=true, initW=Flux.kaiming_uniform, initb = Flux.zeros32),
    DiffEqFlux.FastDense(round(Int64, x_n / 2), x_n, tanh; bias=true, initW=Flux.kaiming_uniform, initb = Flux.zeros32),
  )

end

end
