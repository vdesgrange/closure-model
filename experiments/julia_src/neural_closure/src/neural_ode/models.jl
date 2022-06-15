module Models

using Flux
using DiffEqFlux

function HeatModel(x_n)
  return FastChain(
    DiffEqFlux.FastDense(x_n, x_n, identity; bias=false, initW=Flux.zeros32)
  )
end

function BasicFNN(x_n)
  return FastChain(
    DiffEqFlux.FastDense(x_n, x_n, tanh; bias=true, initW=Flux.kaiming_uniform, initb = Flux.zeros32),
    DiffEqFlux.FastDense(x_n, x_n, tanh; bias=true, initW=Flux.kaiming_uniform, initb = Flux.zeros32),
    DiffEqFlux.FastDense(x_n, x_n, identity; bias=true, initW=Flux.kaiming_uniform, initb = Flux.zeros32),
  )
end

function FastBasicAutoEncoder(x_n)
  return FastChain(
    DiffEqFlux.FastDense(x_n, round(Int64, x_n / 2), tanh; bias=true, initW=Flux.kaiming_uniform, initb = Flux.zeros32),
    DiffEqFlux.FastDense(round(Int64, x_n / 2), round(Int64, x_n / 4), tanh; bias=true, initW=Flux.kaiming_uniform, initb = Flux.zeros32),
    DiffEqFlux.FastDense(round(Int64, x_n / 4), round(Int64, x_n / 2), tanh; bias=true, initW=Flux.kaiming_uniform, initb = Flux.zeros32),
    DiffEqFlux.FastDense(round(Int64, x_n / 2), x_n, identity; bias=true, initW=Flux.kaiming_uniform, initb = Flux.zeros32),
  )
end


function LinearModel(x_n)
  return Flux.Chain(
    Flux.Dense(x_n, x_n, identity; bias=false, init=Flux.zeros32)
  );
end

function BasicAutoEncoder(x_n)
  return Flux.Chain(
    Flux.Dense(x_n, round(Int64, x_n / 2), tanh; bias=true, init=Flux.kaiming_uniform),
    Flux.Dense(round(Int64, x_n / 2), round(Int64, x_n / 4), tanh; bias=true, init=Flux.kaiming_uniform),
    Flux.Dense(round(Int64, x_n / 4), round(Int64, x_n / 2), tanh; bias=true, init=Flux.kaiming_uniform),
    Flux.Dense(round(Int64, x_n / 2), x_n, identity; bias=true, init=Flux.kaiming_uniform),
  )
end

end
