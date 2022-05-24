module Models

using Flux
using DiffEqFlux

function HeatModel(x_n)
  return FastChain(
    DiffEqFlux.FastDense(x_n, x_n, identity; bias=false, initW=Flux.glorot_uniform)
  )
end

end
