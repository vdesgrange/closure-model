module Models

using Flux
using DiffEqFlux

function HeatModel(x_n)
  return FastChain((x, p) ->
    FastDense(x_n,
      x_n,
      activation=identity, 
      initW = Flux.glorot_uniform, 
      initb = Flux.zeros32
    )
  )
end

end
