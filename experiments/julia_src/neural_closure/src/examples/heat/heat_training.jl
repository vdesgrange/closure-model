module HeatTraining

using DiffEqFlux
using OrdinaryDiffEq
using DifferentialEquations

include("objectives.jl")
include("models.jl")

function f(u, K, t)
  return K * u
end

function S(net, u0, t)
  tspan = (t[1], t[end])
  prob = ODEProblem(ODEFunction(f), copy(u0), tspan, net)
  sol = solve(prob, Tsit5(), saveat=t, reltol=1e-8, abstol=1e-8)
end

function training_with_solver(K, epochs, u0, u_true, tsnap)
  optimizer = DiffEqFlux.ADAM(0.01, (0.9, 0.999), 1.0e-8)

  callback(theta, loss, u) = (display(loss); false)

  function loss(K)
    u_pred = Array(S(K, u0, tsnap))
    l = Objectives.mseloss(u_pred, u_true)
    return l
  end

  result = DiffEqFlux.sciml_train(loss, K, optimize; cb = callback, maxiters = epochs);
  return result
end


end
