module Training

using DiffEqFlux
using OrdinaryDiffEq

include("../src/utils/generators.jl")
include("objectives.jl")
include("models.jl")

function f(u, K, t)
  return K * u
end

function S(net, u0, u_true, t)
  tspan = (t[1], t[end])
  prob = ODEProblem(ODEFunction(net), copy(u0), tspan)
  sol = solve(prob, Tsit5(), saveat=t, reltol=1e-8, abstol=1e-8)
end

function process_dataset(dataset)
  # todo - split between training and validation data
  init_set = []
  true_set = []
  for i in range(1, size(dataset, 1), step=1)
    t, u, _, _ = dataset[i]
    u0 = u[1, :]
    push!(init_set, u0)
    push!(true_set, u)
  end

  return t, init_set, true_set
end

function callback(theta, loss, u_pred)
  display(loss)
  pl = scatter(t, dataset[1,:], label = "data")
end

function heat_training(net, epochs, dataset)
  optimizer = DiffEqFlux.ADAM(0.01, (0.9, 0.999), 1.0e-8)
  t, u0, u_true = process_dataset(dataset)

  tspan = (t[1], t[end])
  neural_ode = NeuralODE(net, tspan, Tsit5(), saveat=t)

  function predict_neural_ode(theta)
    return Array(neural_ode(u0, theta))
  end

  function loss(theta)
    u_pred = predict_neural_ode(theta)
    l = Objectives.mseloss(u_pred, u_true)
    return l, u_pred
  end

  result = DiffEqFlux.sciml_train(loss, neural_ode.p, optimizer; cb = callback, maxiters = epochs)
end

end
