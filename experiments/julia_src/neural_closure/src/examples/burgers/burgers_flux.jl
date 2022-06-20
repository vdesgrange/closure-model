module BurgersTraining

using CUDA
using BSON: @save
using Plots
using Flux
using FluxTraining
using OrdinaryDiffEq
using DiffEqFlux
using GalacticOptim
using MLUtils
using IterTools: ncycle

include("../../utils/generators.jl")
include("../../utils/processing_tools.jl")
include("../../utils/graphic_tools.jl")
include("../../neural_ode/models.jl")

function check_result(nn, res, typ)
    t, u0, u = Generator.get_burgers_batch(0.2, 0., 1., 0., 64, 64, 0.01, typ);
    prob_neuralode = DiffEqFlux.NeuralODE(nn, (t[1], t[end]), Tsit5(), saveat=t);
    u_pred = prob_neuralode(u0, res);

    plot(
        GraphicTools.show_state(u, ""),
        GraphicTools.show_state(hcat(u_pred.u...), ""),
        GraphicTools.show_err(hcat(u_pred.u...), u, "");
        layout = (1, 3),
    )
end

function get_data_loader(dataset, batch_size, ratio)
  t, init_set, true_set = ProcessingTools.process_dataset(dataset, false);

  t_train, t_val = splitobs(t, at = ratio);
  train_set, val_set = splitobs(true_set, at = ratio);
  init_train = copy(init_set);
  init_val = copy(val_set[:, :, 1]);

  switch_train_set = permutedims(train_set, (1, 3, 2));
  switch_val_set = permutedims(val_set, (1, 3, 2));

  train_loader = DataLoader((init_train, switch_train_set, collect(ncycle([collect(t_train)], batch_size))), batchsize=batch_size, shuffle=false);
  val_loader = DataLoader((init_val, switch_val_set, collect(ncycle([collect(t_val)], batch_size))), batchsize=batch_size, shuffle=false);

  return (train_loader, val_loader)
end

function training(model, epochs, dataset, batch_size, ratio)
  opt = Flux.ADAM(0.01, (0.9, 0.999), 1.0e-8);

  @info("Loading dataset")
  (train_loader, val_loader) = get_data_loader(dataset, batch_size, ratio);

  @info("Building model")
  p, re = Flux.destructure(model);
  net(u, p, t) = re(p)(u);

  prob = ODEProblem{false}(net, Nothing, (Nothing, Nothing));

  function predict_neural_ode(x, t)
      tspan = (t[1], t[end]);
      _prob = remake(prob; u0=x, p=p, tspan=tspan);
      Array(solve(_prob, Tsit5(), u0=x, p=p, saveat=t));
  end

  function loss(x, y, t)
      u_pred = predict_neural_ode(x, t[1]);
      l = Flux.mse(u_pred, permutedims(y, (1, 3, 2)));
      return l;
  end

  function traincb()
      ltrain = 0;
      for (x, y, t) in train_loader
           ltrain += loss(x, y, t);
      end
      ltrain /= (train_loader.nobs / train_loader.batchsize);
      @show(ltrain);
  end

  function evalcb()
      lval = 0;
      for (x, y, t) in val_loader
           lval += loss(x, y, t);
      end
      lval /= (val_loader.nobs / val_loader.batchsize);
      @show(lval);
  end

  @info("Train")
  Flux.@epochs epochs Flux.train!(loss, Flux.params(p), train_loader, opt, cb = [traincb, evalcb]);

  return model, p
end

function __init__()
end

function main()
  # t_max = 0.2;
  # t_min = 0.;
  # x_max = 1.;
  # x_min = 0.;
  # t_n = 64;
  x_n = 64;
  batch_size = 32;
  epochs = 100;

  high_dataset = Generator.read_dataset("./dataset/burgers_high_dim_training_set.jld2")["training_set"];
  model = Models.BasicAutoEncoder(x_n);
  neural_operator, p = training(model, epochs, high_dataset, batch_size, 0.5);
  # @save "LinearModel.bson" cpu(neural_operator)

  check_result(neural_operator, p, 2)
end


end
