module BurgersDirect

using CUDA
using BSON: @save
using Flux
using OrdinaryDiffEq
using DiffEqFlux

include("../../utils/generators.jl")
include("../../utils/processing_tools.jl")
include("../../neural_ode/models.jl")
include("../../neural_ode/regularization.jl")
include("../../neural_ode/training.jl")
include("./analysis.jl")

function training(model, epochs, dataset, batch_size, ratio, lr=0.01, noise=0., reg=0., cuda=false)
   if cuda && CUDA.has_cuda()
      device = Flux.gpu
      CUDA.allowscalar(false)
      @info "Training on GPU"
  else
      device = Flux.cpu
      @info "Training on CPU"
  end
  model = model |> device;

  opt = Flux.Optimiser(Flux.WeightDecay(reg), Flux.ADAM(lr, (0.9, 0.999), 1.0e-8))
  ltrain = 0.;
  lval = 0.;
  losses = [];

  @info("Loading dataset")
  (train_loader, val_loader) = ProcessingTools.get_data_loader(dataset, batch_size, ratio, false);

  @info("Building model")
  p, re = Flux.destructure(model);
  net(u, p, t) = re(p)(u);

  prob = ODEProblem{false}(net, Nothing, (Nothing, Nothing));

  function predict_neural_ode(x, t)
    tspan = (t[1], t[end]);
    _prob = remake(prob; u0=x, p=p, tspan=tspan);
    Array(solve(_prob, AutoTsit5(Rosenbrock23()), u0=x, p=p, saveat=t));
  end

  function loss(x, y, t)
    u_pred = predict_neural_ode(x, t[1]);
    ŷ = Reg.gaussian_augment(u_pred, noise);
    l = Flux.mse(ŷ, permutedims(y, (1, 3, 2))) # + Reg.l2(p, reg);
    return l;
  end

  function traincb()
    ltrain = 0;
    for (x, y, t) in train_loader
      # (x, y, t) = (x, y, t) |> device;
      ltrain += loss(x, y, t);
    end
    ltrain /= (train_loader.nobs / train_loader.batchsize);
    @show(ltrain);
  end

  function val_loss(x, y, t)
    u_pred = predict_neural_ode(x, t[1]);
    ŷ = u_pred;
    l = Flux.mse(ŷ, permutedims(y, (1, 3, 2)))
    return l;
  end

  function evalcb()
    lval = 0;
    for (x, y, t) in val_loader
      (x, y, t) = (x, y, t) |> device;
      lval += val_loss(x, y, t);
    end
    lval /= (val_loader.nobs / val_loader.batchsize);
    @show(lval);
  end

  @info("Train")
  trigger = Flux.plateau(() -> ltrain, 20; init_score = 1, min_dist = 1f-5);
  Flux.@epochs epochs begin
    Flux.train!(loss, Flux.params(p), train_loader, opt, cb = [traincb, evalcb]);
    trigger() && break;
  end

  return re(p), p, ltrain, lval
end

function main()
  x_n = 64;
  batch_size = 128;
  epochs = 10;

  data = Generator.read_dataset("./dataset/burgers_high_dim_training_set.jld2")["training_set"];
  # model = Models.BasicAutoEncoder(x_n);
  model = Models.FeedForwardNetwork(x_n, 2, x_n);
  K, p, _, _ = training(model, epochs, data, batch_size, 0.5, 0., 0., true);
  # @save "./models/BurgersLinearModel.bson" K

  BurgersAnalysis.check_result(K, p, 2)

  return K, p
end

end
