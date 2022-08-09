module BurgersCombined

using CUDA
using BSON: @save
using Flux
using OrdinaryDiffEq
using DiffEqFlux
using Optimization
using OptimizationOptimisers
using IterTools: ncycle

include("../../utils/generators.jl")
include("../../utils/processing_tools.jl")
include("../../neural_ode/models.jl")
include("../../neural_ode/regularization.jl")
include("../../neural_ode/training.jl")
include("./analysis.jl")

function training(model, epochs, dataset, batch_size, ratio, lr=0.01, noise=0., reg=0., cuda=true)
   if cuda && CUDA.has_cuda()
      device = Flux.gpu
      CUDA.allowscalar(false)
      @info "Training on GPU"
  else
      device = Flux.cpu
      @info "Training on CPU"
  end

  ltrain = 0.;
  lval = 0.;

  @info("Loading dataset")
  (train_loader, val_loader) = ProcessingTools.get_data_loader(dataset, batch_size, ratio, false, cuda);

  @info("Building model")
  model_gpu = model |> device;
  p, re = Flux.destructure(model_gpu);
  p = p |> device;
  net(u, p, t) = re(p)(u);

  prob = ODEProblem{false}(net, Nothing, (Nothing, Nothing));

  function predict_neural_ode(θ, x, t)
    tspan = (t[1], t[end]);
    _prob = remake(prob; u0=x, p=θ, tspan=tspan);
    device(solve(_prob, AutoTsit5(Rosenbrock23()), u0=x, p=θ, saveat=t));
  end

  function loss(θ, x, y, t) # ,x ,y, t
    u_pred = predict_neural_ode(θ, x, t[1]);
    ŷ = Reg.gaussian_augment(u_pred, noise);
    l = Flux.mse(ŷ, permutedims(y, (1, 3, 2)))
    return l;
  end

  function val_loss(θ, x, y, t)
    u_pred = predict_neural_ode(θ, x, t[1]);
    ŷ = u_pred;
    l = Flux.mse(ŷ, permutedims(y, (1, 3, 2)))
    return l;
  end

  function cb(θ, l)
    lval = 0;

    for (x, y, t) in val_loader
      (x, y, t) = (x, y, t) |> device;
      lval += val_loss(θ, x, y, t);
    end

    lval /= (val_loader.nobs / val_loader.batchsize);
    @show(l)
    @show(lval);

    return false
  end

  @info("Initiate training")

  @info("ADAMW")
  opt = OptimizationOptimisers.ADAMW(lr, (0.9, 0.999), reg)
  optf = OptimizationFunction((θ, p, x, y, t) -> loss(θ, x, y, t), Optimization.AutoZygote())
  optprob = Optimization.OptimizationProblem(optf, p)
  result_neuralode = Optimization.solve(optprob, opt, ncycle(train_loader, epochs), callback=cb)

  @info("LBFGS")
  optprob2 = remake(optprob, u0 = result_neuralode.u)
  θ = Optimization.solve(optprob2, Optim.BFGS(initial_stepnorm=0.01),
                                          ncycle(train_loader, 100),
                                          callback=cb,
                                          allow_f_increases = false)

  return re(θ), p, ltrain, lval
end

function main()
  x_n = 64; # Discretization
  epochs = 100; # Iterations
  ratio = 0.7; # train/val ratio
  lr = 0.03; # learning rate
  r = 1e-6; # weigh decay (L2 reg)
  n = 0.01; # noise
  b = 32;

  data = Generator.read_dataset("./dataset/burgers_high_dim_training_set.jld2")["training_set"];
  model = Models.FeedForwardNetwork(x_n, 3, 64);
  K, p, ltrain, lval = training(model, epochs, data, b, ratio, lr, n, r, true);

  # BurgersAnalysis.check_result(K, p, 2)

  return K, p
end

end
