module BurgersCombinedCNN

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
include("./analysis.jl")

add_dim(x::Array{Float64, 1}) = reshape(x, (size(x)[1], 1, 1, 1))
add_dim(x::Array) = reshape(x, (size(x)[1], 1, 1, size(x)[2]))
del_dim(x::Array) = reshape(x, (size(x)[1], size(x)[4], size(x)[5]))

function training(model, epochs, dataset, opt, batch_size, ratio, noise=0., sol=Tsit5(), cuda=false)
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


  function predict_neural_ode(θ, x, t)
    tspan = (t[1], t[end]);
    _prob = ODEProblem(net, x, tspan, θ);
    device(solve(_prob, sol, u0=x, p=θ, abstol=1e-6, reltol=1e-6, saveat=t, sensealg=DiffEqSensitivity.BacksolveAdjoint(autojacvec=ZygoteVJP())));
  end

  function loss(θ, x, y, t)
    x̂ = Reg.gaussian_augment(add_dim(x), noise);
    ȳ = predict_neural_ode(θ, x̂, t[1]);
    ŷ = del_dim(ȳ);
    l = Flux.mse(ŷ, permutedims(y, (1, 3, 2)))
    return l;
  end


  function val_loss(θ, x, y, t)
    ȳ = predict_neural_ode(θ, add_dim(x), t[1]);
    ŷ = del_dim(ȳ);
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
  optf = OptimizationFunction((θ, p, x, y, t) -> loss(θ, x, y, t), Optimization.AutoZygote())
  optprob = Optimization.OptimizationProblem(optf, p)
  result_neuralode = Optimization.solve(optprob, opt, ncycle(train_loader, epochs), callback=cb)

  # @info("LBFGS")
  # optprob2 = remake(optprob, u0 = result_neuralode.u)
  # result_neuralode2 = Optimization.solve(optprob2, Optim.BFGS(initial_stepnorm=0.01),
  #                                         ncycle(train_loader, 100),
  #                                         callback=cb,
  #                                         allow_f_increases = false)

  return re(result_neuralode.u), p, ltrain, lval
end

function main()
  x_n = 64; # Discretization
  epochs = 1; # Iterations
  ratio = 0.75; # train/val ratio
  lr = 0.001; # learning rate
  reg = 1e-7; # weigh decay (L2 reg)
  n = 0.05; # noise
  batch = 16;

  # data = Generator.read_dataset("./dataset/burgers_high_dim_nu_variational_dataset.jld2")["training_set"];
  opt = OptimizationOptimisers.ADAMW(lr, (0.9, 0.999), reg);
  data = Generator.read_dataset("./dataset/viscous_burgers_high_dim_m10_256_j173.jld2")["training_set"];
  model = Models.CNN2(9, [2, 4, 8, 8, 4, 2, 1]);
  K, p, _, _ = training(model, epochs, data, opt, batch, ratio, n, Tsit5());

  return K, p
end

end
