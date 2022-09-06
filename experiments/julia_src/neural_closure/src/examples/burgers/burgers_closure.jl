module BurgersCombinedCNN

using CUDA
using BSON: @save
using Flux
using OrdinaryDiffEq
using DiffEqFlux
using Optimization
using IterTools: ncycle

include("../../utils/processing_tools.jl")
include("../../neural_ode/regularization.jl")

del_dim(x::Array) = reshape(x, (size(x)[1], size(x)[3], size(x)[4]))



function training(model, epochs, dataset, opt, batch_size, ratio, noise=0., sol=Tsit5(), cuda=false)
   if cuda && CUDA.has_cuda()
      device = Flux.gpu
      CUDA.allowscalar(false)
      @info "Training on GPU"
  else
      device = Flux.cpu
      @info "Training on CPU"
  end

  # Monitoring
  ep = 0;
  count = 0;
  lval = 0.;

  @info("Loading dataset")
  (train_loader, val_loader) = ProcessingTools.get_data_loader_cnn(dataset, batch_size, ratio, false, cuda);

  @info("Building model")
  model_gpu = model |> device;
  p, re = Flux.destructure(model_gpu);
  p = p |> device;
  net(u, p, t) = re(p)(u);

  function predict_neural_ode(θ, x, t)
    tspan = (float(t[1]), float(t[end]));
    _prob = ODEProblem(net, x, tspan, θ);
    ȳ = device(solve(_prob, sol, u0=x, p=θ, abstol=1e-6, reltol=1e-6, saveat=t, sensealg=DiffEqSensitivity.BacksolveAdjoint(autojacvec=ZygoteVJP())));
    return permutedims(del_dim(ȳ), (1, 3, 2));
  end

  function loss(θ, x, y, t)
    x̂ = Reg.gaussian_augment(x, noise);
    ŷ = predict_neural_ode(θ, x̂, t[1]);
    l = Flux.mse(ŷ, y)
    return l;
  end

  function val_loss(θ, x, y, t)
    ŷ = predict_neural_ode(θ, x, t[1]);
    l = Flux.mse(ŷ, y)
    return l;
  end

  function cb(θ, l)
    @show(l)
    count += 1;

    iter = (train_loader.nobs / train_loader.batchsize);
    if (count % iter == 0)
      ep += 1;
      count = 0;
      lval = 0;

      for (x, y, t) in val_loader
        (x, y, t) = (x, y, t) |> device;
        lval += val_loss(θ, x, y, t);
      end

      lval /= (val_loader.nobs / val_loader.batchsize);
      @info("Epoch ", ep, lval);
    end

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

  return re(result_neuralode.u), p, lval
end

end
