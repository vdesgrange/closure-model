module BurgersDirect

using CUDA
using BSON: @save
using Flux
using OrdinaryDiffEq
using SciMLSensitivity

include("../../utils/processing_tools.jl");
include("../../neural_ode/regularization.jl");

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

  prob = ODEProblem{false}(net, Nothing, (Nothing, Nothing));

  function predict_neural_ode(x, t)
    tspan = (t[1], t[end]);
    _prob = remake(prob; u0=x, p=p, tspan=tspan);
    device(solve(_prob, sol, u0=x, p=p, saveat=t, abstol=1e-9, reltol=1e-9, sensealg=BacksolveAdjoint(autodiff=true)));
  end

  function loss(x, y, t)
    x̂ = Reg.gaussian_augment(x, noise);
    ŷ = predict_neural_ode(x̂, t[1]);
    l = Flux.mse(ŷ, permutedims(y, (1, 3, 2)))
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

  function val_loss(x, y, t)
    ŷ = predict_neural_ode(x, t[1]);
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

  @info("Initiate training")
  trigger = Flux.plateau(() -> ltrain, 10; init_score = 1, min_dist = 1f-4);
  Flux.@epochs epochs begin
    Flux.train!(loss, Flux.params(p), train_loader, opt, cb = [traincb, evalcb]);
    trigger() && break;
  end

  return re(p), p, ltrain, lval
end

end
