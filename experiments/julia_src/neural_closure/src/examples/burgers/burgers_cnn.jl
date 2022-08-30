module BurgersCNN

using CUDA
using BSON: @save
using Flux
using OrdinaryDiffEq
using DiffEqSensitivity
using DiffEqFlux

include("../../utils/processing_tools.jl")
include("../../neural_ode/regularization.jl")

add_dim(x::Array{Float64, 1}) = reshape(x, (size(x)[1], 1, 1, 1))
add_dim(x::Array) = reshape(x, (size(x)[1], 1, 1, size(x)[2]))
del_dim(x::Array) = reshape(x, (size(x)[1], size(x)[4], size(x)[5]))

function training(model, epochs, dataset, opt, batch_size, ratio, noise=0., sol=Tsit5(), cuda=false)
   if cuda && CUDA.has_cuda()
      device = Flux.gpu;
      CUDA.allowscalar(true); # false
      @info "Training on GPU"
  else
      device = Flux.cpu;
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
    Array(solve(_prob, sol, u0=x, p=p, abstol=1e-9, reltol=1e-9, saveat=t, sensealg=DiffEqSensitivity.BacksolveAdjoint(autodiff=true)));
  end

  function loss(x, y, t)
    x̂ = Reg.gaussian_augment(add_dim(x), noise);
    ȳ = predict_neural_ode(x̂, t[1]);
    ŷ = del_dim(ȳ);  # ŷ = Reg.gaussian_augment(del_dim(ȳ), noise);
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
    ȳ = predict_neural_ode(add_dim(x), t[1]);
    ŷ = del_dim(ȳ);
    l = Flux.mse(ŷ, permutedims(y, (1, 3, 2)))
    return l;
  end

  function evalcb()
    lval = 0;
    for (x, y, t) in val_loader
      lval += val_loss(x, y, t);
    end
    lval /= (val_loader.nobs / val_loader.batchsize);
    @show(lval);
  end

  @info("Train")
  trigger = Flux.plateau(() -> ltrain, 10; init_score = 1, min_dist = 1f-4);
  Flux.@epochs epochs begin
    Flux.train!(loss, Flux.params(p), train_loader, opt, cb = [traincb, evalcb]);
    trigger() && break;
  end

  return re(p), p, ltrain, lval
end

end
