# module BurgersClosure

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

function f(u, p, t)
  k = p[1]
  ν = p[2]
  # u[1] = 0.
  # u[end] = 0.

  û = FFTW.fft(u)
  ûₓ = 1im .* k .* û
  ûₓₓ = (-k.^2) .* û

  uₓ = FFTW.ifft(ûₓ)
  uₓₓ = FFTW.ifft(ûₓₓ)
  uₜ = -u .* uₓ + ν .* uₓₓ
  return real.(uₜ)
end

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
  (train_loader, val_loader) = ProcessingTools.get_data_loader(dataset, batch_size, ratio, false, cuda);

  @info("Building model")
  model_gpu = model |> device;
  p, re = Flux.destructure(model_gpu);
  p = p |> device;

  net(u, p, t) = re(p)(u);
  f_pod(u, p, t) = Φ * Φ' * f(u, p, t)

  function f_NN_ϵ(u, p, t)
    θ, k, ν, Φ = p[1], p[2], p[3], p[4];
    # n = size(u)[1];
    # bas, _ = POD.generate_pod_svd_basis(reshape(Array(u), n, :), false);

    du = f_pod(u, (k, ν, Φ), t);
    ϵ = net(u, θ, t);
    du + ϵ
  end

  function f_NN(u, p, t)
    θ = p[1];
    net(u, θ, t);
  end

  function predict_neural_ode(θ, x, t, Φ)
    tspan = (float(t[1]), float(t[end]));
    xₙ = size(x);
    Δx = 1. / xₙ;
    k = 2 * pi * AbstractFFTs.fftfreq(xₙ, 1. / Δx);

    _prob = ODEProblem(f_NN, x, tspan, (θ, k, ν, Φ));
    ŷ = device(solve(_prob, sol, u0=x, p=θ, abstol=1e-9, reltol=1e-9, saveat=t));
    return permutedims(del_dim(ŷ), (1, 3, 2)); # del_dim(ŷ)
  end

  #loss(p) = loss_derivative_fit(p, U, 1e-8)

  function loss(θ, x, y, t)
    bas, _ = POD.generate_pod_svd_basis(reshape(Array(y), size(y)[1], :), false);
    x̂ = Reg.gaussian_augment(x, noise);
    ŷ = predict_neural_ode(θ, x̂, t[1], bas.modes[:, 1:3]);
    l = Flux.mse(ȳ .+ ŷ, y)
    return l;
  end

  function val_loss(θ, x, y, t)
    bas, _ = POD.generate_pod_svd_basis(reshape(Array(y), size(y)[1], :), false);
    ŷ = predict_neural_ode(θ, x, t[1], bas.modes[:, 1:3]);
    l = Flux.mae(ȳ .+ ŷ, y)
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
  @info("ADAMW")  # 70
  optf = OptimizationFunction((θ, p, x, y, t) -> loss(θ, x, y, t), Optimization.AutoZygote())
  optprob = Optimization.OptimizationProblem(optf, p)
  result_neuralode = Optimization.solve(optprob, opt, ncycle(train_loader, epochs), callback=cb)

  return re(result_neuralode.u), p, lval
end

# end


using OrdinaryDiffEq: Tsit5
using OptimizationOptimisers
include("../../neural_ode/models.jl");
include("../../utils/generators.jl");
include("./burgers_cnn.jl");
include("./burgers_combined_optimizer.jl");

x_n = 64; # Discretization
epochs = 100; # Iterations
ratio = 0.75; # train/val ratio
lr = 0.003; # learning rate
reg = 1e-7; # weigh decay (L2 reg)
n = 0.; # noise
batch = 3;

opt = OptimizationOptimisers.ADAMW(lr, (0.9, 0.999), reg);
data = Generator.read_dataset("./dataset/inviscid_burgers_advecting_shock_nu0001_t2_4_j173.jld2")["training_set"];
model = Models.CNN2(9, [2, 4, 8, 8, 4, 2, 1]);
K, p, _ = BurgersCombinedCNN.training(model, epochs, data, opt, batch, ratio, n, Tsit5());
#@save "./models/inviscid_burgers_advecting_shock_t2_4_j173.bson" K p

return K, p