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
include("../../rom/pod.jl")

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

function f_pod(u, p, t)
  k, ν, Φ  = p[1], p[2], p[3];
  u_pod = Φ * Φ' * f(u, (k, ν), t)
end

# =======

using FFTW
using AbstractFFTs
using OrdinaryDiffEq
using DiffEqSensitivity
using Statistics
using Zygote
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
noise = 0.; # noise
batch_size = 3;
sol = Tsit5();
cuda = false;
ν = 0.04;

opt = OptimizationOptimisers.ADAMW(lr, (0.9, 0.999), reg);
dataset = Generator.read_dataset("./dataset/inviscid_burgers_advecting_shock_nu0001_t2_4_j173.jld2")["training_set"];
model = Models.CNN2(9, [2, 4, 8, 8, 4, 2, 1]);
# K, p, _ = BurgersCombinedCNN.training(model, epochs, dataset, opt, batch_size, ratio, noise, sol);
#@save "./models/inviscid_burgers_advecting_shock_t2_4_j173.bson" K p

# =======

# function training(model, epochs, dataset, opt, batch_size, ratio, noise=0., sol=Tsit5(), cuda=false)
   if cuda && CUDA.has_cuda()
      device = Flux.gpu
      CUDA.allowscalar(false)
      @info "Training on GPU"
  else
      device = Flux.cpu
      @info "Training on CPU"
  end

  # Monitoring
  global ep = 0;
  global count = 0;
  global lval = 0.;

  @info("Loading dataset")
  (train_loader, val_loader) = ProcessingTools.get_data_loader_cnn(dataset, batch_size, ratio, false, cuda);
  
  u_ref = [];
  for (x, y, t) in train_loader
    push!(u_ref, y);
    push!(u_ref, y);
  end
  print(size(reshape(Array(u_ref), size(u_ref)[1])))
  bas, _ = POD.generate_pod_svd_basis(reshape(Array(u_ref), size(y)[1], :), false);
  Φ = bas.modes;

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

    # xₙ = size(x)[1];
    # Δx = 1. / xₙ;
    # k = 2 * pi * AbstractFFTs.fftfreq(xₙ, 1. / Δx);

    # function f_NN_ϵ(u, p, t)
    #   du = f_pod(u, (k, ν, Φ), t);
    #   ϵ = net(u, p, t);
    #   du + ϵ
    # end

    # _prob = ODEProblem(net, x, extrema(t), θ); # (θ, k, ν, Φ)
    # ŷ = device(solve(_prob, sol, u0=x, p=θ, abstol=1e-9, reltol=1e-9, saveat=t));
    # return permutedims(del_dim(ŷ), (1, 3, 2)); # del_dim(ŷ)
  end

  #loss(p) = loss_derivative_fit(p, U, 1e-8)

  function loss(θ, x, y, t)
    x̂ = Reg.gaussian_augment(x, noise);
    ŷ = predict_neural_ode(θ, x̂, t[1]);
    l = Flux.mse(ŷ, y)
    return l;
  end

  function val_loss(θ, x, y, t)
    ŷ = predict_neural_ode(θ, x, t[1], bas.modes[:, 1:3]);
    l = Flux.mse(ŷ, y)
    return l;
  end


  function cb(θ, l)
    @show(l)
    global count += 1;

    iter = (train_loader.nobs / train_loader.batchsize);
    if (count % iter == 0)
      global ep += 1;
      global count = 0;
      global lval = 0;

      for (x, y, t) in val_loader
        (x, y, t) = (x, y, t) |> device;
        global lval += val_loss(θ, x, y, t);
      end

      global lval /= (val_loader.nobs / val_loader.batchsize);
      @info("Epoch ", ep, lval);
    end

    return false
  end

  @info("Initiate training")
  @info("ADAMW")
  optf = OptimizationFunction((θ, p, x, y, t) -> loss(θ, x, y, t), Optimization.AutoZygote());
  optprob = Optimization.OptimizationProblem(optf, p);
  result_neuralode = Optimization.solve(optprob, opt, ncycle(train_loader, epochs), callback=cb)

  # return re(result_neuralode.u), p, lval
# end

# end

K = re(result_neuralode.u)
#  return K, p
