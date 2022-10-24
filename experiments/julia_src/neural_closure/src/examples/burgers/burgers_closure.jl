# module BurgersClosure
using FFTW
using AbstractFFTs
using Statistics
using Zygote
using Flux
using DiffEqFlux
using OrdinaryDiffEq
using DiffEqSensitivity
using Optimization
using OptimizationOptimisers
using IterTools: ncycle
using CUDA
using BSON: @save
using Printf
using Plots

include("../../neural_ode/models.jl")
include("../../utils/processing_tools.jl")
include("../../neural_ode/regularization.jl")
include("../../rom/pod.jl")
include("../../utils/generators.jl")
include("../..//utils/graphic_tools.jl")

x_n = 64; # Discretization
epochs = 100; # Iterations
ratio = 0.75; # train/val ratio
lr = 0.003; # learning rate
reg = 1e-7; # weigh decay (L2 reg)
noise = 0.; # noise
batch_size = 16;
sol = Tsit5();
cuda = false;

ν = 0.;
t_max = 2.;
t_min = 0.;
x_max = pi;
x_min = 0.;
t_n = 64;
x_n = 64;
t =  LinRange(t_min, t_max, t_n);
x =  LinRange(x_min, x_max, x_n);


opt = OptimizationOptimisers.ADAMW(lr, (0.9, 0.999), reg);
dataset = Generator.read_dataset("./dataset/inviscid_burgers_high_dim_m10_256_j173.jld2")["training_set"];
model = Models.CNN2(9, [2, 4, 8, 8, 4, 2, 1]);
model = Models.FeedForwardNetwork(x_n, 3, 48);

# =======

del_dim(x::Array{Float64, 4}) = reshape(x, (size(x)[1], size(x)[3], size(x)[4]))
del_dim(x::Array{Float64, 3}) = x

function f(u, p, t=nothing)
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

function f2(u, p, t)
  k = p[1]
  ν = p[2]
  Δx = pi / size(u, 1)
  u₋ = circshift(u, 1)
  u₊ = circshift(u, -1)
  a₊ = u₊ + u
  a₋ = u + u₋
  du = @. -((a₊ < 0) * u₊^2 + (a₊ > 0) * u^2 - (a₋ < 0) * u^2 - (a₋ > 0) * u₋^2) / Δx + ν * (u₋ - 2u + u₊) / Δx^2
  du
end


function riemann(u, xt)
  S = (u[2:end] .+ u[1:end-1]) ./ 2.;
  a = (u[2:end] .>= u[1:end-1]) .* (((S .> xt) .* u[1:end-1]) .+ ((S .<= xt) .* u[2:end]));
  b = (u[2:end] .< u[1:end-1]) .* (
      ((xt .<= u[1:end-1]) .* u[1:end-1]) .+
      (((xt .> u[1:end-1]) .& (xt .< u[2:end])) .* xt) +
      ((xt .>= u[2:end]) .* u[2:end])
      );
  return a .+ b;
end

function νm_flux(u, xt=0.)
  r = riemann(u, xt);
  return r.^2 ./ 2.;
end

function f3(u, p, t)
  ū = zeros(size(u)[1] + 2);
  ū[2:end-1] = deepcopy(u);
  nf_u = νm_flux(ū, 0.);
  uₜ = - (nf_u[2:end] - nf_u[1:end-1]) ./ Δx
  return uₜ
end

snap_kwargs = repeat([(; t_max, t_min, x_max, x_min, t_n, x_n, nu=ν, typ=2)], 16);
init_kwargs = repeat([(;  m=10 )], 16);
dataset = Generator.generate_closure_dataset(16, 1, "", snap_kwargs, init_kwargs);
GraphicTools.show_state(dataset[1][2], t, x, "", "t", "x")


# function training(model, epochs, dataset, opt, batch_size, ratio, noise=0., sol=Tsit5(), cuda=false)
  global ep = 0;
  global count = 0;
  global lval = 0.;

  @info("Loading dataset")
  (train_loader, val_loader) = ProcessingTools.get_data_loader_cnn(dataset, batch_size, ratio, false, cuda);
  
  tmp = [];
  for (x, y, t) in train_loader
    push!(tmp, y);
  end

  u_ref = cat(tmp...; dims=3);
  xₙ = size(u_ref)[1];
  Δx = 1. / xₙ;
  k = 2 * pi * AbstractFFTs.fftfreq(xₙ, 1. / Δx);
  bas, _ = POD.generate_pod_svd_basis(reshape(Array(u_ref), xₙ, :), false);
  Φ = bas.modes[:, 1:5];

  @info("Building model")
  p, re = Flux.destructure(model);

  f_ref =  (u, p, t) -> begin
    y = f2(u, (k, ν), t)
    y = reshape(y, size(y, 1), size(y, 3))
    y = Φ * (Φ' * y)
    y = reshape(y, size(y, 1), 1, size(y, 2))
    y
  end

  f_nn =  (u, p, t) -> re(p)(u)

  function f_closure(u, p, t)
    f_ref(u, (k, ν), t) + f_nn(u, p, t)
  end

  
  function predict_neural_ode(θ, x, t)
    # _prob = SplitODEProblem(f_ref,  f_nn, x,  extrema(t),  θ, saveat=t);
    _prob = ODEProblem(
        f_nn,
        x,
        extrema(t), 
        θ,
        saveat=t
    );
    ȳ = solve(_prob, sol, u0=x, p=θ, abstol=1e-7, reltol=1e-7, sensealg=DiffEqSensitivity.BacksolveAdjoint(; autojacvec=ZygoteVJP()));
    ȳ = Array(ȳ);
    return permutedims(del_dim(ȳ), (1, 3, 2));
  end

  function loss_derivative_fit2(θ, u, λ)
    t = 0.0
    dudt_ref = f2(u, (k, ν), t)
    dudt_predict = f_nn(u, θ, t) #  f_pod(u, ν, t) +
    data = sum(abs2, dudt_predict - dudt_ref) / length(u)
    reg = sum(abs2, p) / length(p)
    data + λ * reg
  end


  function loss_derivative_fit(θ, x, y, t) # correct ! time step by time step
    sum(eachslice(y; dims = 2)) do y
      y = reshape(y, size(y, 1), 1, :)
      dŷ = f_closure(y, θ, t)
      dy = f2(y, (k, ν), t)
      l = Flux.mse(dŷ, dy)
      l
    end
  end
  
  function loss_trajectory_fit(θ, x, y, t)
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
    # @show(l)
    global count += 1;

    iter = (train_loader.nobs / train_loader.batchsize);
    if (count % iter == 0)
      global ep += 1;
      global count = 0;
      global lval = 0;

      for (x, y, t) in val_loader
        # (x, y, t) = (x, y, t) |> device;
        global lval += val_loss(θ, x, y, t);
      end

      global lval /= (val_loader.nobs / val_loader.batchsize);
      # @info("Epoch ", ep, lval);
    else
      lval = "                   "
    end
    println("Epoch $ep, \t count $count, \t lval $lval, \t l $l")

    return false
  end

  ep = 0;
  count = 0;
  lval = 0;
  @info("Initiate training")
  @info("ADAMW")
  optf = OptimizationFunction((θ, p, x, y, t) -> loss_derivative_fit(θ, x, y, t), Optimization.AutoZygote());
  optprob = Optimization.OptimizationProblem(optf, p);
  result_neuralode = Optimization.solve(optprob, opt, ncycle(train_loader, epochs), callback=cb)

  # return re(result_neuralode.u), result_neuralode.u, lval
# end

prob = ODEProblem(f_ref, u₀, extrema(t), θ, saveat=t);
U = Array(solve(prob, Tsit5()))
GraphicTools.show_state(U[:, 1, 1, :], t, x, "", "t", "x")


t, u = dataset[1];
ux = LinRange(0., pi, xₙ)
u₀ = reshape(u[:, 1], :, 1, 1);
θ = result_neuralode.u
# prob = SplitODEProblem(f_ref, f_nn, u₀, extrema(t), θ, saveat=t);
prob = ODEProblem(f_closure, u₀, extrema(t), θ, saveat=t);
U = Array(solve(prob, Tsit5()))
GraphicTools.show_state(U[:, 1, 1, :], t, x, "", "t", "x")
GraphicTools.show_state( u, t, x, "", "t", "x")

# end

add_dim(x::Array{Float64, 1}) = reshape(x, (size(x)[1], 1, 1))
t, u = dataset[1];
x = LinRange(0., pi, xₙ)
u₀ = reshape(u[:, 1], :, 1, 1);
θ = p
size(u₀)

# function check_result(nn, res, typ)
using BSON: @save, @load
@load "./models/inviscid_burgers_high_dim_m10_256_t2_j173.bson" K p
θ = Flux.params(K)

  add_dim(x::Array{Float64, 1}) = reshape(x, (size(x)[1], 1, 1))
  t, u₀, u = Generator.get_burgers_batch(2., 0., pi, 0., 64, 64, 0., 2, (; m=10));
  # prob = SplitODEProblem(f_ref, f_nn, add_dim(u₀), extrema(t), θ, saveat=t);
  # U = Array(solve(prob, Tsit5()))
  prob = DiffEqFlux.NeuralODE(K,  extrema(t), Tsit5(), saveat=t);
  U = Array(prob(add_dim(u₀)))
  GraphicTools.show_state(u, t, x, "", "t", "x")
  GraphicTools.show_state(U[:, 1, 1, :], t, x, "", "t", "x")

  for (i, t) ∈ enumerate(t)
    pl = plot(; xlabel = "x", title = @sprintf("t = %.3f", t), ylims = extrema(u[:, :]))
    plot!(pl, x, u[:, i]; label = "Full")
    plot!(pl, x, U[:, 1, 1, i]; label = "NN")
    # plot!(pl, x, U[:, 1, 1, i]; label = "POD + NN")
    plot!(pl, x, Φ * (Φ' * u[:, i]); label = "POD")
    display(pl)
end
# end

# prob_neuralode = DiffEqFlux.NeuralODE(K,  extrema(t), Tsit5(), saveat=t);
# u_pred = prob_neuralode(add_dim(u₀))
# GraphicTools.show_state(u, t, x, "", "t", "x")
# GraphicTools.show_state(hcat(u_pred.u...)[:, :], t, x, "", "t", "x")

#  return K, p
