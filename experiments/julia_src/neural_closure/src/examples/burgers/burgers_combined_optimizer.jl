# module BurgersCombinedCNN
using BSON: @save
using Flux
using OrdinaryDiffEq
using DiffEqFlux
using Optimization
using IterTools: ncycle
using LinearAlgebra

include("../../utils/processing_tools.jl")
include("../../neural_ode/regularization.jl")

del_dim(x::Array) = reshape(x, (size(x)[1], size(x)[3], size(x)[4]))

function f(u, (ν, Δx), t)
  u₋ = circshift(u, 1)
  u₊ = circshift(u, -1)
  a₊ = u₊ + u
  a₋ = u + u₋
  du = @. -((a₊ < 0) * u₊^2 + (a₊ > 0) * u^2 - (a₋ < 0) * u^2 - (a₋ > 0) * u₋^2) / Δx + ν * (u₋ - 2u + u₊) / Δx^2
  du
end

function riemann(u, xt)
  u₋ = circshift(u, 1)
  u₊ = circshift(u, -1)

  S = (u₊ .+ u₋) ./ 2.;
  a = (u₊ .>= u₋) .* (((S .> xt) .* u₋) .+ ((S .<= xt) .* u₊));
  b = (u₊ .< u₋) .* (
      ((xt .<= u₋) .* u₋) .+
      (((xt .> u₋) .& (xt .< u₊)) .* xt) +
      ((xt .>= u₊) .* u₊)
      );
  return a .+ b;
end

function νm_flux(u, xt=0.)
  r = riemann(u, xt);
  return r.^2 ./ 2.;
end

function f_godunov(u, (ν, Δx), t)
  ū = deepcopy(u);
  nf_u = νm_flux(ū, 0.);
  nf_u₋ = circshift(nf_u, 1);
  nf_u₊ = circshift(nf_u, -1);

  uₜ = - (nf_u₊ - nf_u₋) ./ Δx
  return uₜ
end

function training(model, epochs, dataset, opt, batch_size, ratio, noise=0., sol=Tsit5(), kwargs=(;))
  ν = kwargs.ν;
  Δx = kwargs.Δx;
  reg = kwargs.reg;

  # Monitoring
  ep = 0;
  count = 0;
  lval = 0.;

  @info("Loading dataset")
  (train_loader, val_loader) = ProcessingTools.get_data_loader(dataset, batch_size, ratio, false);

  @info("Building model")
  p, re = Flux.destructure(model);
  fᵣₒₘ =  (v, p, t) -> re(p)(v)

  function predict_neural_ode(θ, x, t)
    _prob = ODEProblem(fᵣₒₘ, x, extrema(t), θ, saveat=t);
    ȳ = Array(solve(_prob, sol, abstol=1e-6, reltol=1e-6, sensealg=DiffEqSensitivity.InterpolatingAdjoint(; autojacvec=ZygoteVJP())));
    return permutedims(ȳ, (1, 3, 2));
  end

  function loss_trajectory_fit(θ, x, y, t)
    x̂ = Reg.gaussian_augment(x, noise);
    ŷ = predict_neural_ode(θ, x̂, t[:, 1]);
    l = Flux.mse(ŷ, y) + reg * sum(θ);
    return l;
  end

  function val_loss(θ, x, u, t) # y
    ŷ = predict_neural_ode(θ, x, t[:, 1]);
    lt = Flux.mse(ŷ, u);

    lre = norm(ŷ - u) / norm(u);

    y = reshape(u, size(u, 1), :)
    dŷ = fᵣₒₘ(y, θ, t)
    dy = f_godunov(y, (ν, Δx), t[:, 1])
    lv = Flux.mse(dŷ, dy)
    return lv, lt, lre;
  end

  function cb(θ, l)
    @show(l)
    count += 1;
    epochs = Int[]
    errors = zeros(0)
    mses = zeros(0);

    iter = (train_loader.nobs / train_loader.batchsize);
    if (count % iter == 0)
        ep += 1;
        count = 0;
        lval = 0;
        ltraj = 0;
        lrel = 0;

        for (x, y, t) in val_loader
            a, b, c = val_loss(θ, x, y, t[:, 1]); 
            lval += a;
            ltraj += b;
            lrel += c;
        end

        lval /= (val_loader.nobs / val_loader.batchsize);
        ltraj /= (val_loader.nobs / val_loader.batchsize);
        lrel /= (val_loader.nobs / val_loader.batchsize);

        push!(epochs, ep);
        push!(mses, ltraj);
        push!(errors, lrel);

        # pl1 = Plots.plot(epochs, errors; xlabel = "Epochs", title = "Relative error", dpi=600, label="Relative error");
        Plots.plot(epochs, mses; xlabel = "Epochs", title = "Mean square error", dpi=600, label="MSE");
        # savefig(pl1, "old_model_inviscid_relative_error_per_epoch.png");

        @info("Epoch ", ep, lval, ltraj);

    end
    return false
  end


  @info("Initiate training")
  @info("ADAM Trajectory fit")
  optf = OptimizationFunction((θ, p, x, y, t) -> loss_trajectory_fit(θ, x, y, t), Optimization.AutoZygote());
  optprob = Optimization.OptimizationProblem(optf, p);
  result_neuralode = Optimization.solve(optprob, opt, ncycle(train_loader, epochs), callback=cb)


  # @info("LBFGS")
  # optprob2 = remake(optprob, u0 = result_neuralode.u)
  # result_neuralode2 = Optimization.solve(optprob2, Optim.BFGS(initial_stepnorm=0.01),
  #                                         ncycle(train_loader, 100),
  #                                         callback=cb,
  #                                         allow_f_increases = false)

  return re(result_neuralode.u), result_neuralode.u, lval
end

# end
using BSON: @save
using Flux
using OrdinaryDiffEq: Tsit5
using OptimizationOptimisers

include("../../neural_ode/models.jl");
include("../../utils/generators.jl");

epochs = 200; # Iterations
ratio = 0.75; # train/val ratio
lr = 0.001; # learning rate
reg = 1e-7; # weigh decay (L2 reg)
noise = 0.05; # noise
batch_size = 32;
sol = Tsit5();
tₙ = 64;
xₙ = 64;
xₘₐₓ = pi;
Δx = xₘₐₓ / (xₙ - 1);
ν = 0.;
snap_kwargs = (; ν, Δx, reg);

opt = OptimizationOptimisers.Adam(lr, (0.9, 0.999)); 
dataset = Generator.read_dataset("./dataset/inviscid/old_model_inviscid_burgers_fouriers_t2_64_xpi_64_nu0_typ5_K200_256_up8_j173.jld2")["training_set"];
model = Models.CNN2(9, [2, 4, 8, 8, 4, 2, 1]);
K, p, _ = training(model, epochs, dataset, opt, batch_size, ratio, noise, sol, snap_kwargs);
@save "./models/pure_node_inviscid/old_model_inviscid_burgers_fouriers_t2_64_xpi_64_nu0_typ5_K200_256_up8_j173.bson" K p
savefig("old_model_inviscid_mse_per_epoch.png");
