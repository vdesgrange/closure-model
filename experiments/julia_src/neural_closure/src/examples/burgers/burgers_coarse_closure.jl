# module BurgersCombinedCNN

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
  ν = p[1]
  Δx = p[2]
  u₋ = circshift(u, 1)
  u₊ = circshift(u, -1)
  a₊ = u₊ + u
  a₋ = u + u₋
  du = @. -((a₊ < 0) * u₊^2 + (a₊ > 0) * u^2 - (a₋ < 0) * u^2 - (a₋ > 0) * u₋^2) / Δx + ν * (u₋ - 2u + u₊) / Δx^2
  du
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
  (train_loader, val_loader) = ProcessingTools.get_data_loader_cnn(dataset, batch_size, ratio, false, false);

  @info("Building model")
  p, re = Flux.destructure(model);
  f_nn = (u, p, t) -> re(p)(u);

  function f_closure(u, p, t)
    f(u, (ν, Δx), t) + f_nn(u, p, t)
  end

  function predict_neural_ode(θ, x, t)
    _prob = ODEProblem(f_closure, x, extrema(t), θ, saveat=t);
    ȳ = solve(_prob, sol, u0=x, p=θ,  abstol=1e-6, reltol=1e-6, sensealg=DiffEqSensitivity.InterpolatingAdjoint(; autojacvec=ZygoteVJP()));  # BacksolveAdjoint work
    ȳ = Array(ȳ);
    return permutedims(del_dim(ȳ), (1, 3, 2));
  end

  function loss_trajectory_fit(θ, x, y, t)
    x̂ = Reg.gaussian_augment(x, noise);
    ŷ = predict_neural_ode(θ, x̂, t[1]);
    l = Flux.mse(ŷ, y) + reg * sum(θ);
    return l;
  end

  function val_loss(θ, x, u, t) # y
    ŷ = predict_neural_ode(θ, x, t[1]);
    lt = Flux.mse(ŷ, u);

    y = reshape(u, size(u, 1), 1, :)
    dŷ = f_nn(y, θ, t)
    dy = f(y, (ν, Δx), t)
    lv = Flux.mse(dŷ, dy)
    return lv, lt;
  end

  function cb(θ, l)
    @show(l)
    count += 1;

    iter = (train_loader.nobs / train_loader.batchsize);
    if (count % iter == 0)
        ep += 1;
        count = 0;
        lval = 0;
        ltraj = 0;

        for (x, y, t) in val_loader
            a, b = val_loss(θ, x, y, t); 
            lval += a;
            ltraj += b;
        end

        lval /= (val_loader.nobs / val_loader.batchsize);
        ltraj /= (val_loader.nobs / val_loader.batchsize);

        @info("Epoch ", ep, lval, ltraj);

    end
    return false
  end


  @info("Initiate training")
  @info("ADAM Trajectory fit")
  optf = OptimizationFunction((θ, p, x, y, t) -> loss_trajectory_fit(θ, x, y, t), Optimization.AutoZygote());
  optprob = Optimization.OptimizationProblem(optf, p);
  result_neuralode = Optimization.solve(optprob, opt, ncycle(train_loader, epochs), callback=cb)

  return re(result_neuralode.u), result_neuralode.u, lval
end

# end
using BSON: @save
using Flux
using OrdinaryDiffEq: Tsit5
using OptimizationOptimisers

include("../../neural_ode/models.jl");
include("../../utils/generators.jl");

epochs = 2; # Iterations
ratio = 0.75; # train/val ratio
lr = 0.003; # learning rate
reg = 1e-8; # weigh decay (L2 reg)
noise = 0.05; # noise
batch_size = 8;
sol = Tsit5();
tₙ = 128;
xₙ = 64;
xₘₐₓ = pi;
Δx = xₘₐₓ / xₙ;
ν = 0.04;
snap_kwargs = (; ν, Δx, reg);

opt = OptimizationOptimisers.Adam(lr, (0.9, 0.999)); 
dataset = Generator.read_dataset("./dataset/viscous_burgers_high_dim_t2_64_xpi_64_nu0.04_typ2_m10_256_up16_j173.jld2")["training_set"];
model = Models.CNN2(9, [2, 4, 8, 8, 4, 2, 1]);
K, p, _ = training(model, epochs, dataset, opt, batch_size, ratio, noise, sol, snap_kwargs);