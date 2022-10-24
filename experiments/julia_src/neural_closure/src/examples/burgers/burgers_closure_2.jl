module BurgersClosure
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

include("../../neural_ode/models.jl")
include("../../utils/processing_tools.jl")
include("../../neural_ode/regularization.jl")
include("../../rom/pod.jl")
include("../../utils/generators.jl")
include("../..//utils/graphic_tools.jl")

# =======

del_dim(x::Array{Float64, 4}) = reshape(x, (size(x)[1], size(x)[3], size(x)[4]))
del_dim(x::Array{Float64, 3}) = x

function f2(u, p, t)
  ν = p[1]
  Δx = pi / size(u, 1) # TO UPDATE
  u₋ = circshift(u, 1)
  u₊ = circshift(u, -1)
  a₊ = u₊ + u
  a₋ = u + u₋
  du = @. -((a₊ < 0) * u₊^2 + (a₊ > 0) * u^2 - (a₋ < 0) * u^2 - (a₋ > 0) * u₋^2) / Δx + ν * (u₋ - 2u + u₊) / Δx^2
  du
end

function training(model, epochs, dataset, opt, batch_size, ratio, noise=0., sol=Tsit5(), kwargs=(;))
    ep = 0;
    count = 0;
    lval = 0.;
    xₙ, ν = kwargs;

    @info("Loading dataset")
    (train_loader, val_loader) = ProcessingTools.get_data_loader_cnn(dataset, batch_size, ratio, false, false);
  
    tmp = [];
    for (x, y, t) in train_loader
        push!(tmp, y);
    end
    u_ref = cat(tmp...; dims=3);
    bas, _ = POD.generate_pod_svd_basis(reshape(Array(u_ref), xₙ, :), false);
    Φ = bas.modes[:, 1:20];

    @info("Building model")
    p, re = Flux.destructure(model);

    f_ref =  (u, p, t) -> begin
        y = f2(u, (ν), t)
        y = reshape(y, size(y, 1), size(y, 3))
        y = Φ * (Φ' * y)
        y = reshape(y, size(y, 1), 1, size(y, 2))
        y
    end

    f_nn =  (u, p, t) -> re(p)(u)

    function f_closure(u, p, t)
        f_ref(u, (ν), t) + f_nn(u, p, t)
    end

    function predict_neural_ode(θ, x, t)
        _prob = ODEProblem(f_closure, x, extrema(t), θ, saveat=t);
        ȳ = solve(_prob, sol, u0=x, p=θ, abstol=1e-7, reltol=1e-7, sensealg=DiffEqSensitivity.BacksolveAdjoint(; autojacvec=ZygoteVJP()));
        ȳ = Array(ȳ);
        return permutedims(del_dim(ȳ), (1, 3, 2));
    end

    function loss_derivative_fit(θ, x, u, t)
        sum(eachslice(u; dims = 2)) do y
            y = reshape(y, size(y, 1), 1, :)
            dŷ = f_nn(y, θ, t)
            dy = f2(y, (ν), t)
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
        @show(l)
        count += 1;

        iter = (train_loader.nobs / train_loader.batchsize);
        if (count % iter == 0)
            ep += 1;
            count = 0;
            lval = 0;

            for (x, y, t) in val_loader
                lval += val_loss(θ, x, y, t); 
            end

            lval /= (val_loader.nobs / val_loader.batchsize);
            @info("Epoch ", ep, lval);
        end
        return false
    end

    @info("Initiate training")
    @info("ADAMW")
    optf = OptimizationFunction((θ, p, x, y, t) -> loss_trajectory_fit(θ, x, y, t), Optimization.AutoZygote());
    optprob = Optimization.OptimizationProblem(optf, p);
    result_neuralode = Optimization.solve(optprob, opt, ncycle(train_loader, epochs), callback=cb)

    return re(result_neuralode.u), result_neuralode.u, Φ, lval
end

end

# epochs = 2; # Iterations
# ratio = 0.75; # train/val ratio
# lr = 0.003; # learning rate
# reg = 1e-7; # weigh decay (L2 reg)
# noise = 0.; # noise
# batch_size = 16;
# noise = 0.;
# sol = Tsit5();

# ν = 0.;
# t_max = 2.;
# t_min = 0.;
# x_max = pi;
# x_min = 0.;
# tₙ = 64;
# xₙ = 64;
# t =  LinRange(t_min, t_max, tₙ);
# x =  LinRange(x_min, x_max, xₙ);
# snap_kwargs = (; xₙ, ν);
# init_kwargs = (; m=10);

# opt = OptimizationOptimisers.ADAMW(lr, (0.9, 0.999), reg);
# dataset = Generator.read_dataset("./dataset/inviscid_burgers_high_dim_m10_256_j173.jld2")["training_set"];
# model = Models.CNN2(9, [2, 4, 8, 8, 4, 2, 1]);
# K, p, Φ, _ = training(model, epochs, dataset, opt, batch_size, ratio, noise, Tsit5(), snap_kwargs);
# θ = Array(p);


# # ===
# t, u = dataset[1];
# u₀ = reshape(u[:, 1], :, 1, 1);

# t, u₀, u = Generator.get_burgers_batch(t_max, t_min, x_max, x_min, tₙ, xₙ, ν, 2, init_kwargs);
# u₀ = reshape(u[:, 1], :, 1, 1);

# (train_loader, val_loader) = ProcessingTools.get_data_loader_cnn(dataset, batch_size, ratio, false, false);
# tmp = [];
# for (x, y, t) in train_loader
#     push!(tmp, y);
# end
# u_ref = cat(tmp...; dims=3);
# bas, _ = POD.generate_pod_svd_basis(reshape(Array(u_ref), xₙ, :), false);
# Φ = bas.modes[:, 1:20];

# f_ref =  (u, p, t) -> begin
#     y = f2(u, (ν), t)
#     y = reshape(y, size(y, 1), size(y, 3))
#     y = Φ * (Φ' * y)
#     y = reshape(y, size(y, 1), 1, size(y, 2))
#     y
# end
# f_nn =  (u, p, t) -> K(u);
# f_closure = (u, p, t) -> f_ref(u, (ν), t) + f_nn(u, p, t);

# # ===

# prob_pod = ODEProblem(f_ref, u₀, extrema(t), θ, saveat=t);
# U_pod = Array(solve(prob_pod, Tsit5()));

# prob_closure = ODEProblem(f_closure, u₀, extrema(t), θ, saveat=t);
# U_closure = Array(solve(prob_closure, Tsit5()))

# GraphicTools.show_state(u, t, x, "", "t", "x")
# GraphicTools.show_state(U_pod[:, 1, 1, :], t, x, "", "t", "x")
# GraphicTools.show_state(U_closure[:, 1, 1, :], t, x, "", "t", "x")
