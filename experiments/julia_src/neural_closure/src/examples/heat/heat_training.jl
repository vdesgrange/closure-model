# module HeatTraining

using Zygote
using Flux
using DiffEqFlux
using OrdinaryDiffEq
using DiffEqSensitivity
using Optimization
using OptimizationOptimisers
using IterTools: ncycle
using BSON: @save, @load


include("../../neural_ode/objectives.jl")
include("../../neural_ode/models.jl")
include("../../utils/processing_tools.jl")
include("../../neural_ode/regularization.jl")
include("../../utils/generators.jl")


del_dim(x::Array{Float64, 4}) = reshape(x, (size(x)[1], size(x)[3], size(x)[4]))
del_dim(x::Array{Float64, 3}) = x


function f(u, p, t)
    Δx = p[1];
    κ = p[2];
    u₋ = circshift(u, 1);
    u₊ = circshift(u, -1);
    # u₋[1] = 0; # -1
    # u₊[end] = 0; # 1
    uₜ = κ / (Δx^2) .* (u₊ .- 2 .* u .+ u₋);
    return uₜ
end

function training(model, epochs, dataset, opt, batch_size, ratio, noise=0., sol=Tsit5(), kwargs=(;))
    ep = 0;
    count = 0;
    lval = 0.;
    (Δx, κ, reg) = kwargs;

    @info("Loading dataset")
    (train_loader, val_loader) = ProcessingTools.get_data_loader_cnn(dataset, batch_size, ratio, false, false);

    @info("Building model")
    p, re = Flux.destructure(model);
    f_nn =  (u, p, t) -> re(p)(u)

    function predict_neural_ode(θ, x, t)
        _prob = ODEProblem(f_nn, x, extrema(t), θ, saveat=t);
        ȳ = solve(_prob, sol, u0=x, p=θ, sensealg=DiffEqSensitivity.InterpolatingAdjoint(; autojacvec=ZygoteVJP()));  # BacksolveAdjoint work
        ȳ = Array(ȳ);
        return permutedims(del_dim(ȳ), (1, 3, 2));
    end

    function loss_derivative_fit(θ, x, u, t)
        y = reshape(u, size(u, 1), 1, :);
        dŷ = f_nn(y, θ, t);
        dy = f(y, (Δx, κ), t);
        l = Flux.mse(dŷ, dy) + reg * sum(θ);
    end

    function loss_trajectory_fit(θ, x, y, t)
        x̂ = Reg.gaussian_augment(x, noise);
        ŷ = predict_neural_ode(θ, x̂, t[1]);
        l = Flux.mse(ŷ, y)
        return l;
    end

    function val_loss(θ, x, u, t)
        ŷ = predict_neural_ode(θ, x, t[1]);
        lt = Flux.mse(ŷ, u)

        y = reshape(u, size(u, 1), 1, :)
        dŷ = f_nn(y, θ, t)
        dy = f(y, (Δx, κ), t)
        lv = Flux.mse(dŷ, dy)
        return lv, lt;
    end

    function cb(θ, l)
        # @show(l)
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

    # @info("ADAM Trajectory fit")
    # optf2 = OptimizationFunction((θ, p, x, y, t) -> loss_derivative_fit(θ, x, y, t), Optimization.AutoZygote());
    # optprob2 = remake(optprob, f=optf2, u0 = result_neuralode.u)
    # result_neuralode2 = Optimization.solve(optprob2, opt,
    #                                         ncycle(train_loader, epochs),
    #                                         callback=cb,
    #                                         allow_f_increases = false)

    # return re(result_neuralode2.u), result_neuralode2.u, lval
end

using BSON: @save
using Flux
using OrdinaryDiffEq: Tsit5
using OptimizationOptimisers
using Plots

include("../../neural_ode/models.jl");
include("../../utils/generators.jl");
include("../../utils/graphic_tools.jl")

epochs = 500; # Iterations
ratio = 0.75; # train/val ratio
lr = 0.001; # learning rate
reg = 1e-7; # weigh decay (L2 reg)
noise = 0.05; # noise
batch_size = 16;
sol = Tsit5();
tₙ = 64;
xₙ = 64;
xₘₐₓ = 1.;
Δx = xₘₐₓ / xₙ;
κ = 0.01;
snap_kwargs = (; Δx, κ, reg);

opt = OptimizationOptimisers.Adam(lr, (0.9, 0.999)); 
dataset = Generator.read_dataset("./dataset/diffusion_n256_k0.01_N15_analytical_t1_64_x1_64_up16.jld2")["training_set"];
cleaned_dataset = [];
for data in dataset
    push!(cleaned_dataset, [data[1], data[2], data[4], data[5]]);
end

model = Models.LinearModel(xₙ);
K, p, _ = training(model, epochs, cleaned_dataset, opt, batch_size, ratio, noise, sol, snap_kwargs);
 
# @save "./models/diffusion_k0.01_N15_analytical_t1_64_x1_64_ep500_traj.bson" K p

