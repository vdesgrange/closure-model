using Zygote
using Flux
using DiffEqFlux
using OrdinaryDiffEq
using DiffEqSensitivity
using Optimization
using OptimizationOptimisers
using IterTools: ncycle
using BSON: @save

include("../../neural_ode/models.jl")
include("../../utils/processing_tools.jl")
include("../../neural_ode/regularization.jl")
include("../../utils/generators.jl")

# =======

del_dim(x::Array{Float64, 4}) = reshape(x, (size(x)[1], size(x)[3], size(x)[4]))
del_dim(x::Array{Float64, 3}) = x

function f(u, p, t)
    Δx = p[1];
    u₋₋ = circshift(u, 2);
    u₋ = circshift(u, 1);
    u₊ = circshift(u, -1);
    u₊₊ = circshift(u, -2);
  
    uₓ = (u₊ - u₋) ./ (2 * Δx);
    uₓₓₓ = (-u₋₋ .+ 2u₋ .- 2u₊ + u₊₊) ./ (2 * Δx^3);
    uₜ = -(6u .* uₓ) .- uₓₓₓ;
    return uₜ
end

function training(model, epochs, dataset, opt, batch_size, ratio, noise=0., sol=Tsit5(), kwargs=(;))
    ep = 0;
    count = 0;
    lval = 0.;
    Δx = kwargs.Δx;
    reg = kwargs.reg;

    @info("Loading dataset")
    (train_loader, val_loader) = ProcessingTools.get_data_loader_cnn(dataset, batch_size, ratio, false, false);

    @info("Building model")
    p, re = Flux.destructure(model);
    f_nn =  (u, p, t) -> re(p)(u)

    # function f_closure(u, p, t)
    #     f_ref(u, (Δx), t) + f_nn(u, p, t)
    # end

    function predict_neural_ode(θ, x, t)
        _prob = ODEProblem(f_nn, x, extrema(t), θ, saveat=t);
        ȳ = Array(solve(_prob, sol, sensealg=DiffEqSensitivity.InterpolatingAdjoint(; autojacvec=ZygoteVJP())));
        return permutedims(del_dim(ȳ), (1, 3, 2));
    end
    

    # function loss_derivative_fit(θ, x, u, t)
    #     mean(eachslice(u; dims = 2)) do y
    #         y = reshape(y, size(y, 1), 1, :)
    #         dŷ = f_nn(y, θ, t)
    #         dy = f(y, (Δx), t)
    #         l = Flux.mse(dŷ, dy) + reg * sum(θ)
    #         l
    #     end
    # end

    function loss_derivative_fit(θ, x, u, t)
        xₙ, _, _ = size(u);
        y = reshape(u, xₙ, 1, :)
        dŷ = f_nn(y, θ, t)
        dy = f(y, (Δx), t)
        l = Flux.mse(dŷ, dy) + reg * sum(θ)
    end

    function loss_trajectory_fit(θ, x, y, t)
        x̂ = Reg.gaussian_augment(x, noise);
        ŷ = predict_neural_ode(θ, x̂, t[1]);
        l = Flux.mse(ŷ, y)  + reg * sum(θ);
        return l;
    end

    function val_loss(θ, x, u, t) # y
        ŷ = predict_neural_ode(θ, x, t[1]);
        lt = Flux.mse(ŷ, u)

        y = reshape(u, size(u, 1), 1, :)
        dŷ = f_nn(y, θ, t)
        dy = f(y, (Δx), t)
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

    # @info("Initiate training")
    # @info("ADAM Trajectory fit")
    # optf2 = OptimizationFunction((θ, p, x, y, t) -> loss_trajectory_fit(θ, x, y, t), Optimization.AutoZygote());
    # optprob2 = remake(optprob, f=optf2, u0 = result_neuralode.u)
    # result_neuralode2 = Optimization.solve(optprob2, opt,
    #                                         ncycle(train_loader, epochs),
    #                                         callback=cb,
    #                                         allow_f_increases = false)

    return re(result_neuralode.u), result_neuralode.u, lval
end


epochs = 50; # Iterations
ratio = 0.75; # train/val ratio
lr = 0.003; # learning rate
reg = 1e-7; # weigh decay (L2 reg)
noise = 0.05; # noise
batch_size = 8;
sol = Tsit5();
tₘₐₓ = 5.;
tₘᵢₙ = 0.;
xₘₐₓ = 4pi;
xₘᵢₙ = 0.;
tₙ = 128;
xₙ = 64;
x = LinRange(0, xₘₐₓ, xₙ);
Δx = (xₘₐₓ - xₘᵢₙ) / (xₙ - 1);
snap_kwargs = (; Δx, reg);
init_kwargs = (; m=3);
opt = OptimizationOptimisers.Adam(lr, (0.9, 0.999)); 
dataset = Generator.read_dataset("./dataset/kdv_high_dim_n128_m3_t5_128_x4pi_64_up2.jld2")["training_set"];
model = Models.CNN2(9, [2, 4, 8, 8, 4, 2, 1]);

K, Θ, _ = training(model, epochs, dataset, opt, batch_size, ratio, noise, sol, snap_kwargs);
# # ===

include("../../utils/graphic_tools.jl")

t, u₀, u = Generator.get_kdv_batch(tₘₐₓ, tₘᵢₙ, xₘₐₓ, xₘᵢₙ, tₙ, xₙ, 2, init_kwargs);
u₀ = reshape(u₀, :, 1, 1);
# t, u = dataset[1];
# u₀ = reshape(u[:, 1], :, 1, 1);

prob = ODEProblem((u, p, t) -> K(u), u₀, extrema(t), Θ, saveat=t);
U = Array(solve(prob, Tsit5()));
GraphicTools.show_state(U[:, 1, 1, :], t, x, "", "t", "x")