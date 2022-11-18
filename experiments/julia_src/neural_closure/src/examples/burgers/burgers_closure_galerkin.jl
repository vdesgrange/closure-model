using BSON: @save
using Flux
using IterTools: ncycle
using Optimization
using OptimizationOptimisers
using Plots
using SciMLSensitivity

include("../../utils/processing_tools.jl")
include("../../neural_ode/models.jl")
include("../../neural_ode/regularization.jl")
include("../../utils/generators.jl");
include("../../utils/graphic_tools.jl")
include("../../equations/burgers/burgers_gp.jl")
include("../../rom/pod.jl")

function f(u, p, t)
    ν = p[1]
    Δx = p[2]
    u₋ = circshift(u, 1)
    u₊ = circshift(u, -1)
    a₊ = u₊ + u
    a₋ = u + u₋
    du = @. -((a₊ < 0) * u₊^2 + (a₊ > 0) * u^2 - (a₋ < 0) * u^2 - (a₋ > 0) * u₋^2) / Δx +
       ν * (u₋ - 2u + u₊) / Δx^2
    du
end

del_dim(x::Array{Float64,4}) = reshape(x, (size(x)[1], size(x)[3], size(x)[4]))
del_dim(x::Array{Float64,3}) = x

function get_Φ(dataset, m)
    tmp = []
    for (i, data) in enumerate(dataset)
        push!(tmp, data[2])
    end
    u_cat = Array(cat(tmp...; dims = 3))
    xₙ = size(u_cat, 1)
    bas, _ = POD.generate_pod_svd_basis(reshape(u_cat, xₙ, :), false)
    λ = bas.eigenvalues
    @show POD.get_energy(λ, m)

    Φ = bas.modes[:, 1:m]

    return Φ
end

function training(
    model,
    epochs,
    dataset,
    opt,
    batch_size,
    ratio,
    noise = 0.0,
    solver = Tsit5(),
    kwargs = (;),
)
    ep = 0
    count = 0
    ν, Δx, reg = kwargs
    losses = []

    @info("Loading dataset") # train_dataset ou reference
    (train_loader, val_loader) =
        ProcessingTools.get_data_loader(dataset, batch_size, ratio, false)

    @info("Building model")
    p, re = Flux.destructure(model)
    fᵣₒₘ = (v, p, t) -> re(p)(v)

    function f_closure(v, p, t)
        Φ' * f(Φ * v, (ν, Δx), t) + fᵣₒₘ(v, p, t)
    end

    function predict_neural_ode(θ, x, t)
        prob = ODEProblem(f_closure, x, extrema(t), θ)
        # prob = ODEProblem(fᵣₒₘ, x, extrema(t), θ, saveat=t);
        sol = solve(
            prob,
            solver;
            saveat = t,
            sensealg = InterpolatingAdjoint(; autojacvec = ZygoteVJP()),
        )
        ȳ = Array(sol)
        return permutedims(del_dim(ȳ), (1, 3, 2))
    end

    function loss_trajectory_fit(θ, x, u, t)
        xₙ, tₙ, bₙ = size(u)
        mₙ = size(Φ, 2)

        x̂ = Reg.gaussian_augment(Φ' * x, noise)
        # x̂ = reshape(x̂, mₙ, 1, bₙ) # CNN
        v̂ = predict_neural_ode(θ, x̂, t[:, 1])

        u = reshape(u, xₙ, :)
        v = Φ' * u
        v = reshape(v, mₙ, tₙ, :)

        l = Flux.mse(v̂, v) + reg * sum(θ)
        return l
    end

    function loss_derivative_fit(θ, x, u, t)
        xₙ, tₙ, bₙ = size(u)
        mₙ = size(Φ, 2)

        u = reshape(u, xₙ, :)
        v = Φ' * u
        dv = f(v, (ν, Δx), t)
        # dv = reshape(dv, mₙ, tₙ, :);  # CNN

        # v = reshape(v, mₙ, 1, :) # CNN
        dv̂ = f_closure(v, θ, t)
        # dv̂ = fᵣₒₘ(v, θ, t);
        # dv̂ = reshape(dv̂, mₙ, tₙ, bₙ)  # CNN

        l = Flux.mse(dv̂, dv) + reg * sum(θ)
        return l
    end

    function val_loss(θ, x, u, t)
        xₙ = size(u, 1)
        x = Φ' * x

        # x = reshape(x, size(x, 1), 1, :); # CNN
        v̂ = predict_neural_ode(θ, x, t[:, 1])

        u = reshape(u, xₙ, :)
        v = Φ' * u
        v = reshape(v, size(v̂, 1), size(v̂, 2), :)

        lt = Flux.mse(v̂, v)
        return lt
    end

    function cb(θ, l)
        # @show(l)
        count += 1

        # iter = (train_loader.nobs / train_loader.batchsize)
        iter = 8
        if (count % iter == 0)
            ep += 1
            count = 0
            loss = 0

            for (x, y, t) in val_loader
                loss += val_loss(θ, x, y, t)
            end
            # loss /= (val_loader.nobs / val_loader.batchsize)

            @info("Epoch ", ep, loss)
            push!(losses, loss)
            # Plots.plot!(pl, LinRange(1, epochs, 1), val_loss);
            # display(pl)

        end
        return false
    end

    @info("Initiate training")
    optf = OptimizationFunction(
        # (θ, p, x, y, t) -> loss_trajectory_fit(θ, x, y, t),
        (θ, p, x, y, t) -> loss_derivative_fit(θ, x, y, t),
        Optimization.AutoZygote(),
    )
    optprob = Optimization.OptimizationProblem(optf, p)
    result_neuralode =
        Optimization.solve(optprob, opt, ncycle(train_loader, epochs); callback = cb)

    return re(result_neuralode), Array(result_neuralode)
end

dataset = Generator.read_dataset(
    "./dataset/viscous_burgers_high_dim_t1_64_x1_64_nu0.001_typ2_m100_256_up2_j173.jld2",
)["training_set"];
Φ_dataset, train_dataset = dataset[1:128], dataset[129:end];

epochs = 10;
ratio = 0.75; # 0.75
batch_size = 8;
lr = 1e-3;
reg = 1e-7;
noise = 0.05;
tₘₐₓ = 1.0; # 2.
tₘᵢₙ = 0.0;
xₘₐₓ = 1.0; # pi
xₘᵢₙ = 0;
tₙ = 64;
xₙ = 64;
ν = 0.001; # 0.04
Δx = (xₘₐₓ - xₘᵢₙ) / (xₙ - 1)
solver = Tsit5();
x = LinRange(xₘᵢₙ, xₘₐₓ, xₙ);
t = LinRange(tₘᵢₙ, tₘₐₓ, tₙ);
m = 10;
snap_kwargs = (; ν, Δx, reg);

# === Get basis Φ ===
Φ = get_Φ(Φ_dataset, m);

# === Train ===
opt = OptimizationOptimisers.Adam(lr, (0.9, 0.999));
# model = Models.CNN2(9, [2, 4, 8, 8, 4, 2, 1]);
model = Flux.Chain(
    v -> vcat(v, v .^ 2),
    Flux.Dense(2m => 2m, tanh; init = Models.glorot_uniform_float64, bias = true),
    Flux.Dense(2m => m, tanh; init = Models.glorot_uniform_float64, bias = true),
    Flux.Dense(m => m, identity; init = Models.glorot_uniform_float64, bias = true),
)
model.layers[2].weight
model.layers[2].bias

K, θ = training(
    model,
    epochs,
    train_dataset,
    opt,
    batch_size,
    ratio,
    noise,
    solver,
    snap_kwargs,
);
# @save "./models/fnn_3layers_viscous_burgers_high_dim_t2_64_xpi_64_nu0.04_typ2_m10_256_up16_j173.bson" K θ


# === Check results ===

function f_closure(v, p, t)
    Φ' * f(Φ * v, (ν, Δx), t) + K(v)
end

t, u, _, _ = train_dataset[2];
v₀ = Φ' * u[:, 1];
û = galerkin_projection(t, u, Φ, ν, Δx, Δt);
û_prob = ODEProblem((v, p, t) -> (Φ' * f(Φ * v, p, t)), v₀, extrema(t), (ν, Δx));
solve(
    û_prob,
    solver;
    saveat = t,
    sensealg = InterpolatingAdjoint(; autojacvec = ZygoteVJP()),
)
ū = Φ * Array(sol);
uₙₙ_prob = ODEProblem((v, p, t) -> K(v), v₀, extrema(t), θ; saveat = t);
reshape(v₀, (size(v₀, 1), :));
uₙₙ = Φ * Array(solve(uₙₙ_prob, Tsit5()));
uᵧₙₙ_prob = ODEProblem(f_closure, v₀, extrema(t), θ; saveat = t);
uᵧₙₙ = Φ * Array(solve(uᵧₙₙ_prob, Tsit5()));
for (i, t) ∈ enumerate(t)
    pl = Plots.plot(; xlabel = "x", ylim = extrema(u))
    Plots.plot!(pl, x, u[:, i]; label = "FOM - Model 0")
    # Plots.plot!(pl, x, û[:, i]; label = "ROM - Model 1.0 GP")
    Plots.plot!(pl, x, ū[:, i]; label = "ROM - Model 1.5 Φ'f(Φv)")
    # Plots.plot!(pl, x, uₙₙ[:, i]; label = "ROM - Model 2 NN")
    Plots.plot!(pl, x, uᵧₙₙ[:, i]; label = "ROM - Model 3  Φ'f(Φv) + NN(v)")
    # Plots.plot!(pl, x, uₙₙ[:, i]; label = "ROM - Model 4 GP + NN")
    display(pl)
end


# t, u₀, u = Generator.get_burgers_batch(tₘₐₓ, tₘᵢₙ, xₘₐₓ, xₘᵢₙ, tₙ, xₙ, ν, 2, (; m = 10));
# v = Φ' * u;
# v₀ = v[:, 1];
# û = galerkin_projection(t, u, Φ, ν, Δx, Δt);

# _prob = ODEProblem((u, p, t) -> K(u), reshape(v₀, :, 1, 1), extrema(t), θ; saveat = t);
# v̄ = solve(_prob, Tsit5());

# display(GraphicTools.show_state(Φ * v, t, x, "", "t", "x"))
# display(GraphicTools.show_state(Φ * v̄[:, 1, 1, :], t, x, "", "t", "x"))
# display(GraphicTools.show_state(Φ * v̂, t, x, "", "t", "x"))

# for (i, t) ∈ enumerate(t)
#     pl = Plots.plot(; xlabel = "x", ylim = extrema(v))
#     Plots.plot!(pl, x, v[:, i]; label = "FOM - v - Model 0")
#     # Plots.plot!(pl, x, v̂[:, i]; label = "ROM - v̂ - Model 1")
#     Plots.plot!(pl, x, v̄[:, 1, 1, i]; label = "ROM - v̄ - Model 2")
#     display(pl)
# end

# for (i, t) ∈ enumerate(t)
#     pl = Plots.plot(; xlabel = "x", ylim = extrema(v))
#     Plots.plot!(pl, x, u[:, i]; label = "FOM - Model 0")
#     Plots.plot!(pl, x, û[:, i]; label = "ROM - Model 1")
#     Plots.plot!(pl, x, Φ * v̄[:, 1, 1, i]; label = "ROM - Model 2")
#     display(pl)
# end
