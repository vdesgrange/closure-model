using Statistics
using LaTeXStrings
using JLD2
using BSON
include("closure.jl")
include("../../equations/equations.jl");
include("../../utils/graphic_tools.jl")

function f_kdv(u, (Δx), t)
    u₋₋ = circshift(u, 2);
    u₋ = circshift(u, 1);
    u₊ = circshift(u, -1);
    u₊₊ = circshift(u, -2);
  
    uₓ = (u₊ .- u₋) ./ (2 * Δx);
    uₓₓₓ = (-u₋₋ .+ 2 .*u₋ .- 2u₊ + u₊₊) ./ (2 * Δx^3);
    uₜ = -(6u .* uₓ) .- uₓₓₓ;
    return uₜ
end

function downsampling(u, d)
    (Nₓ, b, Nₜ) = size(u);
    Dₜ = Int64(Nₜ / d);
    u₁ = u[:, :, 1:d:end]; # Take t value every d steps
    Dₓ = Int64(Nₓ / d);
    u₂ = sum(reshape(u₁, d, Dₓ, b, Dₜ), dims=1);
    u₃   = reshape(u₂, Dₓ, b, Dₜ) ./ d;
    return u₃
end

function create_data_kdv(f, p, K, x, nsolution, t; decay = k -> 1 / (1 + abs(k)), kwargs...)
    L = x[end]
    basis = [exp(2π * im * k * x / L) for x ∈ x, k ∈ -K:K]
    c = [randn() * decay(k) * exp(-2π * im * rand()) for k ∈ -K:K, _ ∈ 1:nsolution]
    u₀ = real.(basis * c)
    predict(f, u₀, p, t, Rodas4P(), dt=0.1; kwargs...)
end

lr = 1e-3;
reg = 1e-7;
tₘₐₓ= 10; # 2.
tₘᵢₙ = 0.;
xₘₐₓ = 30; # pi
xₘᵢₙ = 0;
tₙ = 64;
xₙ = 64;

sol = Tsit5();

# discretization
up = 2;
x = LinRange(xₘᵢₙ, xₘₐₓ, up * xₙ);

# Training times
t_train = LinRange(0.0, tₘₐₓ, Int(up * tₙ));
t_valid = LinRange(0.0, tₘₐₓ, Int(up * tₙ));
t_test = LinRange(0.0, tₘₐₓ, Int(up * tₙ));

vx = LinRange(xₘᵢₙ, xₘₐₓ, xₙ);
vt_train = Array(t_train)[1:up:end];
vt_valid = Array(t_valid)[1:up:end];
vt_test = Array(t_test)[1:up:end];

# === Data ===
K = 20;  # Maximum frequency in initial conditions
ff = f_kdv
Δx = (xₘₐₓ - xₘᵢₙ) / (up * xₙ - 1);
u_train = create_data_kdv(ff, (Δx), K, x, 8, t_train; sensealg=DiffEqSensitivity.InterpolatingAdjoint(; autojacvec=ZygoteVJP()));
# u_valid  = create_data_kdv(ff, (Δx), K, x, 64, t_valid; reltol = 1e-6, abstol = 1e-6, sensealg=DiffEqSensitivity.InterpolatingAdjoint(; autojacvec=ZygoteVJP()));
# u_test  = create_data_kdv(ff, (Δx), K, x, 32, t_test; reltol = 1e-6, abstol = 1e-6, sensealg=DiffEqSensitivity.InterpolatingAdjoint(; autojacvec=ZygoteVJP()));

# JLD2.save("./dataset/train_kdv_fouriers_t1_64_xpi_64_typ5_K100_256_up16_j173.jld2", "training_set", iv_train);
# JLD2.save("./dataset/valid_kdv_fouriers_t1_64_xpi_64_typ5_K100_256_up16_j173.jld2", "validation_set", v_valid);
# JLD2.save("./dataset/test_kdv_fouriers_t1_64_xpi_64_typ5_K100_256_up16_j173.jld2", "test_set", v_test);


Δx2 = (xₘₐₓ - xₘᵢₙ) / (xₙ - 1);
v_train = downsampling(u_train, up);
# v_valid = downsampling(u_valid, up);
# v_test  = downsampling(u_test, up);

# v_train = JLD2.load("./dataset/train_kdv_fouriers_t1_64_xpi_64_typ5_K100_256_up16_j173.jld2")["training_set"];
# v_valid = JLD2.load("./dataset/valid_kdv_fouriers_t1_64_xpi_64_typ5_K100_256_up16_j173.jld2")["validation_set"];
# v_test = JLD2.load("./dataset/test_kdv_fouriers_t1_64_xpi_64_typ5_K100_256_up16_j173.jld2")["test_set"];


dvdt_train = ff(v_train, (Δx2), 0);
# dvdt_valid = ff(v_valid, (ν, Δx2), 0);
# dvdt_test = ff(v_test, (ν, Δx2), 0);

gr();
display(GraphicTools.show_state(v_train[:, 1, :], vt_train, vx, "Low-resolution with FOM model", "t", "x"))
# display(GraphicTools.show_state(invi_v_train[:, 1, :], vt_train, vx, "Low-resolution with FOM model", "t", "x"))

# begin
#     plt = plot(title="Initial condition", xlabel="t", ylabel="u", background_color_legend = RGBA(1, 1, 1, 0.8))
#     plot!(plt, vt_train, invi_v_train[:, 1, 1], c=:coral, label=L"u_0")
# end

# iplot = 1;
# for (i, t) ∈ enumerate(vt_train)
#     pl = Plots.plot(;title = @sprintf("Solution, t = %.3f", t), xlabel = "x", ylims = extrema(invi_v_train[:, iplot, :]))
#     plot!(pl, vx, invi_v_train[:, iplot, i]; label = "Downscaled FOM")
#     display(pl)
#     sleep(0.05)
# end


# === Model ===

function loss_trajectory_fit_cnn(
    f,
    p,
    v,
    t;
    solver = Tsit5(),
    λ = 1e-8,
    nsolution = size(v, 2),
    ntime = size(v, 3),
    kwargs...)
    # Select a random subset (batch)
    is = Zygote.@ignore sort(shuffle(1:size(v, 2))[1:nsolution])
    it = Zygote.@ignore sort(shuffle(1:length(t))[1:ntime])
    v = v[:, is, it]
    t = t[it]

    # Predicted soluton
    (xₙ, bₙ, tₙ) = size(v);
    # v = reshape(v, xₙ, 1, bₙ, tₙ); # CNN 
    sol = predict(f, v[:, :, 1], p, t, solver; kwargs...);
    sol = reshape(sol, xₙ, bₙ, tₙ); # CNN 
    # v = reshape(v, xₙ, bₙ, tₙ); # CNN 

    # Relative squared error
    data = sum(abs2, sol - v) / sum(abs2, v)

    # Regularization term
    reg = sum(abs2, p) / length(p)

    # Loss
    data + λ * reg
end

function loss_derivative_fit_cnn(f, p, dvdt, v; λ = 1e-8, nsample = size(v, 2))
    # Select a random subset (batch)
    i = Zygote.@ignore sort(shuffle(1:size(v, 2))[1:nsample])
    v = v[:, i]
    dvdt = dvdt[:, i]

    (xₙ, tₙ) = size(dvdt);

    # v = reshape(v, xₙ, 1, :) # CNN
    # dvdt = reshape(dvdt, xₙ, 1, tₙ);  # CNN

    # Predicted right hand side
    rhs = f(v, p, 0)

    # Relative squared error
    data = sum(abs2, rhs - dvdt) / sum(abs2, dvdt)

    # Regularization term
    reg = sum(abs2, p) / length(p)

    # Loss
    data + λ * reg
end

function relative_error_cnn(f, p, v, t, solver; kwargs...)
    # Predicted soluton
    (xₙ, bₙ, tₙ) = size(v);
    sol = predict(f, v[:, :, 1], p, t, solver; kwargs...)
    sol = reshape(sol, xₙ, bₙ, tₙ); # CNN 

    # Relative error
    norm(sol - v) / norm(v)
end

function create_callback_cnn(f, v, t; ncallback = 1, solver = Tsit5(), kwargs...)
    i = 0
    ep = 0;
    iters = Int[]
    epochs = Int[]
    errors = zeros(0)
    function callback(p, loss)
        i += 1
        if i % ncallback == 0
            ep += 1;
            e = relative_error_cnn(f, p, v, t, solver; kwargs...)
            push!(iters, i);
            push!(epochs, ep);
            push!(errors, e);
            println("Epoch $ep \t relative error $e")
            plt = Plots.plot(epochs, errors; xlabel = "Epochs", title = "Relative error");
            savefig(pl, "inviscid_loss_per_epoch.png")
            # display(plot(epochs, errors; xlabel = "Epochs", title = "Relative error")) # Iteration
        end
        return false
    end
end

@info("Load model");
model = Models.CNN2(9, [2, 4, 8, 8, 4, 2, 1]);
p₀, re = Flux.destructure(model);
fᵣₒₘ(v, p, t) = re(p)(v);

# loss_df(p, _) = loss_derivative_fit_cnn(
#     fᵣₒₘ,
#     p,
#     reshape(invi_dvdt_train, xₙ, :),
#     reshape(invi_v_train, xₙ, :);
#     nsample = 64,
# );

# p_df = train(
#     loss_df,
#     p₀;
#     maxiters = 100 * (192 / 64), # * (192 / 16)
#     optimizer = OptimizationOptimisers.Adam(lr),
#     callback = create_callback_cnn(
#         fᵣₒₘ,
#         invi_v_valid,
#         vt_valid;
#         ncallback = (192 / 64),
#         reltol = 1e-6,
#         abstol = 1e-6,
#     ),
# );

loss_tf(p, _) = loss_trajectory_fit_cnn(
    fᵣₒₘ,
    p,
    invi_v_train,
    vt_train;
    nsolution = 16,
    reltol = 1e-6,
    abstol = 1e-6,
);

epochs = 500;
p_tf = train(
    loss_tf,
    p₀;
    maxiters = epochs * (192 / 16),
    optimizer = OptimizationOptimisers.Adam(lr),
    callback = create_callback_cnn(
        fᵣₒₘ,
        invi_v_valid,
        vt_valid;
        ncallback = (192 / 16),
        reltol = 1e-6,
        abstol = 1e-6,
    ),
);

@info("Save results");
savefig("inviscid_loss_per_epoch.png")
BSON.@save "./models/pure_node_inviscid/inviscid_burgers_fouriers_t2_64_xpi_64_nu0_typ2_K100_256_up16_j173.bson" model p_tf

# BSON.@load "./models/pure_node_inviscid/inviscid_burgers_fouriers_t2_64_xpi_64_nu0_typ2_K100_256_up16_j173.bson" model p_tf

# begin
#     # sol_df = predict(
#     #     fᵣₒₘ,
#     #     reshape(v_test[:, :, 1], xₙ, 1, :),
#     #     p_df,
#     #     vt_test,
#     #     Tsit5();
#     #     reltol = 1e-6,
#     #     abstol = 1e-6,
#     # );

#     sol_tf = predict(
#         fᵣₒₘ,
#         invi_v_test[:, :, 1],
#         p_tf,
#         vt_test,
#         Tsit5();
#         reltol = 1e-6,
#         abstol = 1e-6,
#     );

#     iplot = 1;
#     for (i, t) ∈ enumerate(vt_test)
#         pl = Plots.plot(;title = @sprintf("Solution, t = %.3f", t), xlabel = "x", ylims = extrema(invi_v_test[:, iplot, :]))
#         plot!(pl, vx, invi_v_test[:, iplot, i]; label = "FOM")
#         plot!(pl, vx, sol_tf[:, iplot, i]; label = "ROM neural closure (DF)")
#             # plot!(pl, x, sol_tf[:, iplot, i]; label = "ROM neural closure (TF)")
#         display(pl)
#         sleep(0.05)
#     end
# end