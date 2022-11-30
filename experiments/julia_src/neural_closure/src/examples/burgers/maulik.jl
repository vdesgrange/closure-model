using BenchmarkTools
using AbstractFFTs
using FFTW
using Statistics
using LaTeXStrings
include("closure_training.jl");
include("../../equations/equations.jl");

function analytical_solution(x, t)
    Re = 1. / ν;
    t₀ = exp(Re / 8.);
    a =  x ./ (t .+ 1);
    b =  1 .+ sqrt.((t .+ 1) ./ t₀) .* exp.(Re .* (x.^2 ./ (4 .* t .+ 4)));
    u = a ./ b;
end

function fflux2(u, (ν, Δx), t)
    u₋ = circshift(u, 1)
    u₊ = circshift(u, -1)
    # u₊[end] = 0;
    # u₋[1] = 0;
    a₊ = u₊ + u
    a₋ = u + u₋
    du = @. -((a₊ < 0) * u₊^2 + (a₊ > 0) * u^2 - (a₋ < 0) * u^2 - (a₋ > 0) * u₋^2) / Δx + ν * (u₋ - 2u + u₊) / Δx^2
    du = [
        zeros(1, size(du)[2:end]...);
        du[2:end-1, :, :];
        zeros(1, size(du)[2:end]...);
    ]
    du
end
fflux2(u::AbstractMatrix, p, t) = reshape(fflux2(reshape(u, size(u,1), 1, size(u, 2)), p, t), size(u, 1), size(u, 2))
fflux2(u::AbstractVector, p, t) = reshape(fflux2(reshape(u, size(u,1), 1, 1), p, t), size(u, 1))

function create_data(f, p, K, x, nsolution, t; decay = k -> 1 / (1 + abs(k)), kwargs...)
    L = x[end]
    basis = [exp(2π * im * k * x / L) for x ∈ x, k ∈ -K:K]
    c = [randn() * decay(k) * exp(-2π * im * rand()) for k ∈ -K:K, _ ∈ 1:nsolution]
    u₀ = real.(basis * c)
    @show size(u₀)
    predict(f, u₀, p, t, Tsit5(); kwargs...)
end

function create_data_maulik(f, p, K, x, nsolution, t; kwargs...)
    Re = 1. / ν;
    t₀ = exp(Re / 8.);
    c = x ./ (1 .+ sqrt(1 / t₀) * exp.(Re * (x.^2 ./ 4)));
    c[1] = 0;
    c[end] = 0;
    u₀ = hcat([c for _ ∈ 1:nsolution]...);
    # Solve ODE
    predict(f, u₀, p, t, Tsit5(); kwargs...)
end

create_data_maulik_analytique(x, t) = reshape(
    analytical_solution(x, t'),
    length(x),
    1,
    length(t),
)

function downsampling(u, d)
    (Nₓ, b, Nₜ) = size(u);
    Dₜ = Int64(Nₜ / d);
    u₁ = u[:, :, 1:d:end]; # Take t value every d steps
    Dₓ = Int64(Nₓ / d);
    u₂ = sum(reshape(u₁, d, Dₓ, b, Dₜ), dims=1);
    u₃   = reshape(u₂, Dₓ, b, Dₜ) ./ d;
    return u₃
end

epochs = 100;
ratio = 0.75;
batch_size = 64; # High for derivative fitting (i.e. 64)
lr = 1e-3;
reg = 1e-7;
noise = 0.05;
tₘₐₓ= 6.; # 2.
tₘᵢₙ = 0.;
xₘₐₓ = 1.; # pi
xₘᵢₙ = 0;
tₙ = 64;
xₙ = 64;
ν = 0.001; # Viscosity
K = 100;  # Maximum frequency in initial conditions# Sampling rate, inverse of sample spacing
sol = Tsit5();

# FOM discretization
up = 16;
x = LinRange(xₘᵢₙ, xₘₐₓ, up * xₙ);
Δx = (xₘₐₓ - xₘᵢₙ) / (up * xₙ - 1);
k = 2 * pi * AbstractFFTs.fftfreq(up * xₙ, 1. / Δx);

# Training times
t_pod = LinRange(0.0, 2., Int(up * tₙ));
t_train = LinRange(0.0, 2., Int(up * tₙ));
t_valid = LinRange(0.0, 4., Int(up * tₙ));
t_test = LinRange(0.0, tₘₐₓ, Int(up * tₙ));

# === Generate data ===
# u_pod = analytical_solution(x, Array(t_pod)')

# ff = Equations.f_fft;
# pᵣ =  (ν, Δx, k);
# u_pod = create_data_maulik(ff, pᵣ, K, x, 1, t_pod; reltol = 1e-6, abstol = 1e-6);
# u_train = create_data_maulik(ff, pᵣ, K, x, 1, t_train; reltol = 1e-6, abstol = 1e-6);
# u_valid  = create_data_maulik(ff, pᵣ, K, x, 1, t_valid; reltol = 1e-6, abstol = 1e-6);
# u_test  = create_data_maulik(ff, pᵣ, K, x, 1, t_test; reltol = 1e-6, abstol = 1e-6);

u_pod    = create_data_maulik_analytique(x, t_pod);
u_train  = create_data_maulik_analytique(x, t_train);
u_valid  = create_data_maulik_analytique(x, t_valid);
u_test   = create_data_maulik_analytique(x, t_test);



# ū_pod = mean(u_pod, dims=3)[:, 1]; # 2, or 3 if batch
# ū_train = mean(u_train, dims=3)[:, 1];
# ū_valid = mean(u_valid, dims=3)[:, 1];
# ū_test = mean(u_test, dims=3)[:, 1];


# Time derivatives
dudt_pod = ff(u_pod, pᵣ, 0);
dudt_train = ff(u_train, pᵣ, 0);
dudt_valid = ff(u_valid, pᵣ, 0);
dudt_test = ff(u_test, pᵣ, 0);

vx = LinRange(xₘᵢₙ, xₘₐₓ, xₙ);
Δx2 =  (xₘₐₓ - xₘᵢₙ) / (xₙ - 1);
k2 = 2 * pi * AbstractFFTs.fftfreq(xₙ, 1. / Δx2);
vt_pod = Array(t_pod)[1:up:end];
vt_train = Array(t_train)[1:up:end];
vt_valid = Array(t_valid)[1:up:end];
vt_test = Array(t_test)[1:up:end];

v_pod = downsampling(u_pod, up);
v_train = downsampling(u_train, up);
v_valid = downsampling(u_valid, up);
v_test  = downsampling(u_test, up);

p2 =  (ν, Δx2, k2);
dvdt_pod = ff(v_pod, p2, 0);
dvdt_train = ff(v_train, p2, 0);
dvdt_valid = ff(v_valid, p2, 0);
dvdt_test = ff(v_test, p2, 0);

# v̄_pod = mean(v_pod, dims=3)[:, 1]; # 2, or 3 if batch
# v̄_train = mean(v_train, dims=3)[:, 1];
# v̄_valid = mean(v_valid, dims=3)[:, 1];
# v̄_test = mean(v_test, dims=3)[:, 1];

# Get POD basis
# Maulik 2020 : POD.generate_pod_basis(u, true); (f_fft) fflux + (T=2) + eigenproblem + substract mean (0.86 for fflux, 0.88 for fflux2)
# Gupta 2021 : POD.generate_pod_svd_basis(u, false); fflux2 + (T=2 or 4) + svd + substract mean
# bas, _ = POD.generate_pod_svd_basis(u_pod[:, 1, :], true); # gupta
# @show POD.get_energy(bas.eigenvalues.^2, m);
m = 3;  # Number of POD modes
bas, sm = POD.generate_pod_basis(u_pod[:, 1, :], true); # maulik
@show POD.get_energy(bas.eigenvalues, m);
Φ = bas.modes[:, 1:m];
u₀ = u_pod[:, 1, 1];
v₀ = v_pod[:, 1, 1];

## GP 1, f = finite diff (No)
# B_k, L_k, N_k = offline_op(sm, Φ, ν, Δx);
# @time begin
#     a₀_pod = Φ' * (v₀ .- sm);
#     A = predict(Equations.gp,  a₀_pod, (B_k, L_k, N_k), vt_pod, Tsit5());
# end;

## GP 2, f = fft
gg(a, p, t) = Φ' * ff(Φ * a + sm, p, t);
@time begin
    a₀_pod = Φ' * (u₀ .- sm);
    A = predict(gg,  a₀_pod, (ν, Δx, k), t_pod, Tsit5());
end;

# display(GraphicTools.show_state(sm .+ Φ * A, t_pod, x, "", "t", "x"))

begin
    plt = plot(title="Coefficients", xlabel="t", ylabel="ϕ", background_color_legend = RGBA(1, 1, 1, 0.8))
    plot!(plt; dpi=600)
    plot!(plt, t_pod, bas.coefficients[1, :], c=:coral, label=L"a_1")
    plot!(plt, t_pod, bas.coefficients[2, :], c=:green, label=L"a_2")
    plot!(plt, t_pod, bas.coefficients[3, :], c=:blue, label=L"a_3")

    # plot!(plt, vt_pod, A[1, :], c=:coral, linestyle=:dash,label=L"gp_1")
    # plot!(plt, vt_pod, A[2, :], c=:green, linestyle=:dash,label=L"gp_2")
    # plot!(plt, vt_pod, A[3, :], c=:blue, linestyle=:dash,label=L"gp_3")

    # plot!(plt, t_train, bas.coefficients[1, :], c=:coral, label=L"a_1")
    # plot!(plt, t_train, bas.coefficients[2, :], c=:green, label=L"a_2")
    # plot!(plt, t_train, bas.coefficients[3, :], c=:blue, label=L"a_3")
    plot!(plt, t_pod, a_pod[1, 1, :], c=:coral, linestyle=:dot,label=L"1")
    plot!(plt, t_pod, a_pod[2, 1, :], c=:green, linestyle=:dot,label=L"2")
    plot!(plt, t_pod, a_pod[3, 1, :], c=:blue, linestyle=:dot,label=L"3")

    plot!(plt, t_pod, A[1, :], c=:coral, linestyle=:dash,label=L"gp2_1")
    plot!(plt, t_pod, A[2, :], c=:green, linestyle=:dash,label=L"gp2_2")
    plot!(plt, t_pod, A[3, :], c=:blue, linestyle=:dash,label=L"gp2_3")
end

# a_pod = tensormul(Φ', u_pod); # v_pod = Φ' * u_pod ### A = Φ' * (u_pod .- mean(u_pod, dims=2));
# a_train = tensormul(Φ', u_train);
# a_valid = tensormul(Φ', u_valid);
# a_test = tensormul(Φ', u_test);

a_pod = tensormul(Φ', u_pod .- sm); # v_pod = Φ' * u_pod ### A = Φ' * (u_pod .- mean(u_pod, dims=2));
a_train = tensormul(Φ', u_train .- sm);
a_valid = tensormul(Φ', u_valid .- sm);
a_test = tensormul(Φ', u_test .- sm);

# Time derivatives of POD coefficients
dadt_pod = tensormul(Φ', dudt_pod);
dadt_train = tensormul(Φ', dudt_train);
dadt_valid = tensormul(Φ', dudt_valid);
dadt_test = tensormul(Φ', dudt_test);

# display(GraphicTools.show_state(v_train[:, 1, :], vt_train, x, "", "t", "x"))
# display(GraphicTools.show_state(sm .+ Φ * a_train[:, 1, :], vt_train, x, "", "t", "x"))


# === Maulik paper batching : batch per time ===

# ntime = 20
# it = Zygote.@ignore sort(shuffle(1:length(t_pod))[1:20])
# sort(shuffle(1:length(t_pod))[1:20])
# sort(shuffle(1:size(v_train, 2))[1:8])
# it2 = Zygote.@ignore sort(shuffle(1:length(t_pod) - ntime)[1:ntime])

function loss_time_trajectory_fit(
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
    # it = Zygote.@ignore sort(shuffle(1:length(t))[1:ntime])
    st = Zygote.@ignore rand(1:(length(t) - ntime))
    it = Zygote.@ignore st:(st + ntime)

    v = v[:, is, it]
    t = t[it]

    # Predicted soluton
    sol = predict(f, v[:, :, 1], p, t, solver; kwargs...)

    # Relative squared error
    data = sum(abs2, sol - v) / sum(abs2, v)

    # Regularization term
    reg = sum(abs2, p) / length(p)

    # Loss
    data + λ * reg
end


# ==== Model ====
model = Flux.Chain(
    v -> vcat(v, v .^ 2),
    Flux.Dense(2m => 40, tanh; init = Models.glorot_uniform_float64, bias = true),
    Flux.Dense(40 => m, identity; init = Models.glorot_uniform_float64, bias = true),
);
# model = Models.CNN2(9, [2, 4, 8, 8, 4, 2, 1]);
p₀, re = Flux.destructure(model);
fᵣₒₘ(v, p, t) = re(p)(v);
f_closure(v, p, t) = Φ' * fflux2(Φ * v, (ν, Δx, k), t) + re(p)(v);

loss_df(p, _) = loss_derivative_fit(
    fᵣₒₘ, # fᵣₒₘ,
    p,
    reshape(dadt_train, m, :),
    reshape(a_train, m, :);
    nsample = batch_size,
);

p_df = train(
    loss_df,
    p₀;
    maxiters = 5000,
    optimizer = OptimizationOptimisers.Adam(lr),
    callback = create_callback(
        fᵣₒₘ,
        a_valid,
        t_valid;
        ncallback = batch_size,
        reltol = 1e-6,
        abstol = 1e-6,
    ),
);


loss_tf(p, _) = loss_time_trajectory_fit( # loss_time_trajectory_fit
    f_closure, # fᵣₒₘ
    p,
    a_train,
    t_train;
    nsolution = 1,
    ntime = 10,
    reltol = 1e-6,
    abstol = 1e-6,
);

p_tf = train(
    loss_tf,
    p₀;
    maxiters = 3000,
    optimizer = OptimizationOptimisers.Adam(lr),
    callback = create_callback(
        f_closure,
        a_valid,
        t_valid;
        ncallback = 8,
        reltol = 1e-4,
        abstol = 1e-6,
    ),
);

# === Results === 
sol_df = predict(
    fᵣₒₘ,
    a_test[:, :, 1],
    p_df,
    t_train,
    Tsit5();
    reltol = 1e-6,
    abstol = 1e-6,
);

sol_tf₀ = predict(
    fᵣₒₘ,
    a_test[:, :, 1],
    p₀,
    t_train,
    Tsit5();
    reltol = 1e-6,
    abstol = 1e-6,
);



sol_tf = predict(
    fᵣₒₘ,
    a_test[:, :, 1],
    p_tf,
    t_train,
    Tsit5();
    reltol = 1e-6,
    abstol = 1e-6,
);

iplot = 1;
begin
    plt = plot(title="Coefficients", xlabel="t", ylabel="a", background_color_legend = RGBA(1, 1, 1, 0.8))
    plot!(plt; dpi=600)
    
    # Reference
    plot!(plt, t_pod, bas.coefficients[1, :], c=:coral, label=L"a_1")
    plot!(plt, t_pod, bas.coefficients[2, :], c=:green, label=L"a_2")
    plot!(plt, t_pod, bas.coefficients[3, :], c=:blue, label=L"a_3")

    # Baseline
    # plot!(plt, t_train, A[1, :], c=:coral, linestyle=:dash,label=L"gp_1")
    # plot!(plt, t_train, A[2, :], c=:green, linestyle=:dash,label=L"gp_2")
    # plot!(plt, t_train, A[3, :], c=:blue, linestyle=:dash,label=L"gp_3")

    # NODE (DF)
    # plot!(plt, t_train, sol_df[1, 1, :], c=:coral, linestyle=:dot,label=L"NODE_1\ (DF)")
    # plot!(plt, t_train, sol_df[2, 1, :], c=:green, linestyle=:dot,label=L"NODE_2\ (DF)")
    # plot!(plt, t_train, sol_df[3, 1, :], c=:blue, linestyle=:dot,label=L"NODE_3\ (DF)")

    # # NODE (TF)
    plot!(plt, t_train, sol_tf[1, 1, :], c=:coral, linestyle=:dot,label=L"NODE_1")
    plot!(plt, t_train, sol_tf[2, 1, :], c=:green, linestyle=:dot,label=L"NODE_2")
    plot!(plt, t_train, sol_tf[3, 1, :], c=:blue, linestyle=:dot,label=L"NODE_3")

    # plot!(plt, t_train, sol_tf[1, 1, :] .- Φ[:, 1]' * sm, c=:coral, linestyle=:dash,label=L"NODE_1")
    # plot!(plt, t_train, sol_tf[2, 1, :] .- Φ[:, 2]' * sm, c=:green, linestyle=:dash,label=L"NODE_2")
    # plot!(plt, t_train, sol_tf[3, 1, :] .- Φ[:, 3]' * sm, c=:blue, linestyle=:dash,label=L"NODE_3")

    # plot!(plt, t_train, a_train[1, 1, :], c=:red, linestyle=:dot,label=L"NODE_1")
    # plot!(plt, t_train, a_train[2, 1, :], c=:red, linestyle=:dot,label=L"NODE_2")
    # plot!(plt, t_train, a_train[3, 1, :], c=:red, linestyle=:dot,label=L"NODE_3")

    # plot!(plt, t_train, sol_tf₀[1, 1, :], c=:coral, linestyle=:dot,label=L"NODE_1 Initial")
    # plot!(plt, t_train, sol_tf₀[2, 1, :], c=:green, linestyle=:dot,label=L"NODE_2 Initial")
    # plot!(plt, t_train, sol_tf₀[3, 1, :], c=:blue, linestyle=:dot,label=L"NODE_3 Initial")

end

plot()
plot!(x, Φ[:, 1])
plot!(x, Φ[:, 2])
plot!(x, Φ[:, 3])

iplot = 1
for (i, t) ∈ collect(enumerate(t_test))[1:8:end]
    pl = Plots.plot(;title = @sprintf("Projected field, t = %.3f", t), xlabel = "x")
    plot!(pl, x, u_test[:, iplot, i]; label = "FOM")
    plot!(pl, x, Φ * a_test[:, iplot, i] .+ sm; label = "B")
    # plot!(pl, x, Φ * A[:, i] .+ sm; label = "ROM GP")
    # plot!(pl, x, Φ * sol_tf[:, iplot, i] .+ sm; label = "ROM NODE (TF)")
    # plot!(pl, x, Φ * sol_tf[:, iplot, i] .+ sm; label = "ROM NODE (TF)")
    display(pl)
    sleep(0.05)
end

iplot = 1
for (i, t) ∈ enumerate(t_test)
    pl = Plots.plot(;title = @sprintf("Projected field, t = %.3f", t), xlabel = "x")
    plot!(pl, x, u_test[:, iplot, i]; label = "FOM")
    display(pl)
    sleep(0.02)
end


pl = Plots.plot(;title = "Projection", xlabel = "x", ylims = extrema(v_train[1, iplot, :]))
plot!(pl, t_train, v_train[1, iplot, :]; label = "Projected FOM")
plot!(pl, t_train, sol_tf[1, iplot, :]; label = "ROM neural closure (TF)")


pl = Plots.