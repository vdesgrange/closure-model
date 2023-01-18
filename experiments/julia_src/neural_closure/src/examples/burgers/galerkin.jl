using BenchmarkTools
using AbstractFFTs
using FFTW
using Statistics
using LaTeXStrings

include("closure_training.jl");
include("../../equations/equations.jl");

function fflux2(u, (ν, Δx), t)
    u₋ = circshift(u, 1)
    u₊ = circshift(u, -1)
    a₊ = u₊ + u
    a₋ = u + u₋
    du = @. -((a₊ < 0) * u₊^2 + (a₊ > 0) * u^2 - (a₋ < 0) * u^2 - (a₋ > 0) * u₋^2) / Δx + ν * (u₋ - 2u + u₊) / Δx^2
    # du = [
    #     zeros(1, size(du)[2:end]...);
    #     du[2:end-1, :, :];
    #     zeros(1, size(du)[2:end]...);
    # ]
    du
end
fflux2(u::AbstractMatrix, p, t) = reshape(fflux2(reshape(u, size(u,1), 1, size(u, 2)), p, t), size(u, 1), size(u, 2))
fflux2(u::AbstractVector, p, t) = reshape(fflux2(reshape(u, size(u,1), 1, 1), p, t), size(u, 1))

function downsampling(u, d)
    (Nₓ, b, Nₜ) = size(u);
    Dₜ = Int64(Nₜ / d);
    u₁ = u[:, :, 1:d:end]; # Take t value every d steps
    Dₓ = Int64(Nₓ / d);
    u₂ = sum(reshape(u₁, d, Dₓ, b, Dₜ), dims=1);
    u₃   = reshape(u₂, Dₓ, b, Dₜ) ./ d;
    return u₃
end

function downsampling(u, dₓ, dₜ)
    (Nₓ, b, Nₜ) = size(u);
    Dₜ = Int64(Nₜ / dₜ);
    u₁ = u[:, :, 1:dₜ:end]; # Take t value every dₜ steps
    Dₓ = Int64(Nₓ / dₓ);
    u₂ = sum(reshape(u₁, dₓ, Dₓ, b, Dₜ), dims=1);
    u₃ = reshape(u₂, Dₓ, b, Dₜ) ./ dₓ;
    return u₃
end

function create_data(f, p, K, x, nsolution, t; decay = k -> 1 / (1 + abs(k)), kwargs...)
    L = x[end]
    basis = [exp(2π * im * k * x / L) for x ∈ x, k ∈ -K:K]
    c = [randn() * decay(k) * exp(-2π * im * rand()) for k ∈ -K:K, _ ∈ 1:nsolution]
    u₀ = real.(basis * c)
    @show size(u₀)
    predict(f, u₀, p, t, Tsit5(); kwargs...)
end

# === Parameters ===
epochs = 100;
ratio = 0.75;
batch_size = 64; # High for derivative fitting (i.e. 64)
lr = 1e-3;
reg = 1e-7;
noise = 0.05;
tₘₐₓ= 1.; # 2.
tₘᵢₙ = 0.;
xₘₐₓ = pi; # pi
xₘᵢₙ = 0;
tₙ = 64;
xₙ = 64;
ν = 0.001; # Viscosity
K = 40;  # Maximum frequency in initial conditions
sol = Tsit5();

# FOM discretization
upₓ = 4;
x = LinRange(xₘᵢₙ, xₘₐₓ, upₓ * xₙ);
Δx = (xₘₐₓ - xₘᵢₙ) / (upₓ * xₙ - 1);
k = 2 * pi * AbstractFFTs.fftfreq(upₓ * xₙ, 1. / Δx);

# Training times
up = 16;
t_pod = LinRange(0.0, 0.5, Int(up * tₙ));
t_train = LinRange(0.0, 0.5, Int(up * tₙ));
t_valid = LinRange(0.0, 1., Int(up * tₙ));
t_test = LinRange(0.0, 2., Int(up * tₙ));

# === Generate data ===
ff = fflux2;
pᵣ =  (ν, Δx);

u_pod = create_data(ff, pᵣ, K, x, 32, t_pod; reltol = 1e-6, abstol = 1e-6);
u_train = create_data(ff, pᵣ, K, x, 128, t_train; reltol = 1e-6, abstol = 1e-6);
u_valid  = create_data(ff, pᵣ, K, x, 32, t_valid; reltol = 1e-6, abstol = 1e-6);
u_test  = create_data(ff, pᵣ, K, x, 32, t_test; reltol = 1e-6, abstol = 1e-6);
display(GraphicTools.show_state(downsampling(u_train, upₓ, up)[:, 1, :], t_pod[1:up:end], x[1:4:end], "", "t", "x"))

# Time derivatives
dudt_pod = ff(u_pod, pᵣ, 0);
dudt_train = ff(u_train, pᵣ, 0);
dudt_valid = ff(u_valid, pᵣ, 0);
dudt_test = ff(u_test, pᵣ, 0);

# Get POD basis
# Maulik 2020 : POD.generate_pod_basis(u, true); (f_fft) fflux + (T=2) + eigenproblem + substract mean (0.86 for fflux, 0.88 for fflux2)
# Gupta 2021 : POD.generate_pod_svd_basis(u, false); fflux2 + (T=2 or 4) + svd + substract mean
m = 30;  # Number of POD modes
# bas, _ = POD.generate_pod_svd_basis(reshape(downsampling(u_pod, upₓ, up), xₙ, :), false); # 64-by-64
bas, sm = POD.generate_pod_svd_basis(reshape(downsampling(u_pod, upₓ, 1), xₙ, :), false); # 64-by-1024
# bas, _ = POD.generate_pod_svd_basis(reshape(u_pod, upₓ * xₙ, :), false); # 256-by-1024
λ = bas.eigenvalues;
@show POD.get_energy(bas.eigenvalues.^2, m);
Φ = bas.modes[:, 1:m];
u₀ = downsampling(u_pod, upₓ, 1)[:, :, 1];
# v₀ = v_pod[:, 1, 1];

x2 = x[1:4:end];
# x2 = x;
begin
    plt = plot(title="POD bases", xlabel="x", ylabel=L"Φ_i", background_color_legend = RGBA(1, 1, 1, 0.8))
    plot!(plt, x2, Φ[:, 1], c=:coral, label=L"Φ_1")
    plot!(plt, x2, Φ[:, 2], c=:green, label=L"Φ_2")
    plot!(plt, x2, Φ[:, 3], c=:blue, label=L"Φ_3")
    # savefig("maulik_3_modes_substracted_mean.png")
end

# Get POD basis
#size(cat([u_pod, u_pod]...; dims=2))
display(GraphicTools.show_state(downsampling(u_train, upₓ, up)[:, 10, :], t_train[1:up:end], x2, "", "t", "x"))
display(GraphicTools.show_state(Φ * Φ' * downsampling(u_train, upₓ, up)[:, 10, :],  t_train, x, "", "t", "x"))

ff2 = Equations.f_fft;
gg(a, p, t) = Φ' * ff(Φ * a, p, t); # .+ sm
k = 2 * pi * AbstractFFTs.fftfreq(xₙ, 1. / Δx); # Careful on xₙ dimension
a₀_pod = Φ' * (u₀); # Reconstruction with Φ * a
# a₀_pod = Φ' * (u₀ .- sm);  # Reconstruction with Φ * a .+ sm
A = predict(gg,  a₀_pod, (ν, Δx, k), t_pod, Tsit5());
reconstruction =  tensormul(Φ, A); # sm .+
down_reconstruction = downsampling(reconstruction, 1, up);

display(GraphicTools.show_state(downsampling(u_pod, upₓ, up)[:, 10, :], t_pod[1:up:end], x2, "", "t", "x"))
display(GraphicTools.show_state(Φ * Φ' * downsampling(u_pod, upₓ, up)[:, 10, :],  t_pod, x, "", "t", "x"))
display(GraphicTools.show_state(down_reconstruction[:, 10, :], x2, t_pod[1:up:end], "", "x", "t"))

# POD coefficients
# a_pod = tensormul(Φ', downsampling(u_pod, upₓ, 1) .- sm); # v_pod = Φ' * u_pod ### A = Φ' * (u_pod .- mean(u_pod, dims=2));
# a_train = tensormul(Φ',  downsampling(u_train, upₓ, 1) .- sm);
# a_valid = tensormul(Φ',  downsampling(u_valid, upₓ, 1) .- sm);
# a_test = tensormul(Φ', downsampling(u_test, upₓ, 1) .- sm);

a_pod = tensormul(Φ',  downsampling(u_pod, upₓ, 1)); # v_pod = Φ' * u_pod ### A = Φ' * (u_pod .- mean(u_pod, dims=2));
a_train = tensormul(Φ',  downsampling(u_train, upₓ, 1));
a_valid = tensormul(Φ',  downsampling(u_valid, upₓ, 1));
a_test = tensormul(Φ',  downsampling(u_test, upₓ, 1));

# Time derivatives of POD coefficients
size(dudt_pod)
dadt_pod = tensormul(Φ',  downsampling(dudt_pod, upₓ, 1));
dadt_train = tensormul(Φ', downsampling(dudt_train, upₓ, 1));
dadt_valid = tensormul(Φ', downsampling(dudt_valid, upₓ, 1));
dadt_test = tensormul(Φ', downsampling(dudt_test, upₓ, 1));

# size(dadt_train)
# size(a_pod)
# size(bas.coefficients)
# size(t_pod)
# size(A)
iplot = 1;
begin
    plt = plot(title="Coefficients", xlabel="t", ylabel="ϕ", background_color_legend = RGBA(1, 1, 1, 0.8))
    plot!(plt; dpi=600)

    plot!(plt, t_pod, a_pod[1, iplot, :], c=:coral, label=L"a_1") # Reduce coefficient a1 = Φᵣ'û (mean value is substracted !)
    plot!(plt, t_pod, a_pod[2, iplot, :], c=:green,  label=L"a_2") # Reduce coefficient a2 = Φᵣ'û
    plot!(plt, t_pod, a_pod[3, iplot, :], c=:blue, label=L"a_3") # Reduce coefficient a3 = Φᵣ'û

    plot!(plt, t_pod, A[1, iplot, :], c=:coral, linestyle=:dash,label=L"GP_1")
    plot!(plt, t_pod, A[2, iplot, :], c=:green, linestyle=:dash,label=L"GP_2")
    plot!(plt, t_pod, A[3, iplot, :], c=:blue, linestyle=:dash,label=L"GP_3")
end

# === Models ===

## === Neural network ===
# model = Models.CNN2(9, [2, 4, 8, 8, 4, 2, 1]);
model = Flux.Chain(
    Flux.Dense(m => Int(m / 2), tanh; init = Models.glorot_uniform_float64, bias = true),
    Flux.Dense(Int(m / 2) => Int(m/ 2), tanh; init = Models.glorot_uniform_float64, bias = true),
    Flux.Dense(m => m, identity; init = Models.glorot_uniform_float64, bias = true),
);
# model = Models.CNN2(9, [2, 4, 8, 8, 4, 2, 1]);
ff = fflux2;
p₀, re = Flux.destructure(model);
fᵣₒₘ(v, p, t) = re(p)(v);
f_closure(v, p, t) = Φ' * ff(Φ * v, (ν, Δx), t) + re(p)(v); # ν, Δx, k
#f_ϵ(u, p, t) = Φ * f_closure(Φ' * u, p, t) + fᵣₒₘ(u, p, t);

# === Training methods ===

# Derivative fitting
loss_df(p, _) = loss_derivative_fit(
    fᵣₒₘ, # fᵣₒₘ,
    p,
    reshape(dadt_train[:, 1:32, :], m, :),
    reshape(a_train[:, 1:32, :], m, :);
    nsample = 8,
);

p_df = train(
    loss_df,
    p₀;
    maxiters = 2000,
    optimizer = OptimizationOptimisers.Adam(lr),
    callback = create_callback(
        fᵣₒₘ,
        a_valid,
        t_valid;
        ncallback = 8,
        reltol = 1e-6,
        abstol = 1e-6,
    ),
);

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
    st = Zygote.@ignore rand(1:(length(t) - ntime))
    it = Zygote.@ignore st:(st + ntime)

    v = v[:, is, it]
    t = t[it]

    # Predicted solution
    sol = predict(f, v[:, :, 1], p, t, solver; kwargs...)

    # Relative squared error
    # data = sum(abs2, sol - v) / sum(abs2, v)
    data = sqrt(mean((sol .- v) .^ 2));
    

    # Regularization term
    reg = sum(abs2, p) / length(p)

    # Loss
    data + λ * reg
end

size(a_train)

# loss_time_trajectory_fit
loss_tf(p, _) = loss_time_trajectory_fit( 
    fᵣₒₘ, # fᵣₒₘ
    p,
    a_train[:, 1:128, :],
    t_train;
    nsolution = 32,
    ntime = 10,
    reltol = 1e-6,
    abstol = 1e-6,
);

# trajectory fitting
loss_tf(p, _) = loss_trajectory_fit(
    fᵣₒₘ,
    p,
    a_train[:, 1:128, :],
    t_train;
    nsolution = 32,
    ntime = 10,
    reltol = 1e-6,
    abstol = 1e-6,
);

callback = create_callback(
    fᵣₒₘ, # fᵣₒₘ
    a_valid,
    t_valid;
    ncallback = 4,
    reltol = 1e-4,
    abstol = 1e-6,
);

p_tf = train(
    loss_tf,
    p₀;
    maxiters = 1000,
    optimizer = OptimizationOptimisers.Adam(lr),
    callback = callback,
);

# Compare models

sol_nomodel = predict(
    g, 
    a_test[:, :, 1], 
    nothing, 
    t_test, 
    Tsit5(); 
    reltol = 1e-6, 
    abstol = 1e-6
);

sol_df = predict(
    fᵣₒₘ,
    a_valid[:, :, 1],
    p_df[1],
    t_valid,
    Tsit5();
    reltol = 1e-6,
    abstol = 1e-6,
);

sol_tf = predict(
    fᵣₒₘ,
    a_valid[:, :, 1],
    p_tf[1],
    t_valid,
    Tsit5();
    reltol = 1e-6,
    abstol = 1e-6,
);

iplot = 1;
begin
    plt = plot(title="POD-space coefficients", xlabel="t", ylabel=L"a_i", background_color_legend = RGBA(1, 1, 1, 0.8))
    plot!(plt; dpi=600)
    
    # Reference
    plot!(plt, t_pod, a_pod[1, iplot, :], c=:coral, label=L"a_1") # Reduce coefficient a1 = Φᵣ'û (mean value is substracted !)
    plot!(plt, t_pod, a_pod[2, iplot, :], c=:green,  label=L"a_2") # Reduce coefficient a2 = Φᵣ'û
    plot!(plt, t_pod, a_pod[3, iplot, :], c=:blue, label=L"a_3") # Reduce coefficient a3 = Φᵣ'û
    plot!(plt, t_pod, a_pod[4, iplot, :], c=:blue, label=L"a_3") # Reduce coefficient a3 = Φᵣ'û
    plot!(plt, t_pod, a_pod[5, iplot, :], c=:blue, label=L"a_3") # Reduce coefficient a3 = Φᵣ'û

    # Baseline
    plot!(plt, t_valid, A[1, iplot, :], c=:coral, linestyle=:dash,label=L"GP_1")
    plot!(plt, t_valid, A[2, iplot, :], c=:green, linestyle=:dash,label=L"GP_2")
    plot!(plt, t_valid, A[3, iplot, :], c=:blue, linestyle=:dash,label=L"GP_3")
    plot!(plt, t_valid, A[4, iplot, :], c=:blue, linestyle=:dash,label=L"GP_3")
    plot!(plt, t_valid, A[5, iplot, :], c=:blue, linestyle=:dash,label=L"GP_3")

    # NODE (DF)
    # plot!(plt, t_valid, sol_df[1, 1, :], c=:coral, line=(3, :dash),label=L"NODE_1\ (DF)")
    # plot!(plt, t_valid, sol_df[2, 1, :], c=:green, line=(3, :dash),label=L"NODE_2\ (DF)")
    # plot!(plt, t_valid, sol_df[3, 1, :], c=:blue, line=(3, :dash),label=L"NODE_3\ (DF)")

    # # NODE (TF)
    # plot!(plt, t_train, sol_tf[1, 1, :], c=:coral, linestyle=:dot,label=L"NODE_1")
    # plot!(plt, t_train, sol_tf[2, 1, :], c=:green, linestyle=:dot,label=L"NODE_2")
    # plot!(plt, t_train, sol_tf[3, 1, :], c=:blue, linestyle=:dot,label=L"NODE_3")

    # plot!(plt, t_valid, sol_tf[1, 1, :], c=:coral, linestyle=:dot,label=L"NODE_1\ (TF)")
    # plot!(plt, t_valid, sol_tf[2, 1, :], c=:green, linestyle=:dot,label=L"NODE_2\ (TF)")
    # plot!(plt, t_valid, sol_tf[3, 1, :], c=:blue, linestyle=:dot,label=L"NODE_3\ (TF)")
end


for (i, t) ∈ collect(enumerate(t_train))[1:8:end]
    pl = Plots.plot(;title = @sprintf("Solution, t = %.3f", t), xlabel = "x", ylims = extrema(Φ * a_train[:, iplot, :]))
    # plot!(pl, x, u_valid[:, iplot, i]; label = "FOM")
    plot!(pl, x2, Φ * a_train[:, iplot, i]; label = "Projected FOM")
    # plot!(pl, x, Φ * sol_nomodel[:, iplot, i]; label = "ROM no closure")
    plot!(pl, x2, Φ * sol_df[:, iplot, i]; label = "ROM neural closure (DF)")
    plot!(pl, x2, Φ * sol_tf[:, iplot, i]; label = "ROM neural closure (TF)")
    display(pl)
    sleep(0.05)
end


# === Check results ===
t, u₀, u = Generator.get_burgers_batch(tₘₐₓ, tₘᵢₙ, xₘₐₓ, xₘᵢₙ, tₙ, xₙ, ν, 2, (; m));
size(u)
v₀ =  Φ' * u[:, 1];

û = galerkin_projection(t, u, Φ, ν, Δx, Δt);
û_prob = ODEProblem((v, p, t) ->  (Φ' * f(Φ * v, p, t)), v₀, extrema(t), (ν, Δx), saveat=t);
ū = Φ * Array(solve(û_prob, sol, sensealg=DiffEqSensitivity.InterpolatingAdjoint(; autojacvec=ZygoteVJP())));
uₙₙ_prob = ODEProblem((v, p, t) -> K2(v), v₀, extrema(t), θ2; saveat=t); reshape(v₀, (size(v₀, 1), :));
uₙₙ = Φ * Array(solve(uₙₙ_prob, Tsit5()));
uᵧₙₙ_prob = ODEProblem(f_closure, v₀, extrema(t), θ; saveat=t);
uᵧₙₙ = Φ * Array(solve(uᵧₙₙ_prob, Tsit5()));
for (i, t) ∈ enumerate(t)
  pl = Plots.plot(; xlabel = "x", ylim=extrema(u))
  Plots.plot!(pl, x, u[:, i]; label = "FOM - Model 0")
  Plots.plot!(pl, x, Φ * Φ' * u[:, i]; label = "FOM - Model 0.5 - ΦΦ'u")
  Plots.plot!(pl, x, û[:, i]; label = "ROM - Model 1.0 GP")
  Plots.plot!(pl, x, ū[:, i]; label = "ROM - Model 1.5 Φ'f(Φv)")
  Plots.plot!(pl, x, uₙₙ[:, i]; label = "ROM - Model 2 NN")
  Plots.plot!(pl, x, uᵧₙₙ[:, i]; label = "ROM - Model 3  Φ'f(Φv) + NN(v)")
  # Plots.plot!(pl, x, uₙₙ[:, i]; label = "ROM - Model 4 GP + NN")
  display(pl)
  # sleep(0.02)
end

display(GraphicTools.show_state(u, t, x, "", "t", "x"))
display(GraphicTools.show_state(Φ * Φ' * u, t, x, "", "t", "x"))
display(GraphicTools.show_state(ū, t, x, "", "t", "x"))
display(GraphicTools.show_state(uᵧₙₙ, t, x, "", "t", "x"))

display(GraphicTools.show_err(Φ * Φ' * u, ū, t, x, "", "t", "x"))
display(GraphicTools.show_err(Φ * Φ' * u, uᵧₙₙ, t, x, "", "t", "x"))
display(GraphicTools.show_err(Φ * Φ' * u, uₙₙ, t, x, "", "t", "x"))
