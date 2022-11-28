include("burgers_closure_galerkin_2.jl")

# === Parameters ===
epochs = 100;
ratio = 0.75;
batch_size = 64; # High for derivative fitting (i.e. 64)
lr = 1e-3;
reg = 1e-7;
noise = 0.05;
tₘₐₓ= 0.5; # 2.
tₘᵢₙ = 0.;
xₘₐₓ = 1.; # pi
xₘᵢₙ = 0;
tₙ = 128;
xₙ = 64;
ν = 0.001; # Viscosity
Δt = (tₘₐₓ - tₘᵢₙ) / (tₙ - 1);
m = 20;  # Number of POD modes
k = 100;  # Maximum frequency in initial conditions
sol = Tsit5();

# FOM discretization
x = LinRange(xₘᵢₙ, xₘₐₓ, xₙ);
Δx = (xₘₐₓ - xₘᵢₙ) / (xₙ - 1);

# Training times
t_pod = LinRange(0.0, tₘₐₓ / 2, Int(tₙ / 2));
t_train = LinRange(0.0, tₘₐₓ / 10, Int(tₙ / 2));
t_valid = LinRange(0.0, tₘₐₓ / 2, Int(tₙ / 2));
t_test = LinRange(0.0, tₘₐₓ, Int(tₙ));


# === Generate data ===
u_pod = Generators.create_data(fflux, (ν, Δx), K, x, 200, t_pod; reltol = 1e-4, abstol = 1e-6);
u_train = Generators.create_data(fflux, (ν, Δx), K, x, 500, t_train; reltol = 1e-4, abstol = 1e-6);
u_valid  = Generators.create_data(fflux, (ν, Δx), K, x, 20, t_valid; reltol = 1e-4, abstol = 1e-6);
u_test  = Generators.create_data(fflux, (ν, Δx), K, x, 50, t_test; reltol = 1e-4, abstol = 1e-6);
# display(GraphicTools.show_state(u_test[:, 1, :], t_test, x, "", "t", "x"))

# Time derivatives
dudt_pod = fflux(u_pod, (ν, Δx), 0);
dudt_train = fflux(u_train, (ν, Δx), 0);
dudt_valid = fflux(u_valid, (ν, Δx), 0);
dudt_test = fflux(u_test, (ν, Δx), 0);

# Get POD basis
Φ = get_Φ(reshape(u_pod, N, :), m);
# display(GraphicTools.show_state(u_test[:, 1, :], t, x, "", "t", "x"))
# display(GraphicTools.show_state(Φ * Φ' * u_test[:, 1, :], t, x, "", "t", "x"))

# plot(Φ[:, 1:3])

# POD coefficients
v_pod = tensormul(Φ', u_pod);
v_train = tensormul(Φ', u_train);
v_valid = tensormul(Φ', u_valid);
v_test = tensormul(Φ', u_test);

# Time derivatives of POD coefficients
dvdt_pod = tensormul(Φ', dudt_pod);
dvdt_train = tensormul(Φ', dudt_train);
dvdt_valid = tensormul(Φ', dudt_valid);
dvdt_test = tensormul(Φ', dudt_test);

# === Models ===

## === Neural network ===
# model = Models.CNN2(9, [2, 4, 8, 8, 4, 2, 1]);
model = Flux.Chain(
    v -> vcat(v, v .^ 2),
    Flux.Dense(2m => 2m, tanh; init = Models.glorot_uniform_float64, bias = true),
    Flux.Dense(2m => m, tanh; init = Models.glorot_uniform_float64, bias = true),
    Flux.Dense(m => m, identity; init = Models.glorot_uniform_float64, bias = true),
);
p₀, re = Flux.destructure(model);
fᵣₒₘ(v, p, t) = re(p)(v);


## === Reduced order model ===
g(v, p, t) = Φ' * f(Φ * v, (ν, Δx), t);

## === Closure model ===
f_closure(v, p, t) =  g(v, nothing, t) + fᵣₒₘ(v, p, t);
#f_ϵ(u, p, t) = Φ * f_closure(Φ' * u, p, t) + fᵣₒₘ(u, p, t);

# === Training methods ===

# Derivative fitting
loss_df(p, _) = loss_derivative_fit(
    f_closure, # fᵣₒₘ,
    p,
    reshape(dvdt_train, m, :),
    reshape(v_train, m, :);
    nsample = batch_size,
);

p_df = train(
    loss_df,
    p₀;
    maxiters = 5000,
    optimizer = OptimizationOptimisers.Adam(lr),
    callback = create_callback(
        f_closure,
        v_valid,
        t_valid;
        ncallback = batch_size,
        reltol = 1e-6,
        abstol = 1e-6,
    ),
);

# trajectory fitting
loss_tf(p, _) = loss_trajectory_fit(
    f_closure,
    p,
    v_train,
    t_train;
    nsolution = 16,
    ntime = 20,
    reltol = 1e-6,
    abstol = 1e-6,
);

p_tf = train(
    loss_tf,
    p₀;
    maxiters = 100,
    optimizer = OptimizationOptimisers.Adam(lr),
    callback = create_callback(
        f_closure,
        v_valid,
        t_valid;
        ncallback = 16,
        reltol = 1e-4,
        abstol = 1e-6,
    ),
);

# Compare models
size(v_test[:, :, 1])
sol_nomodel = predict(g, v_test[:, :, 1], nothing, t_test, Tsit5(); reltol = 1e-6, abstol = 1e-6);
sol_df = predict(
    f_closure,
    v_test[:, :, 1],
    p_df,
    t_test,
    Tsit5();
    reltol = 1e-6,
    abstol = 1e-6,
);
sol_tf = predict(
    f_closure,
    v_test[:, :, 1],
    p_tf,
    t_test,
    Tsit5();
    reltol = 1e-6,
    abstol = 1e-6,
);

iplot = 1;
size(u_test)
for (i, t) ∈ enumerate(t_test)
    pl = Plots.plot(;title = @sprintf("Solution, t = %.3f", t), xlabel = "x", ylims = extrema(Φ * v_test[:, iplot, :]))
    plot!(pl, x, u_test[:, iplot, i]; label = "FOM")
    plot!(pl, x, Φ * v_test[:, iplot, i]; label = "Projected FOM")
    plot!(pl, x, Φ * sol_nomodel[:, iplot, i]; label = "ROM no closure")
    plot!(pl, x, Φ * sol_df[:, iplot, i]; label = "ROM neural closure (DF)")
    plot!(pl, x, Φ * sol_tf[:, iplot, i]; label = "ROM neural closure (TF)")
    display(pl)
    # sleep(0.05)
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
