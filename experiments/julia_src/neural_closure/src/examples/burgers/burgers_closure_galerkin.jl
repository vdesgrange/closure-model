using BSON: @save
using Flux
using IterTools: ncycle
using LinearAlgebra
using Optimization
using OptimizationOptimisers
using OrdinaryDiffEq
using Plots
using Printf
using Random
using SciMLSensitivity
using Zygote

include("../../utils/processing_tools.jl")
include("../../neural_ode/models.jl")
include("../../neural_ode/regularization.jl")
include("../../utils/generators.jl");
include("../../utils/graphic_tools.jl")
include("../../equations/burgers/burgers_gp.jl")
include("../../rom/pod.jl")

function get_Φ(u, m)
    U, S, V = svd(u)
    Φ = U[:, 1:m]
end

"""
    predict(f, v₀, p, t, solver; kwargs...)

Predict solution given parameters `p`.
"""
function predict(f, v₀, p, t, solver; kwargs...)
    problem = ODEProblem(f, v₀, extrema(t), p)
    sol = solve(problem, solver; saveat = t, kwargs...)
    Array(sol)
end

"""
    loss_trajectory_fit(
        f, p, u, t, solver, λ = 1e-8;
        nsolution = size(u, 2),
        ntime = size(u, 3),
        kwargs...,
    )

Compute trajectory-fitting loss.
Selects a random subset of solutions and time points at each evaluation.
Further `kwargs` are passed to the ODE solver.
"""
function loss_trajectory_fit(
    f,
    p,
    v,
    t;
    solver = Tsit5(),
    λ = 1e-8,
    nsolution = size(v, 2),
    ntime = size(v, 3),
    kwargs...,
)
    # Select a random subset (batch)
    is = Zygote.@ignore sort(shuffle(1:size(v, 2))[1:nsolution])
    it = Zygote.@ignore sort(shuffle(1:length(t))[1:ntime])
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

"""
    loss_derivative_fit(f, p, dvdt, v, λ; nsample = size(v, 2))

Compute derivative-fitting loss.
Selects a random subset of samples at each evaluation.
"""
function loss_derivative_fit(f, p, dvdt, v; λ = 1e-8, nsample = size(v, 2))
    # Select a random subset (batch)
    i = Zygote.@ignore sort(shuffle(1:size(v, 2))[1:nsample])
    v = v[:, i]
    dvdt = dvdt[:, i]

    # Predicted right hand side
    rhs = f(v, p, 0)

    # Relative squared error
    data = sum(abs2, rhs - dvdt) / sum(abs2, dvdt)

    # Regularization term
    reg = sum(abs2, p) / length(p)

    # Loss
    data + λ * reg
end

function relative_error(f, p, v, t, solver; kwargs...)
    # Predicted soluton
    sol = predict(f, v[:, :, 1], p, t, solver; kwargs...)

    # Relative error
    norm(sol - v) / norm(v)
end

function train(
    loss,
    p₀;
    maxiters = 100,
    optimizer = OptimizationOptimisers.Adam(0.001),
    callback = p -> false,
)
    func = OptimizationFunction(loss, Optimization.AutoZygote())
    problem = OptimizationProblem(func, p₀, nothing)
    sol = solve(problem, optimizer; maxiters, callback)
    sol.u
end

function create_callback(f, v, t; ncallback = 1, solver = Tsit5(), kwargs...)
    i = 0
    iters = Int[]
    errors = zeros(0)
    function callback(p, loss)
        i += 1
        if i % ncallback == 0
            e = relative_error(f, p, v, t, solver; kwargs...)
            push!(iters, i)
            push!(errors, e)
            println("Epoch $i\trelative error $e")
            display(plot(iters, errors; xlabel = "Iterations", title = "Relative error"))
        end
        return false
    end
end

"""
    create_data(f, p, K, x, nsolution, t; decay = k -> 1 / (1 + abs(k)), kwargs...)

Args:

- Right hand side `f(u, p, t)`
- Parameters `p`
- Maximum frequency in initial conditions `K`
- Spatial points `x = LinRange(0, L, N + 1)[2:end]`
- Number of different initial conditions `nsolution`
- Time points to save `t = LinRange(0, T, ntime)`

Kwargs:

- Frequency decay function (for initial conditions) `decay(k)`
- Other kwargs: Pass to ODE solver (e.g. `reltol = 1e-4`, `abstol = 1e-6`)

Returns:

- Solution `u` of size `(nx, nsolution, ntime)`

To get initial conditions, do `u₀ = u[:, :, 1]`.
"""
function create_data(f, p, K, x, nsolution, t; decay = k -> 1 / (1 + abs(k)), kwargs...)
    # Domain length (assume x = LinRange(0, L, N + 1)[2:end]) for some `L` and `N`)
    L = x[end]

    # Fourier basis
    basis = [exp(2π * im * k * x / L) for x ∈ x, k ∈ -K:K]

    # Fourier coefficients with random phase and amplitude
    c = [randn() * decay(k) * exp(-2π * im * rand()) for k ∈ -K:K, _ ∈ 1:nsolution]

    # Random initial conditions (real-valued)
    u₀ = real.(basis * c)

    # Solve ODE
    predict(f, u₀, p, t, Tsit5(); kwargs...)
end

"""
    tensormul(A, x)

Multiply `A` of size `(ny, nx)` with each column of tensor `x` of size `(nx, d1, ..., dn)`.
Return `y = A * x` of size `(ny, d1, ..., dn)`.
"""
function tensormul(A, x)
    s = size(x)
    x = reshape(x, s[1], :)
    y = A * x
    reshape(y, size(y, 1), s[2:end]...)
end

function f(u, (ν, Δx), t)
    u₋ = circshift(u, 1)
    u₊ = circshift(u, -1)
    a₊ = u₊ + u
    a₋ = u + u₋
    du = @. -((a₊ < 0) * u₊^2 + (a₊ > 0) * u^2 - (a₋ < 0) * u^2 - (a₋ > 0) * u₋^2) / Δx +
       ν * (u₋ - 2u + u₊) / Δx^2
    du
end

# Viscosity
ν = 0.001;

# FOM discretization
N = 200
x = LinRange(0.0, 1.0, N + 1)[2:end]
Δx = 1 / N

# Training times
t_pod = LinRange(0.0, 0.5, 60)
t_train = LinRange(0.0, 0.1, 60)
t_valid = LinRange(0.0, 0.5, 10)
t_test = LinRange(0.0, 1.0, 30)

# Maximum frequency in initial conditions
K = 100

# Solutions
u_pod = create_data(f, (ν, Δx), K, x, 200, t_pod; reltol = 1e-4, abstol = 1e-6)
u_train = create_data(f, (ν, Δx), K, x, 500, t_train; reltol = 1e-4, abstol = 1e-6)
u_valid  = create_data(f, (ν, Δx), K, x, 20, t_valid; reltol = 1e-4, abstol = 1e-6)
u_test  = create_data(f, (ν, Δx), K, x, 50, t_test; reltol = 1e-4, abstol = 1e-6)

# Time derivatives
dudt_pod = f(u_pod, (ν, Δx), 0)
dudt_train = f(u_train, (ν, Δx), 0)
dudt_valid = f(u_valid, (ν, Δx), 0)
dudt_test = f(u_test, (ν, Δx), 0)

# Number of POD modes
m = 10;

# Create POD basis
Φ = get_Φ(reshape(u_pod, N, :), m)
plot(Φ[:, 1:3])

# POD coefficients
v_pod = tensormul(Φ', u_pod)
v_train = tensormul(Φ', u_train)
v_valid = tensormul(Φ', u_valid)
v_test = tensormul(Φ', u_test)

# Time derivatives of POD coefficients
dvdt_pod = tensormul(Φ', dudt_pod)
dvdt_train = tensormul(Φ', dudt_train)
dvdt_valid = tensormul(Φ', dudt_valid)
dvdt_test = tensormul(Φ', dudt_test)

# Reduced order model
g(v, p, t) = Φ' * f(Φ * v, (ν, Δx), t)

## Neural network
# model = Models.CNN2(9, [2, 4, 8, 8, 4, 2, 1]);
model = Flux.Chain(
    v -> vcat(v, v .^ 2),
    Flux.Dense(2m => 2m, tanh; init = Models.glorot_uniform_float64, bias = true),
    Flux.Dense(2m => m, tanh; init = Models.glorot_uniform_float64, bias = true),
    Flux.Dense(m => m, identity; init = Models.glorot_uniform_float64, bias = true),
)
p₀, re = Flux.destructure(model)
fᵣₒₘ(v, p, t) = re(p)(v)

function f_closure(v, p, t)
    g(v, nothing, t) + fᵣₒₘ(v, p, t)
end

# Derivative fitting
loss_df(p, _) = loss_derivative_fit(
    f_closure,
    p,
    reshape(dvdt_train, m, :),
    reshape(v_train, m, :);
    nsample = 100,
)
p_df = train(
    loss_df,
    p₀;
    maxiters = 5000,
    callback = create_callback(
        f_closure,
        v_valid,
        t_valid;
        ncallback = 100,
        reltol = 1e-4,
        abstol = 1e-6,
    ),
);

loss_tf(p, _) = loss_trajectory_fit(
    f_closure,
    p,
    v_train,
    t_train;
    nsolution = 10,
    ntime = 20,
    reltol = 1e-4,
    abstol = 1e-6,
)

p_tf = train(
    loss_tf,
    p₀;
    maxiters = 100,
    callback = create_callback(
        f_closure,
        v_valid,
        t_valid;
        ncallback = 10,
        reltol = 1e-4,
        abstol = 1e-6,
    ),
);

# Compare models
sol_nomodel =
    predict(g, v_test[:, :, 1], nothing, t_test, Tsit5(); reltol = 1e-4, abstol = 1e-6)
sol_df = predict(
    f_closure,
    v_test[:, :, 1],
    p_df,
    t_test,
    Tsit5();
    reltol = 1e-4,
    abstol = 1e-6,
)
sol_tf = predict(
    f_closure,
    v_test[:, :, 1],
    p_tf,
    t_test,
    Tsit5();
    reltol = 1e-4,
    abstol = 1e-6,
)

iplot = 1
for (i, t) ∈ enumerate(t_test)
    pl = Plots.plot(;
        title = @sprintf("Solution, t = %.3f", t),
        xlabel = "x",
        ylims = extrema(Φ * v_test[:, iplot, :]),
    )
    plot!(pl, x, Φ * v_test[:, iplot, i]; label = "Projected FOM")
    plot!(pl, x, Φ * sol_nomodel[:, iplot, i]; label = "ROM no closure")
    plot!(pl, x, Φ * sol_df[:, iplot, i]; label = "ROM neural closure (DF)")
    plot!(pl, x, Φ * sol_tf[:, iplot, i]; label = "ROM neural closure (TF)")
    display(pl)
    sleep(0.05)
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
