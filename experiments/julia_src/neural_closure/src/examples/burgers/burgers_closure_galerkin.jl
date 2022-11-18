using BSON: @save
using Flux
using IterTools: ncycle
using LinearAlgebra
using Optimization
using OptimizationOptimisers
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

dataset = Generator.read_dataset(
    "./dataset/viscous_burgers_high_dim_t1_64_x1_64_nu0.001_typ2_m100_256_up2_j173.jld2",
)["training_set"];
Φ_dataset, train_dataset = dataset[1:128], dataset[129:end];

xₘₐₓ = 1.0; # pi
xₘᵢₙ = 0;
xₙ = 64;
ν = 0.001; # 0.04
Δx = (xₘₐₓ - xₘᵢₙ) / (xₙ - 1)
x = LinRange(xₘᵢₙ, xₘₐₓ, xₙ);
m = 10;

# === Get basis Φ ===
Φ = get_Φ(Φ_dataset, m);

g(v, p, t) = Φ' * f(Φ * v, (ν, Δx), t)

# === Train ===
# model = Models.CNN2(9, [2, 4, 8, 8, 4, 2, 1]);
model = Flux.Chain(
    v -> vcat(v, v .^ 2),
    Flux.Dense(2m => 2m, tanh; init = Models.glorot_uniform_float64, bias = true),
    Flux.Dense(2m => m, tanh; init = Models.glorot_uniform_float64, bias = true),
    Flux.Dense(m => m, identity; init = Models.glorot_uniform_float64, bias = true),
)

@info("Building model")
p₀, re = Flux.destructure(model)
fᵣₒₘ(v, p, t) = re(p)(v)

function f_closure(v, p, t)
    g(v, nothing, t) + fᵣₒₘ(v, p, t)
end

# Assume that all times are the same
@assert all(d -> d[1] == train_dataset[1][1], train_dataset)
t_train = train_dataset[1][1]
u_train = reduce(hcat, (reshape(d[2], xₙ, 1, :) for d ∈ train_dataset))
dudt_train = f(u_train, (ν, Δx), 0)

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

# Reference coefficients
v_train = tensormul(Φ', u_train)
dvdt_train = tensormul(Φ', dudt_train)

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
        v_train[:, 1:20, :],
        t_train;
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
        v_train[:, 1:20, :],
        t_train;
        ncallback = 10,
        reltol = 1e-4,
        abstol = 1e-6,
    ),
);


# === Check results ===

sol_nomodel =
    predict(g, v_train[:, :, 1], nothing, t_train, Tsit5(); reltol = 1e-4, abstol = 1e-6)
sol_df = predict(
    f_closure,
    v_train[:, :, 1],
    p_df,
    t_train,
    Tsit5();
    reltol = 1e-4,
    abstol = 1e-6,
)
sol_tf = predict(
    f_closure,
    v_train[:, :, 1],
    p_tf,
    t_train,
    Tsit5();
    reltol = 1e-4,
    abstol = 1e-6,
)

iplot = 1
for (i, t) ∈ enumerate(t_train)
    pl = Plots.plot(;
        title = @sprintf("Solution, t = %.3f", t),
        xlabel = "x",
        ylim = extrema(v_train[:, iplot, :]),
    )
    plot!(pl, x, Φ * v_train[:, iplot, i]; label = "Projected FOM")
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
