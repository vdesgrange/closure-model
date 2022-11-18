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
    t,
    solver,
    λ = 1e-8;
    nsolution = size(v, 2),
    ntime = size(v, 3),
    kwargs...,
)
    # Select a random subset (batch)
    is = Zygote.@ignore sort(shuffle(1:size(v, 2))[1:nsolution])
    it = Zygote.@ignore sort(shuffle(1:length(t))[1:ntime])
    v = v[:, is, it]

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
function loss_derivative_fit(f, p, dvdt, v, λ; nsample = size(v, 2))
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

function train_derivative_fit(
    f,
    dvdt,
    v,
    p₀;
    maxiters = 100,
    optimizer = OptimizationOptimisers.Adam(0.001),
    solver = Tsit5(),
    callback = p -> false,
    λ = 1e-8,
    nsample = size(v, 2),
)
    func = OptimizationFunction(
        (p, _) -> loss_derivative_fit(f, p, dvdt, v, λ; nsample),
        Optimization.AutoZygote(),
    )
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

maxiters = 10;
λ = 1e-7;
xₘₐₓ = 1.0; # pi
xₘᵢₙ = 0;
xₙ = 64;
ν = 0.001; # 0.04
Δx = (xₘₐₓ - xₘᵢₙ) / (xₙ - 1)
solver = Tsit5();
x = LinRange(xₘᵢₙ, xₘₐₓ, xₙ);
m = 10;

# === Get basis Φ ===
Φ = get_Φ(Φ_dataset, m);

# === Train ===
# model = Models.CNN2(9, [2, 4, 8, 8, 4, 2, 1]);
model = Flux.Chain(
    v -> vcat(v, v .^ 2),
    Flux.Dense(2m => 2m, tanh; init = Models.glorot_uniform_float64, bias = true),
    Flux.Dense(2m => m, tanh; init = Models.glorot_uniform_float64, bias = true),
    Flux.Dense(m => m, identity; init = Models.glorot_uniform_float64, bias = true),
)
model.layers[2].weight
model.layers[2].bias

@info("Building model")
p₀, re = Flux.destructure(model)
fᵣₒₘ(v, p, t) = re(p)(v)

function f_closure(v, p, t)
    Φ' * f(Φ * v, (ν, Δx), t) + fᵣₒₘ(v, p, t)
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

predict(f_closure, v_train[:, :, 1], p₀, t_train, Tsit5())
create_callback(f_closure, v_train[:, 1:20, :], t_train; reltol = 1e-4, abstol = 1e-6)(p₀, nothing)

p = train_derivative_fit(
    f_closure,
    reshape(dvdt_train, m, :),
    reshape(v_train, m, :),
    p₀;
    maxiters = 1000,
    callback = create_callback(f_closure, v_train[:, 1:20, :], t_train; ncallback = 100, reltol = 1e-4, abstol = 1e-6),
    λ = 1e-8,
    nsample = 100,
);
# @save "./models/fnn_3layers_viscous_burgers_high_dim_t2_64_xpi_64_nu0.04_typ2_m10_256_up16_j173.bson" p


# === Check results ===

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
