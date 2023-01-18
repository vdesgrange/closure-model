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
using DiffEqSensitivity
using Zygote

include("../../utils/processing_tools.jl");
include("../../neural_ode/models.jl");
include("../../neural_ode/regularization.jl");
include("../../utils/generators.jl");
include("../../utils/graphic_tools.jl");
include("../../equations/equations.jl");
include("../../rom/pod.jl");

"""
    get_Φ(u, m)

    Get m-first POD basis of u.
    Log retained energy.
"""
function get_Φ(u, m)
    bas, _ = POD.generate_pod_svd_basis(u, false);
    λ = bas.eigenvalues;
    @show POD.get_energy(λ.^2, m)
    Φ = bas.modes[:, 1:m];
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
    kwargs...)
    # Select a random subset (batch)
    is = Zygote.@ignore sort(shuffle(1:size(v, 2))[1:nsolution])
    it = Zygote.@ignore sort(shuffle(1:length(t))[1:ntime])
    v = v[:, is, it]
    t = t[it]

    # Predicted soluton
    sol = predict(f, v[:, :, 1], p, t, solver; kwargs...)

    # Relative squared error
    # data = sum(abs2, sol - v) / sum(abs2, v)
    data = mean((sol .- v) .^ 2);
    
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
    # Predicted solution
    sol = predict(f, v[:, :, 1], p, t, solver; kwargs...)

    # Relative error
    e = norm(sol - v) / norm(v);

    # Mean Square Error
    mse = mean((sol .- v) .^ 2);

    return e, mse
end

function train(loss, p₀; maxiters = 100, optimizer = OptimizationOptimisers.Adam(0.001), callback = p -> false)
    @info("Initiate training")
    optf = OptimizationFunction(loss, Optimization.AutoZygote())
    optprob = OptimizationProblem(optf, p₀, nothing)
    sol = solve(optprob, optimizer; maxiters, callback=callback)
    sol.u, callback
end

function create_callback(f, v, t; ncallback = 1, solver = Tsit5(), kwargs...)
    i = 0
    iters = Int[]
    errors = zeros(0)
    mses = zeros(0)
    function callback(p, loss)
        i += 1
        if i % ncallback == 0
            e, mse = relative_error(f, p, v, t, solver; kwargs...)
            push!(iters, i)
            push!(errors, e);
            push!(mses, mse);
            println("Epoch $i\trelative error $(e)\t mean square error $(mse)")
            display(plot(iters, errors; xlabel = "Iterations", title = "Relative error"))
        end
        return false
    end
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
