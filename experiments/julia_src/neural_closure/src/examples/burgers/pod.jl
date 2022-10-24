using LinearAlgebra
using OrdinaryDiffEq
using Plots
using Printf
using Random

function f(u, ν, t)
    Δx = one(t) / size(u, 1)
    u₋ = circshift(u, 1)
    u₊ = circshift(u, -1)
    a₊ = u₊ + u
    a₋ = u + u₋
    du = @. -((a₊ < 0) * u₊^2 + (a₊ > 0) * u^2 - (a₋ < 0) * u^2 - (a₋ > 0) * u₋^2) / Δx + ν * (u₋ - 2u + u₊) / Δx^2
    du
end

function s(u₀, ν, t; kwargs...)
    problem = ODEProblem(f, u₀, extrema(t), ν)
    solve(problem, Tsit5(); saveat = t, kwargs...)
end

# Viscosity
ν = 0.001

# Spatial discretization
N = 100
x = LinRange(0, 1, N + 1)[2:end]

# Plot example solution
u₀ = @. sinpi(2x) + sinpi(5x) + cospi(10x)
t = LinRange(0, 0.2, 51)
sol = s(u₀, ν, t)
for (i, t) ∈ enumerate(t)
    pl = plot(; xlabel = "x", title = @sprintf("t = %.3f", t), ylims = extrema(sol[:, :]))
    plot!(pl, x, sol[i])
    display(pl)
    sleep(0.05)
end

# Fourier basis
K = 20
basis = [exp(2π * im * k * x) for x ∈ x, k ∈ -K:K]

# Fourier coefficients with random phase and amplitude
n_train = 100
n_test = 50
c_train = [randn() * exp(-2π * im * rand()) / (1 + abs(k)) for k ∈ -K:K, _ ∈ 1:n_train]
c_test = [randn() * exp(-2π * im * rand()) / (1 + abs(k)) for k ∈ -K:K, _ ∈ 1:n_test]

# Initial conditions (real-valued)
u₀_train = real.(basis * c_train)
u₀_test = real.(basis * c_test)

# Evaluation times
t_train = LinRange(0, 0.1, 101)
t_test = LinRange(0, 0.2, 61)

plot(x, u₀_train[:, 1:3]; xlabel = "x", title = "Initial conditions")

# Solutions
u_train = s(u₀_train, ν, t_train; reltol = 1e-4, abstol = 1e-6)
u_test = s(u₀_test, ν, t_test; reltol = 1e-4, abstol = 1e-6)
size(u_test)
N
print(size(reshape(Array(u_test), N, :)))
(; U, S, V) = svd(reshape(Array(u_train), N, :))

size(U)
size(S)
size(V)
plot(x, U[:, 1:3])
scatter(S; yscale = :log10)
size(u_test)
nmode = 30
Φ = U[:, 1:nmode]

i = 1
plot(u₀_test[:, i])
plot!(Φ * Φ' * u₀_test[:, i])

a₀_train = Φ' * u₀_train
a₀_test = Φ' * u₀_test

Φ

f_pod(u, ν, t) = Φ * Φ' * f(u, ν, t)
# f_pod(u, ν, t) = f(Φ * Φ' * u, ν, t)
# f_pod(u, ν, t) = Φ * Φ' * f(Φ * Φ' * u, ν, t)
function s_pod(u₀, ν, t; kwargs...)
    problem = ODEProblem(f_pod, u₀, extrema(t), ν)
    solve(problem, Tsit5(); saveat = t, kwargs...)
end

fa(a, ν, t) = Φ' * f(Φ * a, ν, t)

function sa(a₀, ν, t; kwargs...)
    problem = ODEProblem(fa, a₀, extrema(t), ν)
    solve(problem, Tsit5(); saveat = t, kwargs...)
end

a_train = sa(a₀_train, ν, t_train; reltol = 1e-4, abstol = 1e-6)
a_test = sa(a₀_test, ν, t_test; reltol = 1e-4, abstol = 1e-6)

ua_train = [Φ * a_train[i] for i = 1:length(t_train)]

## Plot example solution
u₀ = @. sinpi(2x) + sinpi(5x) + cospi(10x)
# u₀ = u₀_train[:, 1]
u₀ = u₀_test[:, 2]
# t = LinRange(0, 0.5, 201)
sol = s(Φ * Φ' * u₀, ν, t)
sol_pod = s_pod(Φ * Φ' * u₀, ν, t)
sola = sa(Φ' * u₀, ν, t)
for (i, t) ∈ enumerate(t)
    pl = plot(; xlabel = "x", title = @sprintf("t = %.3f", t), ylims = extrema(sol[:, :]))
    plot!(pl, x, sol[i]; label = "Full")
    plot!(pl, x, sol_pod[i]; label = "POD")
    # plot!(pl, x, Φ * sola[i]; label = "POD")
    display(pl)
    sleep(0.05)
end

# Plot evolution of modes
pl = plot(; xlabel = "t", legend = :bottomright)
for i = 1:3
    plot!(pl, t, Array(sol)' * Φ[:, i]; linestyle = :dash, color = i, label = "a[$i] true")
    plot!(
        pl, t, sola[i, :];
        color = i,
        label = false,
        # label = "a[$i] predicted",
    )
end
pl

using Lux
using Zygote
using DiffEqSensitivity
using Optimization
using Optimisers

# Discrete closure term for filtered Burgers equation
d₁ = 3
d₂ = 4
d₃ = 1
NN = Chain(
    # From (nx, nbatch) to (nx, nchannel, nbatch)
    u -> reshape(u, size(u, 1), 1, size(u, 2)),
    # Add square channel to mimic Burgers term
    u -> cat(u, u .* u; dims = 2),
    # Manual padding to account for periodicity
    u -> [u[end-(d₁+d₂+d₃)+1:end, :, :]; u; u[1:(d₁+d₂+d₃), :, :]],
    # Some convolutional layers to mimic local differential operators
    Conv((2d₁ + 1,), 2 => 4, Lux.relu),
    Conv((2d₂ + 1,), 4 => 3, Lux.relu),
    Conv((2d₃ + 1,), 3 => 1),
    # From (nx, nchannel, nbatch) to (nx, nbatch)
    u -> reshape(u, size(u, 1), size(u, 3)),
);
NN

# Get parameter structure for NN
rng = Random.default_rng()
Random.seed!(rng, 0)
params, state = Lux.setup(rng, NN)
p₀, re = Lux.destructure(params)

"""
Compute right hand side of filtered Burgers equation.
This is modelled as unfiltered RHS + neural closure term.
"""
function f_NN(u, p, t)
    du = f_pod(u, ν, t)
    closure = first(Lux.apply(NN, u, re(p), state))
    du + closure
end

function s_NN(u₀, p, t; kwargs...)
    problem = ODEProblem(f_NN, u₀, extrema(t), p)
    solve(problem, Tsit5(); saveat = t, kwargs...)
end

"""
Derivative-fitting loss function.
"""
function loss_derivative_fit(p, u, λ)
    t = 0.0
    dudt_ref = f(u, ν, t)
    dudt_predict = f_NN(u, p, t) #  f_pod(u, ν, t) +
    data = sum(abs2, dudt_predict - dudt_ref) / length(u)
    reg = sum(abs2, p) / length(p)
    data + λ * reg
end

"""
Trajectory-fitting loss function.
"""
function loss_embedded(p, u, t, λ; kwargs...)
    sol = s_NN(u[:, :, 1], p, t; kwargs...)
    data = sum(abs2, sol - u) / prod(size(u)[2:3])
    reg = sum(abs2, p) / length(p)
    data + λ * reg
end

U = reshape(Array(u_train), N, :)[:, 1:1]
loss(p) = loss_derivative_fit(p, U, 1e-8)

# U = Array(u_train)[:, 1:1, :]
# loss(p) = loss_embedded(
#     p, U, t_train, 1e-8;
#     # sensealg = BacksolveAdjoint(; autojacvec = ZygoteVJP()),
#     sensealg = InterpolatingAdjoint(; autojacvec = ZygoteVJP()),
#     # sensealg = QuadratureAdjoint(; autojacvec = ZygoteVJP()),
# )

p₀
loss(p₀)

p = p₀
grad = first(gradient(loss, p))

opt = Optimisers.setup(Optimisers.ADAM(0.001f0), p)

for i ∈ 1:1000
    println("Iteration $i, \t loss = $(loss(p))")
    grad = first(gradient(loss, p))
    opt, p = Optimisers.update(opt, p, grad)
end

## Plot example solution
# u₀ = @. sinpi(2x) + sinpi(5x) + cospi(10x)
u₀ = u₀_train[:, 1]
# u₀ = u₀_test[:, 2]
# t = LinRange(0, 0.5, 201)
sol = s(Φ * Φ' * u₀, ν, t)
sol_pod = s_pod(Φ * Φ' * u₀, ν, t)
sol_NN = s_NN(reshape(Φ * Φ' * u₀, :, 1), p₀, t)
for (i, t) ∈ enumerate(t)
    pl = plot(; xlabel = "x", title = @sprintf("t = %.3f", t), ylims = extrema(sol[:, :]))
    plot!(pl, x, sol[i]; label = "Full")
    plot!(pl, x, sol_pod[i]; label = "POD")
    plot!(pl, x, sol_NN[i]; label = "POD + NN")
    # plot!(pl, x, Φ * sola[i]; label = "POD")
    display(pl)
    sleep(0.01)
end

include("../..//utils/graphic_tools.jl")

problem = ODEProblem(f_NN, u₀, extrema(t), p)
solve(problem, Tsit5(); saveat = t, kwargs...)

GraphicTools.show_state(Array(sol'), x, t, "", "", "")
GraphicTools.show_state(Array(sol_pod'), x, t, "", "", "")
GraphicTools.show_state(Array(sol_NN[:, 1, :]'), x, t, "", "", "")