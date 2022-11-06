using FFTW
using AbstractFFTs
using OrdinaryDiffEq
using DiffEqSensitivity
using SparseArrays
using Distributions

include("../initial_functions.jl")

function get_heat(t, x, n=[], c=[], ka=1.)
  return InitialFunctions.analytical_heat_1d(t, x, n, c, ka)
end

"""
Central finite difference method w/ ODE solver
Solve diffusion equation using matrix multiplication
"""
function get_heat_fd_impl(Δt, Δx, tₙ, xₙ, u₀)
  u = copy(u₀)
  s = Δt / Δx^2
  D = spdiagm(-1 => fill(-s, xₙ - 1), 0 => fill(1 - 2 * s, xₙ), 1 => fill(-s, xₙ - 1))

  for i in range(2, tₙ + 1, step=1)
    u[i, :] = D \ u[i-1, :]
    u[i, 1] = 0
    u[i, end] = 0
  end
  
  return u
end

"""
Central finite difference method
Solve diffusion equation 
with central finite difference method using order 2 of accuracy.
"""
function get_heat_fd(t, Δx, u₀, κ)
  function f(u, p, t)
    Δx = p[1];
    κ = p[2];
    u₋ = circshift(u, 1);
    u₊ = circshift(u, -1);

    u₋[1] = 0; # -1
    u₊[end] = 0; # 1

    uₓₓ = (u₊ .- 2 .* u .+ u₋) ./ Δx^2;
    uₜ = κ * uₓₓ;
  
    return uₜ
  end
  
  prob = ODEProblem(ODEFunction(f), u₀, extrema(t), (Δx, κ));
  sol = solve(prob, Tsit5(), saveat=t, dt=0.01); 
  return sol.t, Array(sol)
end


function get_heat_fft(t, Δx, xₙ, κ, u₀)
  k = 2 * pi * AbstractFFTs.fftfreq(xₙ, 1. / Δx) # Sampling rate, inverse of sample spacing

  function f(u, p, t)
    k = p[1]
    κ = p[2]

    û = FFTW.fft(u)
    ûₓₓ = (-k.^2) .* û

    uₓₓ = FFTW.ifft(ûₓₓ)
    uₜ = κ * uₓₓ;

    return real.(uₜ)
  end

  prob = ODEProblem(ODEFunction(f), copy(u0), extrema(t), (k, κ))
  sol = solve(prob, Tsit5(), saveat=t, reltol=1e-8, abstol=1e-8)
  return sol.t, Array(sol)
end
