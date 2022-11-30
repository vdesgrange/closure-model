using FFTW
using AbstractFFTs
using OrdinaryDiffEq
using DiffEqSensitivity
using Statistics
using Zygote

"""
    predict(f, v₀, p, t, solver; kwargs...)

Predict solution given parameters `p`.
"""
function predict(f, u₀, p, t, solver; kwargs...)
    problem = ODEProblem(f, u₀, extrema(t), p)
    sol = solve(problem, solver; saveat=t, kwargs...)
end

function f_fft(u, (ν, Δx, k), t)
  û = FFTW.fft(u)
  ûₓ = 1im .* k .* û
  ûₓₓ = (-k.^2) .* û

  uₓ = FFTW.ifft(ûₓ)
  uₓₓ = FFTW.ifft(ûₓₓ)
  uₜ = -u .* uₓ + ν .* uₓₓ
  return real.(uₜ)
end

function fflux(u, (ν, Δx), t)
  u₋ = circshift(u, 1)
  u₊ = circshift(u, -1)
  a₊ = u₊ + u
  a₋ = u + u₋
  du = @. -((a₊ < 0) * u₊^2 + (a₊ > 0) * u^2 - (a₋ < 0) * u^2 - (a₋ > 0) * u₋^2) / Δx + ν * (u₋ - 2u + u₊) / Δx^2
  du
end

function riemann(u, xt)
  S = (u[2:end] .+ u[1:end-1]) ./ 2.;
  a = (u[2:end] .>= u[1:end-1]) .* (((S .> xt) .* u[1:end-1]) .+ ((S .<= xt) .* u[2:end]));
  b = (u[2:end] .< u[1:end-1]) .* (
      ((xt .<= u[1:end-1]) .* u[1:end-1]) .+
      (((xt .> u[1:end-1]) .& (xt .< u[2:end])) .* xt) +
      ((xt .>= u[2:end]) .* u[2:end])
      );
  return a .+ b;
end

function νm_flux(u, xt=0.)
  r = riemann(u, xt);
  return r.^2 ./ 2.;
end

function f_godunov(u, (ν, Δx), t)
  ū = zeros(size(u)[1] + 2);
  ū[2:end-1] = deepcopy(u);
  nf_u = νm_flux(ū, 0.);
  uₜ = - (nf_u[2:end] - nf_u[1:end-1]) ./ Δx
  return uₜ
end

function get_burgers_fft(t, Δx, xₙ, ν, u₀)
  """
  Pseudo-spectral method.
  Solve non-conservative Burgers equation with pseudo-spectral method.
  Convinient for viscous case (preferable to finite-difference)
  """
  k = 2 * pi * AbstractFFTs.fftfreq(xₙ, 1. / Δx) # Sampling rate, inverse of sample spacing
  sol = predict(f_fft, copy(u₀), (ν, Δx, k), t, Tsit5(); reltol=1e-6, abstol=1e-6, sensealg=DiffEqSensitivity.InterpolatingAdjoint(; autojacvec=ZygoteVJP()))

  return sol.t, Array(sol)
end

function get_burgers_flux(t, Δx, xₙ, ν, u₀)
  """
  Flux method
  Solve non-conservative Burgers equation with numerical flux method (Jameson). Convenient for viscous case.
  """
  sol = predict(fflux, copy(u₀), (ν, Δx), t, Tsit5(); reltol=1e-6, abstol=1e-6, sensealg=DiffEqSensitivity.InterpolatingAdjoint(; autojacvec=ZygoteVJP()))

  return sol.t, Array(sol)
end

function get_burgers_ccdf(t, Δx, xₙ, ν, u₀)
  """
    get_burgers_ccdf(t, Δx, xₙ, ν, u₀)

    Solve inviscid Burgers equation with a central difference schemes.
  Conservative Central Difference Schemes for One-Dimensional Scalar Conservation Laws
  "Energy Estimates for Nonlinear Conservation Laws with Applications to Solutions of the Burgers Equation and
    One-Dimensional Viscous Flow in a Shock Tube by Central Difference Schemes"

  """
  u₀2 = zeros(size(u₀)[1] + 2);
  u₀2[2:end-1] = copy(u₀);

  function νm_flux(u)
    f = ((u[2:end].^2) + (u[2:end] .* u[1:end-1]) + (u[1:end-1].^2)) ./ 6
    return f
  end

  function f(u, p, t)
    nf_u = νm_flux(u);

    uₜ = zeros(size(u)[1])
    uₜ[2:end-1] = - (nf_u[2:end] - nf_u[1:end-1]) ./ Δx
    return uₜ
  end

  # tspan = (t[1], t[end])
  # prob = ODEProblem(ODEFunction(f), copy(u₀), tspan, (ν))
  # sol = solve(prob, RK4(), saveat=t, reltol=1e-8, abstol=1e-8)
  sol = predict(f, copy(u₀), (ν, Δx), t, Tsit5(); reltol=1e-6, abstol=1e-6, sensealg=DiffEqSensitivity.InterpolatingAdjoint(; autojacvec=ZygoteVJP()))
  return sol.t, hcat(sol.u...)
end

function get_burgers_godunov(t, Δx, xₙ, ν, u₀)
  """
    get_burgers_godunov(t, Δx, xₙ, ν, u₀)

  Godunov method. Convinient for inviscid case.
  "A Difference Method for the Numerical Calculation of Discontinous Solutions of Hydrodynamic Equations"
  """
  # tspan = (t[1], t[end])
  # prob = ODEProblem(ODEFunction(f), copy(u₀), tspan, (ν))
  # sol = solve(prob, AutoTsit5(Rosenbrock23()), saveat=t, reltol=1e-8, abstol=1e-8)
  sol = predict(f_godunov, copy(u₀), (ν, Δx), t, Tsit5(); reltol=1e-6, abstol=1e-6, sensealg=DiffEqSensitivity.InterpolatingAdjoint(; autojacvec=ZygoteVJP()))

  return sol.t, hcat(sol.u...)
end

function get_burgers_2d(t, Δ::Array{T, N}, n::Array{Integer, N}, ν, u₀) where {T, N}
  """
  Pseudo-spectral method
  Solve non-conservative 2-D Burgers equation with pseudo-spectral method.
  """
  Δx, Δy = Δ[1], Δ[2];
  xₙ, yₙ = n[1], n[2];

  kₓ = 2 * pi * AbstractFFTs.fftfreq(xₙ, 1. / Δx) # Sampling rate, inverse of sample spacing
  kⱼ = 2 * pi * AbstractFFTs.fftfreq(yₙ, 1. / Δy)


  function f(uₜ, u, p, t)
    û = FFTW.fft(u[1])
    v̂ = FFTW.fft(u[2])

    ûₓ = 1im .* kₓ .* û
    ûₓₓ = (-kₓ.^2) .* û
    uₓ = FFTW.ifft(ûₓ)
    uₓₓ = FFTW.ifft(ûₓₓ)

    ûⱼ = 1im .* kⱼ .* û
    ûⱼⱼ = (-kⱼ.^2) .* û
    uⱼ = FFTW.ifft(ûⱼ)
    uⱼⱼ = FFTW.ifft(ûⱼⱼ)

    v̂ₓ = 1im .* kₓ .* v̂
    v̂ₓₓ = (-kₓ.^2) .* v̂
    vₓ = FFTW.ifft(v̂ₓ)
    vₓₓ = FFTW.ifft(v̂ₓₓ)

    v̂ⱼ = 1im .* kⱼ .* v̂
    v̂ⱼⱼ = (-kⱼ.^2) .* v̂
    vⱼ = FFTW.ifft(v̂ⱼ)
    vⱼⱼ = FFTW.ifft(v̂ⱼⱼ)

    uₜ[1] =  real.(-u[1] .* uₓ - u[2] .* uⱼ + ν .* (uₓₓ + uⱼⱼ));
    uₜ[2] =  real.(-u[1] .* vₓ - u[2] .* vⱼ + ν .* (vₓₓ + vⱼⱼ));

    return uₜ
  end

  tspan = (t[1], t[end])
  prob = ODEProblem(ODEFunction(f), copy(u₀), tspan, (k, ν))
  sol = solve(prob, Tsit5(), saveat=t, reltol=1e-8, abstol=1e-8)

  return sol.t, hcat(sol.u...)
end
