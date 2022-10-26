using FFTW
using AbstractFFTs
using OrdinaryDiffEq
using DiffEqSensitivity
using Statistics
using Zygote

function get_kdv_fft(t, Δx, xₙ, u₀)
  """
  Pseudo-spectral method
  Solve original Korteweg de Vries(KdV) equation with pseudo-spectral method.
  """
  k = 2 * pi * AbstractFFTs.fftfreq(xₙ, 1. / Δx) # Sampling rate, inverse of sample spacing

  function f(u, p, t)
    k = p[1]

    û = FFTW.fft(u)
    ûₓ = 1im .* k .* û
    ûₓₓₓ = -1im .* k.^3 .* û
  
    uₓ = FFTW.ifft(ûₓ)
    uₓₓₓ = FFTW.ifft(ûₓₓₓ)
    uₜ = (-6 * u .* uₓ) .- uₓₓₓ
    return real.(uₜ)
  end

  prob = ODEProblem(ODEFunction(f), copy(u₀), extrema(t), (k));
  # sol = solve(prob, Tsit5(), saveat=t, reltol=1e-7, abstol=1e-7);
  sol = solve(prob, Rodas4P(), saveat=t, dt=0.01);

  return sol.t, hcat(sol.u...)
end

function get_kdv_fd(t, Δx, u₀;)
  """
  Central finite difference method
  Solve original Korteweg de Vries (KdV) equation 
  with central finite difference method using order 2 of accuracy.
  """

  function f(u, p, t)
    u₋₋ = circshift(u, 2);
    u₋ = circshift(u, 1);
    u₊ = circshift(u, -1);
    u₊₊ = circshift(u, -2);
  
    uₓ = (u₊ - u₋) ./ (2 * Δx);
    uₓₓₓ = (-u₋₋ .+ 2u₋ .- 2u₊ + u₊₊) ./ (2 * Δx^3);
    uₜ = -(6u .* uₓ) .- uₓₓₓ;
    return uₜ
  end
  
  prob = ODEProblem(ODEFunction(f), copy(u₀), extrema(t));
  # sol = solve(prob, Tsit5(), saveat=t, reltol=1e-7, abstol=1e-7);
  sol = solve(prob, Tsit5(), saveat=t, dt=0.01);

  return sol.t, hcat(sol.u...)
end
