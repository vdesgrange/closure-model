using FFTW
using AbstractFFTs
using OrdinaryDiffEq
using Statistics
using Zygote


function get_burgers_fft(t, dx, x_n, nu, u0)
  """
  Pseudo-spectral method
  Solve non-conservative Burgers equation with pseudo-spectral method.
  """
  k = 2 * pi * AbstractFFTs.fftfreq(x_n, 1. / dx) # Sampling rate, inverse of sample spacing

  function f(u, p, t)
    k = p[1]
    nu = p[2]

    u_hat = FFTW.fft(u)
    u_hat_x = 1im .* k .* u_hat
    u_hat_xx = (-k.^2) .* u_hat

    u_x = FFTW.ifft(u_hat_x)
    u_xx = FFTW.ifft(u_hat_xx)
    u_t = -u .* u_x + nu .* u_xx
    return real.(u_t)
  end

  tspan = (t[1], t[end])
  prob = ODEProblem(ODEFunction(f), copy(u0), tspan, (k, nu))
  sol = solve(prob, RK4(), saveat=t, reltol=1e-8, abstol=1e-8) # AutoTsit5(Rosenbrock23()), 

  return sol.t, hcat(sol.u...)
end


function get_burgers_ccdf(t, dx, x_n, nu, u0)
  """
    get_burgers_ccdf(t, dx, x_n, nu, u0)

    Solve inviscid Burgers equation with a central difference schemes.
  Conservative Central Difference Schemes for One-Dimensional Scalar Conservation Laws
  "Energy Estimates for Nonlinear Conservation Laws with Applications to Solutions of the Burgers Equation and
    One-Dimensional Viscous Flow in a Shock Tube by Central Difference Schemes"

  """
  u02 = zeros(size(u0)[1] + 2);
  u02[2:end-1] = copy(u0);

  function num_flux(u)
    f = ((u[2:end].^2) + (u[2:end] .* u[1:end-1]) + (u[1:end-1].^2)) ./ 6
    return f
  end

  function f(u, p, t)
    nf_u = num_flux(u);

    u_t = zeros(size(u)[1])
    u_t[2:end-1] = - (nf_u[2:end] - nf_u[1:end-1]) ./ dx
    return u_t
  end

  tspan = (t[1], t[end])
  prob = ODEProblem(ODEFunction(f), copy(u0), tspan, (nu))
  sol = solve(prob, RK4(), saveat=t, reltol=1e-8, abstol=1e-8) # AutoTsit5(Rosenbrock23()), 

  return sol.t, hcat(sol.u...)
end

function get_burgers_godunov(t, dx, x_n, nu, u0)
  """
    get_burgers_godunov(t, dx, x_n, nu, u0)

  Godunov method
  "A Difference Method for the Numerical Calculation of Discontinous Solutions of Hydrodynamic Equations"
  """
  function riemann()
  end

  function num_flux(u)
  end

  function f(u, p, t)
  end

  tspan = (t[1], t[end])
  prob = ODEProblem(ODEFunction(f), copy(u0), tspan, (nu))
  sol = solve(prob, AutoTsit5(Rosenbrock23()), saveat=t, reltol=1e-8, abstol=1e-8)

  return sol.t, hcat(sol.u...)
end
