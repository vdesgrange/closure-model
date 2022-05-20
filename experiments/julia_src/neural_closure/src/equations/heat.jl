using FFTW
using OrdinaryDiffEq
using SuiteSparse
using SparseArrays

include("initial_functions.jl")

function get_heat(t, x, n=[], c=[], k=1.)
  return InitialFunctions.analytical_heat_1d(t, x, n, c, k)
end

function get_heat_fd_impl(dt, dx, t_n, x_n, u0=none)
  u = copy(u0)
  s = dt / dx^2
  D = spdiagm(-1 => fill(-s, x_n - 1), 0 => fill(1 - 2 * s, x_n), 1 => fill(-s, x_n - 1))

  for i in range(2, t_n + 1, step=1)
    u[i, :] = D \ u[i-1, :]
    u[i, 1] = 0
    u[i, end] = 0
  end

  return u
end

function get_heat_fft(t, dx, x_n, d=1., u0=none)
  k = 2 * pi * fftfreq(x_n, dx)

  function f(u, p, t)
    k = p[1]
    d = p[2]
    u_hat = fft(u)
    u_hat_xx = -k.^2 .* u_hat
    u_xx = ifft(u_hat_xx)
    u_t = d * u_xx
    return real.(u_t)
  end

  tspan = [t[1], t[end]]
  prob = ODEProblem(f, u0, tspan, (k, d))
  sol = solve(prob, Tsit5(), tstops=t, reltol=1e-8, abstol=1e-8)

  return sol.t, sol.u
end
