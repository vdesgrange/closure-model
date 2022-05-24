using FFTW
using AbstractFFTs
using OrdinaryDiffEq
using SuiteSparse
using SparseArrays

function get_burgers_fft(t, dx, x_n, nu, u0)
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
  sol = solve(prob, Tsit5(), saveat=t, reltol=1e-8, abstol=1e-8)

  return sol.t, hcat(sol.u...)
end
