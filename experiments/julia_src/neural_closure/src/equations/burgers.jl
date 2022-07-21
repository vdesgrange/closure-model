using FFTW
using AbstractFFTs
using OrdinaryDiffEq
using SuiteSparse
using SparseArrays
using Statistics

function weno5()
end

function get_burgers_fft(t, dx, x_n, nu, u0)
  """
  Pseudo-spectral method
  Solve conservative Burgers equation with pseudo-spectral method.
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
  sol = solve(prob, AutoTsit5(Rosenbrock23()), saveat=t, reltol=1e-8, abstol=1e-8)

  return sol.t, hcat(sol.u...)
end


function galerkin_projection(t, S, Φ, ν, dx, dt)
  """
  POD-GP for non-conservative Burgers equation
  """
  u0 = copy(S[:, 1]);
  ū = mean(S, dims=2);
  a0 = u0 - ū;
  n_modes = size(Φ)[2];

  function L(u)
    û = zeros(size(u)[1] + 2);
    û[2:end-1] = copy(u);
    û[1] = u[end];
    û[end-1] = u[1];
    return (û[1:end-2] - 2 * û[2:end-1] + û[3:end]) ./ dx^2;
  end

  function N(u, v)
    v̂ = zeros(size(v)[1] + 2);
    v̂[2:end-1] = copy(v);
    v̂[1] = v[end];
    v̂[end-1] = v[1];

    dvdx = (v̂[1:end-2] - v̂[3:end]) ./ (2 * dx);
    return u .* dvdx
  end

  # Compute offline operators
  Lū = ν * L(ū);
  Nū = N(ū, ū);

  r = n_modes;
  B_k = zeros(r);
  L_k = zeros(r, r);
  N_k = zeros(r, r, r);

  for k in n_modes
    B_k[k] = sum((Lū - Nū) .* Φ[:, k]);

    for i in n_modes
      L_k[k, i] = sum((ν * L(Φ[:, i]) - N(ū, Φ[:, i])- N(Φ[:, i], ū)) .* Φ[:, k]);

      for j in n_modes
        N_k[k, i, j] = sum(N(Φ[:, i], Φ[:, j]) .* Φ[:, k]);
      end
    end
  end

  # Compute online operators
  # function gp(a, p, t)
  function gp(a, B_k, L_k, N_k)
    # B_k = p[0]
    # L_k = p[1]
    # N_k = p[2]
    dadt = B_k;
    dadt += L_k * a;

    for k in r
      dadt[k] = dadt[k] + (N_k[:, :, k] * a)' * a;
    end

    return dadt
  end

  a0 = Φ' * S[:, 1];
  a = copy(a0);
  A = zeros(size(a)[1], size(t)[1])
  for (i, v) in enumerate(t)
    rhs = gp(a, B_k, L_k, N_k);
    a1 = a + dt * rhs;
    rhs = gp(a1, B_k, L_k, N_k);
    a2 = 0.75 * a + 0.25 * a1 + 0.25 * dt * rhs;
    rhs = gp(a2, B_k, L_k, N_k);
    a = (1.0/3.0) * a + (2.0/3.0) * a2 + (2.0/3.0) * dt * rhs;
    A[:, i] = copy(a);
  end

  # tspan = (t[1], t[end])
  # prob = ODEProblem(ODEFunction(gp), copy(a0), tspan, (B_k, L_k, N_k))
  # A = solve(prob, RK4(), saveat=t, reltol=1e-8, abstol=1e-8)
  return ū .+ Φ * A;
  # return ū + Φ * a;
end
