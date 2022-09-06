using Statistics
using OrdinaryDiffEq

include("../solvers.jl");

"""
  galerkin_projection(t, S, Φ, ν, dx, dt)

POD-GP for non-conservative Burgers equation

# Arguments
- `t::Vector<Float>` t-axis values
- `S::Matrix` : matrix of snapshots
- `Φ::Vector<Float>`: pod modes
- `ν::Float`: viscosity
- `dx::Float`: x-axis discretization
- `dt::Float`: t-axis discretization
"""
function galerkin_projection(t, S, Φ, ν, dx, dt)
  u0 = copy(S[:, 1]);
  ū = mean(S, dims=2)[:, 1];
  n_modes = size(Φ)[2];

  function L(u)
    """ Linear operator """
    û = zeros(size(u)[1] + 2);
    û[2:end-1] = copy(u);
    û[1] = u[end];
    û[end] = u[1];
    return (û[1:end-2] - 2 * û[2:end-1] + û[3:end]) ./ (dx^2);
  end

  function N(u, v)
    """ Non-linear operator """
    v̂ = zeros(size(v)[1] + 2);
    v̂[2:end-1] = copy(v);
    v̂[1] = v[end];
    v̂[end] = v[1];

    dvdx =  - (v̂[3:end] - v̂[1:end-2]) ./ (2. * dx);
    return u .* dvdx
  end

  # Compute offline operators
  Lū = ν * L(ū);
  Nū = N(ū, ū);

  r = n_modes;
  B_k = zeros(r);
  L_k = zeros(r, r);
  N_k = zeros(r, r, r);

  for k in range(1, n_modes, step=1)
    B_k[k] = sum((Lū + Nū) .* Φ[:, k]);

    for i in range(1, n_modes, step=1)
      L_k[k, i] = sum((ν * L(Φ[:, i]) + N(ū, Φ[:, i]) + N(Φ[:, i], ū)) .* Φ[:, k]);

      for j in range(1, n_modes, step=1)
        N_k[k, i, j] = sum(N(Φ[:, i], Φ[:, j]) .* Φ[:, k]);
      end
    end
  end

  # Compute online operators
  function gp(a, B_k, L_k, N_k)
    dadt = B_k;
    dadt += L_k * a;

    for k in r
      dadt[k] = dadt[k] + (N_k[k, :, :] * a)' * a;
    end

    return dadt
  end

  a = Φ' * (u0 .- ū);

  # function f(dudt, u, p, t)
  #   B_k = p[1];
  #   L_k = p[2];
  #   N_k = p[3];
  #
  #   dudt = B_k;
  #   dudt += L_k * u;
  #
  #   for k in r
  #     dudt[k] = dudt[k] + (N_k[k, :, :] * u)' * u;
  #   end
  #
  #   return dudt
  # end

  # A2 = zeros(size(a)[1], size(t)[1])
  # tspan = (t[1], t[end])
  # prob = ODEProblem(ODEFunction(f), copy(a), tspan, (B_k, L_k, N_k))
  # sol = solve(prob, Tsit5(), saveat=t, reltol=1e-9, abstol=1e-9) 
  # A2 = hcat(sol.u)

  g = (x) -> gp(x, B_k, L_k, N_k);
  A = zeros(size(a)[1], size(t)[1])
  for (i, v) in enumerate(t)
    A[:, i] = copy(a);
    a = Solver.tvd_rk3(g, a, dt);
  end

  return ū .+ Φ * A;
end

