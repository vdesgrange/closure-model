using Statistics
using OrdinaryDiffEq

include("../solvers.jl");

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
  u₀ = copy(S[:, 1]);
  ū = mean(S, dims=2)[:, 1];
  # ū = zeros(size(S, 1));
  n_modes = size(Φ)[2];

  function L(u)
    """ Linear operator """
    u₊ = circshift(u, -1);
    u₋ = circshift(u, +1);
    return (u₊ - 2 * u + u₋) ./ (dx^2);
  end

  function N(u, v)
    """ Non-linear operator """
    v₊ = circshift(v, -1);
    v₋ = circshift(v, +1);

    vₓ =  - (v₊ - v₋) ./ (2. * dx);
    return u .* vₓ
  end

  # Compute offline operators
  Lū = ν * L(ū);
  Nū = N(ū, ū);

  r = n_modes;
  B_k = zeros(r);
  L_k = zeros(r, r);
  N_k = zeros(r, r, r);

  B_k = ((Lū + Nū)' * Φ)[1, :];
  L_k = ((ν * L(Φ) + N(ū, Φ) + N(Φ, ū))' * Φ)';
  for k in range(1, n_modes, step=1)
    for i in range(1, n_modes, step=1)
      cst = N(Φ[:, i], Φ);
      N_k[k, i, :] .= sum(cst' * Φ[:, k]; dims=2);
    end
  end

  # for k in range(1, n_modes, step=1)
  #   B_k[k] = sum((Lū + Nū) .* Φ[:, k]);

  #   for i in range(1, n_modes, step=1)
  #     L_k[k, i] = sum((ν * L(Φ[:, i]) + N(ū, Φ[:, i]) + N(Φ[:, i], ū)) .* Φ[:, k]);

  #     for j in range(1, n_modes, step=1)
  #       N_k[k, i, j] = sum(N(Φ[:, i], Φ[:, j]) .* Φ[:, k]);
  #     end
  #   end
  # end

  # Compute online operators
  function gp(a, B_k, L_k, N_k)
    dadt = zeros(size(a))
    dadt .+= B_k;
    dadt .+= L_k * a;

    for k in size(N_k, 1)
      dadt[k] = dadt[k] + (N_k[k, :, :] * a)' * a;
    end

    return dadt
  end

  a = Φ' * (u₀ .- ū);

  g = (x) -> gp(x, B_k, L_k, N_k);
  A = zeros(size(a)[1], size(t)[1])
  for (i, v) in enumerate(t)
    A[:, i] = copy(a);
    a = Solver.tvd_rk3(g, a, dt);
  end

  # return ū .+ Φ * A;
  return ū .+ Φ * A, A;
end

"""
  offline_op(ū, Φ, ν, Δx)

  Compute offline operator used by Galerkin projection for non-conservative Burgers equation
    Use finite-difference for offline operator. Losing precision (to replace).
  # Arguments
  - `ū::Vector<Float>` time averaged of solution snapshot
  - `Φ::Vector<Float>`: pod modes
  - `ν::Float`: viscosity
  - `Δx::Float`: x-axis discretization
"""
function offline_op(ū, Φ, ν, Δx)
  n_modes = size(Φ)[2];

  function L(u) # Finite difference not accurate
    """ Linear operator """
    u₊ = circshift(u, -1);
    u₋ = circshift(u, +1);
    return (u₊ - 2 * u + u₋) ./ (Δx^2);
  end

  function N(u, v)  # Finite difference not accurate
    """ Non-linear operator """
    v₊ = circshift(v, -1);
    v₋ = circshift(v, +1);

    vₓ =  - (v₊ - v₋) ./ (2. * Δx);
    return u .* vₓ
  end

  # Compute offline operators
  Lū = ν * L(ū);
  Nū = N(ū, ū);

  r = n_modes;
  B_k = zeros(r);
  L_k = zeros(r, r);
  N_k = zeros(r, r, r);

  B_k = ((Lū + Nū)' * Φ)[1, :];
  L_k = ((ν * L(Φ) + N(ū, Φ) + N(Φ, ū))' * Φ)';

  for k in range(1, n_modes, step=1)
    for i in range(1, n_modes, step=1)
      cst = N(Φ[:, i], Φ);
      N_k[k, i, :] .= sum(cst' * Φ[:, k]; dims=2);
    end
  end

  return B_k, L_k, N_k
end


"""
  gp(a, (B_k, L_k, N_k), t)

  POD-GP for non-conservative Burgers equation

  # Arguments
  - `a::Vector<Float>` a(t)
  - `(B_k, L_k, N_k)` : offline operators
  - `t::Vector<Float>` t-axis values
"""
function gp(a, (B_k, L_k, N_k), t)
  aₜ = zeros(size(a));
  aₜ .+= B_k;
  aₜ .+= L_k * a;
  for k in size(N_k, 1)
    cst = ((N_k[k, :, :] * a)' * a)[1, 1];
    aₜ[k, 1] = aₜ[k, 1] + cst;
  end
  return aₜ
end



"""
  galerkin_projection_2(u₀, (ū, Φ, ν, Δx), t)

  POD-GP for non-conservative Burgers equation
  Equivalent (but faster) than:
    gg(a, p, t) = Φ' * ff(Φ * a + ū_pod, p, t)
    
  # Arguments
  - `u₀::Vector<Float>` :initial conditions
  - `ū::Vector<Float>` time averaged of solution snapshot
  - `Φ::Vector<Float>`: pod modes
  - `ν::Float`: viscosity
  - `Δx::Float`: x-axis discretization
  - `t::Vector<Float>` t-axis values
"""
function galerkin_projection_2(u₀, (ū, Φ, ν, Δx), t)
  # if ū != 0 then a = Φ'(u - ū)
  # if  ū == 0 then a = Φ'u
  a₀ = Φ' * (u₀ .- ū);
  B_k, L_k, N_k = Equations.offline_op(ū, Φ, ν, Δx);
  a_prob = ODEProblem(Equations.gp, a₀, extrema(t), (B_k, L_k, N_k), saveat=t);
  A = Array(solve(a_prob, Tsit5(), sensealg=DiffEqSensitivity.InterpolatingAdjoint(; autojacvec=ZygoteVJP())));
  return ū .+ tensormul(Φ, A), A;
end


