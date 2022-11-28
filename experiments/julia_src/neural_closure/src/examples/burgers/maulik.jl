using AbstractFFTs
using FFTW
using Statistics
using LaTeXStrings
include("burgers_closure_galerkin_2.jl")
include("../../equations/equations.jl");

function analytical_solution(x, t)
    Re = 1. / ν;
    t₀ = exp(Re / 8.);
    a =  x ./ (t .+ 1);
    b =  1 .+ sqrt.((t .+ 1) ./ t₀) .* exp.(Re .* (x.^2 ./ (4 .* t .+ 4)));
    u = a ./ b;
end

function f_fft(u, (ν, Δx), t)
    û = FFTW.fft(u)
    ûₓ = 1im .* k .* û
    ûₓₓ = (-k.^2) .* û

    uₓ = FFTW.ifft(ûₓ)
    uₓₓ = FFTW.ifft(ûₓₓ)
    uₜ = -u .* uₓ + ν .* uₓₓ
    return real.(uₜ)
end

function fflux2(u, (ν, Δx), t)
    u₋ = circshift(u, 1)
    u₊ = circshift(u, -1)
    u₊[end] = 0;
    u₋[1] = 0;
    a₊ = u₊ + u
    a₋ = u + u₋
    du = @. -((a₊ < 0) * u₊^2 + (a₊ > 0) * u^2 - (a₋ < 0) * u^2 - (a₋ > 0) * u₋^2) / Δx + ν * (u₋ - 2u + u₊) / Δx^2
    du
end  

function offline_op(ū, Φ, ν, Δx)
    n_modes = size(Φ)[2];
  
    function L(u)
      """ Linear operator """
      u₊ = circshift(u, -1);
      u₋ = circshift(u, +1);
      return (u₊ - 2 * u + u₋) ./ (Δx^2);
    end
  
    function N(u, v)
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

function galerkin_projection_2(u, (ū, Φ, ν, dx), t)
    u₀ = copy(u[:, 1, :]);
    a₀ = Φ' * (u₀ .- ū);
    B_k, L_k, N_k = offline_op(ū, Φ, ν, dx);

    # Compute online operators
    function gp(a, (B_k, L_k, N_k), t)
      dadt = zeros(size(a))
      dadt .+= B_k;
      dadt .+= L_k * a;
      for k in size(N_k, 1)
        cst = ((N_k[k, :, :] * a)' * a)[1, 1];
        dadt[k, 1] = dadt[k, 1] + cst;
      end
      return dadt
    end

    a_prob = ODEProblem(gp, a₀, extrema(t), (B_k, L_k, N_k), saveat=t);
    A = Array(solve(a_prob, Tsit5(), sensealg=DiffEqSensitivity.InterpolatingAdjoint(; autojacvec=ZygoteVJP())));
    
    return ū .+ Φ * A, A;
end

function create_data(f, p, K, x, nsolution, t; decay = k -> 1 / (1 + abs(k)), kwargs...)
    L = x[end]
    basis = [exp(2π * im * k * x / L) for x ∈ x, k ∈ -K:K]
    c = [randn() * decay(k) * exp(-2π * im * rand()) for k ∈ -K:K, _ ∈ 1:nsolution]
    u₀ = real.(basis * c)
    @show size(u₀)
    predict(f, u₀, p, t, Tsit5(); kwargs...)
end

function create_data_maulik(f, p, K, x, nsolution, t; kwargs...)
    Re = 1. / ν;
    t₀ = exp(Re / 8.);
    c = x ./ (1 .+ sqrt(1 / t₀) * exp.(Re * (x.^2 ./ 4)));
    c[1] = 0;
    c[end] = 0;
    u₀ = hcat([c for _ ∈ 1:nsolution]...);
    # Solve ODE
    predict(f, u₀, p, t, Tsit5(); kwargs...)
end

epochs = 100;
ratio = 0.75;
batch_size = 64; # High for derivative fitting (i.e. 64)
lr = 1e-3;
reg = 1e-7;
noise = 0.05;
tₘₐₓ= 6.; # 2.
tₘᵢₙ = 0.;
xₘₐₓ = 1.; # pi
xₘᵢₙ = 0;
tₙ = 64;
xₙ = 64;
ν = 0.001; # Viscosity
m = 3;  # Number of POD modes
K = 100;  # Maximum frequency in initial conditions# Sampling rate, inverse of sample spacing
sol = Tsit5();

# FOM discretization
x = LinRange(xₘᵢₙ, xₘₐₓ, xₙ);
Δx = (xₘₐₓ - xₘᵢₙ) / (xₙ - 1);
k = 2 * pi * AbstractFFTs.fftfreq(xₙ, 1. / Δx) 

# Training times
t_pod = LinRange(0.0, 2., Int(tₙ));
Δt = (2. - tₘᵢₙ) / (tₙ - 1);
t_train = LinRange(0.0, 2., Int(tₙ * 4));
t_valid = LinRange(0.0, 4., Int(tₙ * 2));
t_test = LinRange(0.0, tₘₐₓ, Int(tₙ * 3));

# === Generate data ===
fflux = Equations.fflux;
# u_pod = analytical_solution(x, Array(t_pod)')
u_pod = create_data_maulik(f_fft, (ν, Δx), K, x, 16, t_pod; reltol = 1e-6, abstol = 1e-6);
u_train = create_data_maulik(fflux, (ν, Δx), K, x, 16, t_train; reltol = 1e-6, abstol = 1e-6);
u_valid  = create_data_maulik(fflux, (ν, Δx), K, x, 16, t_valid; reltol = 1e-6, abstol = 1e-6);
u_test  = create_data_maulik(fflux, (ν, Δx), K, x, 16, t_test; reltol = 1e-6, abstol = 1e-6);

ū_pod = mean(u_pod, dims=3)[:, 1]; # 2, or 3 if batch
ū_train = mean(u_train, dims=3)[:, 1];
ū_valid = mean(u_valid, dims=3)[:, 1];
ū_test = mean(u_test, dims=3)[:, 1];

display(GraphicTools.show_state(u_pod, t_pod, x, "", "t", "x"))

# Time derivatives
dudt_pod = fflux(u_pod, (ν, Δx), 0);
dudt_train = fflux(u_train, (ν, Δx), 0);
dudt_valid = fflux(u_valid, (ν, Δx), 0);
dudt_test = fflux(u_test, (ν, Δx), 0);

û2, A2 = galerkin_projection_2(u_pod, (ū_pod, Φ, ν, Δx), t_pod);


# Get POD basis
# Maulik 2020 : POD.generate_pod_basis(u, true); (f_fft) fflux + (T=2) + eigenproblem + substract mean (0.86 for fflux, 0.88 for fflux2)
# Gupta 2021 : POD.generate_pod_svd_basis(u, false); fflux2 + (T=2 or 4) + svd + substract mean
# bas, _ = POD.generate_pod_svd_basis(u_pod, true); # gupta
# @show POD.get_energy(bas.eigenvalues, m);
bas, _ = POD.generate_pod_basis(u_pod, true); # maulik
@show POD.get_energy(bas.eigenvalues, m);
Φ = (bas.modes[:, 1:m]);
û, A = Equations.galerkin_projection(t_pod, u_pod, Φ, ν, Δx, Δt);
û2, A2 = galerkin_projection_2(u_pod, (ū_pod, Φ, ν, Δx), t_pod);
# Φ = get_Φ(reshape(u_pod, xₙ, :), 3); # Eigenproblem
# A = Φ' * (u_pod .- mean(u_pod, dims=2));
begin
    plt = plot(title="Coefficients", xlabel="t", ylabel="ϕ", background_color_legend = RGBA(1, 1, 1, 0.8))
    plot!(plt; dpi=600)
    plot!(plt, t_pod, bas.coefficients[1, :], c=:coral, label=L"a_1")
    plot!(plt, t_pod, bas.coefficients[2, :], c=:green, label=L"a_2")
    plot!(plt, t_pod, bas.coefficients[3, :], c=:blue, label=L"a_3")

    plot!(plt, t_pod, A[1, :], c=:coral, linestyle=:dash,label=L"gp_1")
    plot!(plt, t_pod, A[2, :], c=:green, linestyle=:dash,label=L"gp_2")
    plot!(plt, t_pod, A[3, :], c=:blue, linestyle=:dash,label=L"gp_3")

    # plot!(plt, t_pod, A2[1, :], c=:red, linestyle=:dash,label=L"gp2_1")
    # plot!(plt, t_pod, A2[2, :], c=:red, linestyle=:dash,label=L"gp2_2")
    # plot!(plt, t_pod, A2[3, :], c=:red, linestyle=:dash,label=L"gp2_3")
end


v_pod = tensormul(Φ', u_pod .- ū_pod); # v_pod = Φ' * u_pod ### A = Φ' * (u_pod .- mean(u_pod, dims=2));
v_train = tensormul(Φ', u_train);
v_valid = tensormul(Φ', u_valid);
v_test = tensormul(Φ', u_test);

# Time derivatives of POD coefficients
dvdt_pod = tensormul(Φ', dudt_pod);
dvdt_train = tensormul(Φ', dudt_train);
dvdt_valid = tensormul(Φ', dudt_valid);
dvdt_test = tensormul(Φ', dudt_test);

# === Maulik paper batching : batch per time ===

ntime = 20
it = Zygote.@ignore sort(shuffle(1:length(t_pod))[1:20])
sort(shuffle(1:length(t_pod))[1:20])

sort(shuffle(1:size(v_train, 2))[1:8])

it2 = Zygote.@ignore sort(shuffle(1:length(t_pod) - ntime)[1:ntime])

function loss_time_trajectory_fit(
    f,
    p,
    v,
    t;
    solver = Tsit5(),
    λ = 1e-8,
    nsolution = size(v, 2),
    ntime = size(v, 3),
    kwargs...)
    # Select a random subset (batch)
    is = Zygote.@ignore sort(shuffle(1:size(v, 2))[1:nsolution])
    # it = Zygote.@ignore sort(shuffle(1:length(t))[1:ntime])
    st = Zygote.@ignore rand(1:(length(t_pod) - ntime))
    it = Zygote.@ignore st:(st + ntime)

    v = v[:, is, it]
    t = t[it]

    # Predicted soluton
    sol = predict(f, v[:, :, 1], p, t, solver; kwargs...)

    # Relative squared error
    data = sum(abs2, sol - v) / sum(abs2, v)

    # Regularization term
    reg = sum(abs2, p) / length(p)

    # Loss
    data + λ * reg
end

# ==== Model ====

model = Flux.Chain(
    v -> vcat(v, v .^ 2),
    Flux.Dense(2m => 2m, tanh; init = Models.glorot_uniform_float64, bias = true),
    Flux.Dense(2m => m, tanh; init = Models.glorot_uniform_float64, bias = true),
    Flux.Dense(m => m, identity; init = Models.glorot_uniform_float64, bias = true),
);
p₀, re = Flux.destructure(model);
fᵣₒₘ(v, p, t) = re(p)(v);

loss_tf(p, _) = loss_trajectory_fit( # loss_time_trajectory_fit
    fᵣₒₘ,
    p,
    v_train,
    t_train;
    nsolution = 16,
    ntime = 10,
    reltol = 1e-6,
    abstol = 1e-6,
);

p_tf = train(
    loss_tf,
    p₀;
    maxiters = 1000,
    optimizer = OptimizationOptimisers.Adam(lr),
    callback = create_callback(
        fᵣₒₘ,
        v_valid,
        t_valid;
        ncallback = 8,
        reltol = 1e-4,
        abstol = 1e-6,
    ),
);

sol_tf = predict(
    fᵣₒₘ,
    v_test[:, :, 1],
    p_tf,
    t_test,
    Tsit5();
    reltol = 1e-6,
    abstol = 1e-6,
);

iplot = 1;
for (i, t) ∈ enumerate(t_test)
    pl = Plots.plot(;title = @sprintf("Solution, t = %.3f", t), xlabel = "x", ylims = extrema(Φ * v_test[:, iplot, :]))
    # plot!(pl, x, u_test[:, iplot, i]; label = "FOM")
    plot!(pl, x, Φ * v_test[:, iplot, i]; label = "Projected FOM")
    plot!(pl, x, Φ * sol_tf[:, iplot, i]; label = "ROM neural closure (TF)")
    display(pl)
    # sleep(0.05)
end

pl = Plots.plot(;title = "Projection", xlabel = "x", ylims = extrema(v_train[1, iplot, :]))
plot!(pl, t_train, v_train[1, iplot, :]; label = "Projected FOM")
plot!(pl, t_train, sol_tf[1, iplot, :]; label = "ROM neural closure (TF)")