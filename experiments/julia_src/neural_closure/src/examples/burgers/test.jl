using BenchmarkTools
using Flux
using Optimization
using OptimizationOptimisers
using DiffEqSensitivity
using IterTools: ncycle
using Plots;
using BSON: @save, @load

include("../../utils/generators.jl");
include("../../equations/burgers/burgers_gp.jl")
include("../../utils/graphic_tools.jl")
include("../../rom/pod.jl")
include("../../utils/processing_tools.jl")
include("../../neural_ode/models.jl")
include("../../neural_ode/regularization.jl")

tₘₐₓ= 1.;
tₘᵢₙ = 0.;
xₘₐₓ = 1.;
xₘᵢₙ = 0;
tₙ = 64;
xₙ = 64;
ν = 0.001;
x = LinRange(xₘᵢₙ, xₘₐₓ, xₙ);
t = LinRange(tₘᵢₙ, tₘₐₓ, tₙ);
Δx = (xₘₐₓ / (xₙ - 1));
Δt = (tₘₐₓ / (tₙ - 1));
m = 100;
snap_kwarg= repeat([(; tₘₐₓ, tₘᵢₙ, xₘₐₓ, xₘᵢₙ, tₙ, xₙ, ν, typ=5)], 256);
init_kwarg = repeat([(; mu=100)], 256);
dataset = Generator.generate_closure_dataset(256, 4, "./viscous_burgers_high_dim_t0.5_128_x1_64_nu0.001_typ2_m100_8_up2_j173.jld2", snap_kwarg, init_kwarg);
for (i, data) in enumerate(dataset)
  t, u, _, _ = data;
  display(Plots.plot(x, u[:, 1]; label = "u₀"));
  display(GraphicTools.show_state(u, t, x, "", "t", "x"))
end

# dataset = Generator.read_dataset("kdv_high_dim_m25_t10_128_x30_64_up8.jld2")["training_set"];

dataset = Generator.read_dataset("./dataset/viscous_burgers_high_dim_t2_64_xpi_64_nu0.04_typ2_m10_256_up16_j173.jld2")["training_set"];
for (i, data) in enumerate(dataset)
  t, u, _, _ = data;
  # display(GraphicTools.show_state(u, t, x, "", "t", "x"))
end

# === Test ===
begin
  tmp = []
  for (i, data) in enumerate(dataset)
    push!(tmp, data[2]);
  end
  u_ref = cat(tmp...; dims=3);
  # size(u_ref)

  # u_ref = dataset[1][2]
  umean = mean(reshape(Array(u_ref[:, :, 1]), xₙ, :); dims = 2)
  bas, _ = POD.generate_pod_svd_basis(reshape(Array(u_ref), xₙ, :), false);
  λ = bas.eigenvalues;
  POD.get_energy(λ, m)
  Φ = bas.modes[:, 1:m];

  t, u₀, u = Generator.get_burgers_batch(2., 0., pi, 0., 64, 64, 0.04, 2, (; m=5));

  display(GraphicTools.show_state(u, t, x, "", "t", "x"))
  # display(GraphicTools.show_state(Φ * Φ' * u, t, x, "", "t", "x"))
  # display(GraphicTools.show_state(umean .+ Φ * Φ' * (u .- umean), t, x, "", "t", "x"))
  gp = galerkin_projection(t, u, Φ, 0.04, Δx, Δt);
  display(GraphicTools.show_state(gp, t, x, "", "t", "x"))

  size(Φ)
  plot(x, Φ[:, 1:3])
  Plots.heatmap(Φ' * Φ; xmirror = true, yflip = true)
  Plots.heatmap(Φ * Φ'; xmirror = true, yflip = true)
  Plots.heatmap(I(64); xmirror = true, yflip = true)


  for (i, t) ∈ enumerate(t)
    pl = Plots.plot(; xlabel = "x")
    Plots.plot!(pl, x, u[:, i]; label = "Reference")
    Plots.plot!(pl, x, Φ * Φ' * u[:, i]; label = "POD")
    Plots.plot!(pl, x, gp[:, i]; label = "Galerkin Projection")
    display(pl)
    # sleep(0.05) # Time for plot pane to update
  end
end

function f(u, p, t)
  ν = p[1]
  Δx = p[2]
  u₋ = circshift(u, 1)
  u₊ = circshift(u, -1)
  a₊ = u₊ + u
  a₋ = u + u₋
  du = @. -((a₊ < 0) * u₊^2 + (a₊ > 0) * u^2 - (a₋ < 0) * u^2 - (a₋ > 0) * u₋^2) / Δx + ν * (u₋ - 2u + u₊) / Δx^2
  du
end

  # === Check results ===

using Plots
include("../../utils/graphic_tools.jl")

dataset = Generator.read_dataset("./dataset/viscous_burgers_high_dim_t2_64_xpi_64_nu0.04_typ2_m10_256_up16_j173.jld2")["training_set"];
using BSON
BSON.@load "./models/cnn_viscous_256_2/viscous_burgers_high_dim_m10_256_500epoch_model2_j173.bson" K p
t, u₀, u = Generator.get_burgers_batch(4 * tₘₐₓ, tₘᵢₙ, 4 * xₘₐₓ, xₘᵢₙ, 4 * tₙ, 4 * xₙ, ν, 2, (; m=10));
_prob = ODEProblem((u, p, t) -> K(u), reshape(u₀, :, 1, 1), extrema(t), p; saveat=t);
û = Array(solve(_prob, Tsit5(), reltol=1e-6, abstol=1e-6, sensealg=DiffEqSensitivity.InterpolatingAdjoint(; autojacvec=ZygoteVJP())));

function test1()
  _ref = ODEProblem(f, u₀, extrema(t), (ν, Δx); saveat=t);
  ū = Array(solve(_ref, Tsit5(), reltol=1e-6, abstol=1e-6, sensealg=DiffEqSensitivity.InterpolatingAdjoint(; autojacvec=ZygoteVJP()))); 
end;
@btime test1();
@profview test1();

function test2()
_prob = ODEProblem((u, p, t) -> K(u), reshape(u₀, :, 1, 1), extrema(t), p; saveat=t);
  û = Array(solve(_prob, Tsit5(), reltol=1e-6, abstol=1e-6, sensealg=DiffEqSensitivity.InterpolatingAdjoint(; autojacvec=ZygoteVJP())));
end;
@btime test2();
@profview test2();

display(GraphicTools.show_state(u, t, x, "", "t", "x"))
display(GraphicTools.show_state(ū, t, x, "", "t", "x"))
display(GraphicTools.show_state(û[:, 1, 1, :], 2 * t, 2 * x, "", "t", "x"))
display(GraphicTools.show_err(u, û[:, 1, 1, :], 2 * t, 2 * x, "", "t", "x"))
  
# # display(GraphicTools.show_state(Φ * v̄[:, 1, 1, :], t, x, "", "t", "x"))
  # # display(GraphicTools.show_state(Φ * v̂, t, x, "", "t", "x"))

  # for (i, t) ∈ enumerate(t)
  #   pl = Plots.plot(; xlabel = "x", ylim=extrema(v))
  #   Plots.plot!(pl, x, v[:, i]; label = "FOM - v - Model 0")
  #   # Plots.plot!(pl, x, v̂[:, i]; label = "ROM - v̂ - Model 1")
  #   Plots.plot!(pl, x, v̄[:, 1, 1, i]; label = "ROM - v̄ - Model 2")
  #   display(pl)
  # end

  # for (i, t) ∈ enumerate(t)
  #   pl = Plots.plot(; xlabel = "x", ylim=extrema(v))
  #   Plots.plot!(pl, x, u[:, i]; label = "FOM - Model 0")
  #   Plots.plot!(pl, x, û[:, i]; label = "ROM - Model 1")
  #   Plots.plot!(pl, x, Φ * v̄[:, 1, 1, i]; label = "ROM - Model 2")
  #   display(pl)
  # end



# === Check results ===
u = train_dataset[1][2];
u₀ = u[:, 1];
v = Φ' * u;
v₀ = v[:, 1];
û = gp_set[1][2]; # include Φ

_prob = ODEProblem(K, reshape(v₀, :, 1, 1), extrema(t), θ; saveat=t);
v̄ = solve(_prob, Tsit5());
# v̄ = Array(solve(_prob, Tsit5(), p=θ, sensealg=DiffEqSensitivity.InterpolatingAdjoint(; autojacvec=ZygoteVJP())));
size(v̄)
display(GraphicTools.show_state(Φ * v, t, x, "", "t", "x"))
display(GraphicTools.show_state(Φ * v̄[:, 1, 1, :], t, x, "", "t", "x"))
display(GraphicTools.show_state(Φ * v̂, t, x, "", "t", "x"))

for (i, t) ∈ enumerate(t)
  pl = Plots.plot(; xlabel = "x", ylim=extrema(v))
  Plots.plot!(pl, x, u[:, i]; label = "FOM - Model 0")
  Plots.plot!(pl, x, û[:, i]; label = "ROM - Model 1")
  Plots.plot!(pl, x, Φ * v̄[:, 1, 1, i]; label = "ROM - Model 2")
  display(pl)
end

t, u₀, u = Generator.get_burgers_batch(2., 0., pi, 0., 64, 64, 0.04, 2, (; m=5));
hcat([u₀, u₀]...)
_prob = ODEProblem(f, hcat([u₀, u₀]...), extrema(t), (ν,  Δx); saveat=t);
u = solve(_prob, Tsit5());
size(u)
display(GraphicTools.show_state(u[:, 1, :], t, x, "", "t", "x"))

# ===== Compare initial conditions ====
function f(u, (ν, Δx), t)
  u₋ = circshift(u, 1)
  u₊ = circshift(u, -1)
  a₊ = u₊ + u
  a₋ = u + u₋
  du = @. -((a₊ < 0) * u₊^2 + (a₊ > 0) * u^2 - (a₋ < 0) * u^2 - (a₋ > 0) * u₋^2) / Δx +
     ν * (u₋ - 2u + u₊) / Δx^2
  du
end


using Distributions
K = 50;
u₀ = Generator.InitialFunctions.high_dim_random_init3(t, x, K);
display(Plots.plot(x, u₀[1, :]; label = "u₀"));
u = Array(predict(f, u₀[1, :], (ν, Δx), t, Tsit5()));
display(GraphicTools.show_state(u, t, x, "", "t", "x"))

function create_data_tmp(f, p, K, x, nsolution, t; decay = k -> 1 / (1 + abs(k)))
  L = x[end]
  basis = [exp(2π * im * k * x / L) for x ∈ x, k ∈ -K:K]
  c = [randn() * decay(k) * exp(-2π * im * rand()) for k ∈ -K:K, _ ∈ 1:nsolution]
  u₀ = real.(basis * c)
  predict(f, u₀, p, t, Tsit5())
end

u3 = Array(create_data_tmp(f, (ν, Δx), K, x, 1, t));
display(Plots.plot(x, u3[:, 1, 1]; label = "u₀"));
display(GraphicTools.show_state(u3[:, 1, :], t, x, "", "t", "x"))