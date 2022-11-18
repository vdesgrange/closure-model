using OrdinaryDiffEq
using Plots
using LinearAlgebra
using SparseArrays
using Random
using Distributions
using JLD2

include("../../equations/initial_functions.jl")
include("../../utils/graphic_tools.jl")

t_max=1.;
t_min=0.;
x_max=pi;
x_min=0.;
t_n=1024;
x_n=1024;
typ=2;
mu = 3;
t = LinRange(t_min, t_max, t_n);
x = LinRange(x_min, x_max, x_n);
Δx = (x_max - x_min) / (x_n - 1);
Δt = (t_max - t_min) / (t_n - 1);

# === Function ====
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

function f3(u, p, t)
  u₋₋₋ = circshift(u, 3);
  u₋₋ = circshift(u, 2);
  u₋ = circshift(u, 1);
  u₊ = circshift(u, -1);
  u₊₊ = circshift(u, -2);
  u₊₊₊ = circshift(u, -3);

  uₓ = (1/12 * u₋₋ - 2/3 * u₋ + 2/3 * u₊ - 1/12 * u₊₊) ./ Δx;
  uₓₓₓ = (1/8 * u₋₋₋ - u₋₋ + 13/8 * u₋ - 13/8 * u₊ + u₊₊ - 1/12 * u₊₊₊) ./ (Δx^3);
  uₜ = -(6u .* uₓ) .- uₓₓₓ;
  return uₜ
end

# === Gaussian filter
G(Δ, x) = √(6 / π) / Δ * exp(-6x^2 / Δ^2);
Δ = 4Δx;
W = sum(G.(Δ, x .- x' .- z) for z ∈ -2:2);
W = W ./ sum(W; dims = 2);

# === Test initial conditions ===
u0 = @. exp(-(x - 0.5)^2 / 0.005);
u0 = @. sinpi(2x) + sinpi(6x) + cospi(10x);
u0 = InitialFunctions.high_dim_random_init2(t, x, 3)[1, :];
Plots.plot(x, u0; label = "Unfiltered")

prob = ODEProblem(ODEFunction(f), copy(u0), extrema(t), (k));
sol = solve(prob, Rodas4P(), saveat=t, dt=0.01);
t2 = sol.t;
u = hcat(sol.u...);
Wsol = W * sol;
tmp = GraphicTools.show_state(u, t, x, "", "t", "x")
GraphicTools.show_state(Wsol, t, x, "", "t", "x")
GraphicTools.show_state(u .- Wsol, t, x, "", "t", "x")


# for (i, t) ∈ enumerate(t)
#   pl = Plots.plot(; xlabel = "x", ylims = extrema(sol[:, :]))
#   Plots.plot!(pl, x, sol[i]; label = "Unfiltered")
#   Plots.plot!(pl, x, Wsol[:, i]; label = "Filtered")
#   display(pl)
#   # sleep(0.05) # Time for plot pane to update
# end


# ==== Generate + Fix data sets for KdV

include("../../utils/generators.jl");

snap_kwarg =(; t_max, t_min, x_max, x_min, t_n, x_n, typ);
init_kwarg = (; mu);
t, u₀, u = Generator.get_kdv_batch(t_max, t_min, x_max, x_min, t_n, x_n, 2, (; m=3));

_ref = ODEProblem(f, u₀, extrema(t), (ν, Δx); saveat=t);
@time ū = Array(solve(_ref, Rodas4P(), sensealg=DiffEqSensitivity.InterpolatingAdjoint(; autojacvec=ZygoteVJP())));

# _prob = ODEProblem((u, p, t) -> K(u), reshape(u₀, :, 1, 1), extrema(t), p; saveat=t);
# @time û = Array(solve(_prob, Tsit5(), sensealg=DiffEqSensitivity.InterpolatingAdjoint(; autojacvec=ZygoteVJP())));


# dataset = Generator.generate_kdv_dataset(1, 1, "", snap_kwarg, init_kwarg);
display(GraphicTools.show_state(u, t, x, "Unfiltered", "t", "x"))
t, u , _, _ = dataset[128];

# fix = [];
# for (i, data) in enumerate(dataset2)
#   print(i)
#   if (size(data[2]) != (64, 128))
#     push!(fix, i);
#     println(i)
#     println(size(data[2]))
#   end
# end
# size(fix)
# for (i, j) in enumerate(fix)
#   println(j)
#   println(i)
#   println(size(dataset2[j][2]));
#   dataset2[j] = dataset3[i];
#   println(size(dataset2[j][2]));
# end

# JLD2.save("kdv_high_dim_n128_m3_t5_128_x4pi_64_up2.jld2", "training_set", dataset);

# ============

dataset = Generator.read_dataset("dataset/kdv_high_dim_n128_m3_t5_128_x4pi_64_up2.jld2")["training_set"];
for (i, data) ∈ enumerate(dataset[1:3])
  (t, u, _, _) = data;
  display(GraphicTools.show_state(u, t, x, "Unfiltered", "t", "x"))
  display(GraphicTools.show_state(W * u, t, x, "Filtered", "t", "x"))
end

t, u, _, _ = dataset[3];
for (i, t) ∈ enumerate(t)
  pl = Plots.plot(; xlabel = "x", ylims = extrema(u[:, :]))
  Plots.plot!(pl, x, u[:, i]; label = "Unfiltered")
  Plots.plot!(pl, x, W * u[:, i]; label = "Filtered")
  display(pl)
  # sleep(0.05) # Time for plot pane to update
end