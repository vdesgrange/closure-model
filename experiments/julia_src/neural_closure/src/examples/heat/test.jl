using OrdinaryDiffEq
using SciMLSensitivity
using Plots
using PyPlot
using SparseArrays
using Distributions

include("../../utils/graphic_tools.jl");

function f(u, p, t)
    Δx = p[1];
    κ = p[2];
    u₋ = circshift(u, 1);
    u₊ = circshift(u, -1);
    u₋[1] = 0; # -1
    u₊[end] = 0; # 1
    uₜ = κ / (Δx^2) .* (u₊ .- 2 .* u .+ u₋);
    return uₜ
end

xₙ = 64;
xₘₐₓ = 1.;
xₘᵢₙ = 0.;
tₙ = 64;
tₘₐₓ = 2;
tₘᵢₙ = 0.;
L = xₘₐₓ - xₘᵢₙ;
N = 2;
k = 1:N;
x = LinRange(xₘᵢₙ, xₘₐₓ, xₙ);
t = LinRange(tₘᵢₙ, tₘₐₓ, tₙ);
Δx = (xₘₐₓ - xₘᵢₙ) / (xₙ - 1);

c = Array(randn(N) ./ k);
aₖ(k, x) = sqrt(2 / L) * sin(π * k * x / L);
u(x, t) = sum(cₖ * exp(- π^2 * k^2 * t) * aₖ(k, x) for (cₖ, k) in zip(c, k))
uᵣ = Array([u(a, b) for a in x, b in t]) # Reference solution for u(0,t) = u(1,t) = 0;
u₀ = uᵣ[:, 1]

# u₀ = [min(1, max(-1, 10*x-5)) for x in x]
plt = Plots.plot(x, u₀; label = "u(t, x)", xlabel="x", dpi=600)


prob = ODEProblem(ODEFunction(f), u₀, extrema(t), (Δx, 0.01));
sol = solve(prob, Tsit5(), saveat=t); #  dt=0.01
û = Array(sol);

for (i, t) ∈ enumerate(sol.t)
    pl = Plots.plot(; xlabel = "x", ylims = extrema(û[:, :]));
    Plots.plot!(pl, x, û[:, i]; label = "u(t, x)", xlabel="x")
    display(pl)
end

plt = Plots.plot(x, û[:, 1]; label = "u(t, x)", xlabel="x", dpi=600, ylims = extrema(1.05 * û[:, :]))
Plots.savefig(plt, "heat_u_t0_forward.png")
plt = Plots.plot(x,  û[:, end]; label = "u(t, x)", xlabel="x", dpi=600, ylims = extrema(1.05 * û[:, :]))
Plots.savefig(plt, "heat_u_t2_forward.png")


GraphicTools.show_state(û, sol.t, x, "", "t", "x")
# GraphicTools.show_state(uᵣ, t, x, "", "t", "x")

uₘₐₓ = û[:, end];
reverse(extrema(sol.t))
rev_prob = ODEProblem(ODEFunction(f), uₘₐₓ, reverse(extrema(sol.t)), (Δx, 0.01));
rev_sol = solve(rev_prob, Rodas4P());
ū = Array(rev_sol);

plt = Plots.plot(x, ū[:, 10]; label = "u(t, x)", xlabel="x", ylims=extrema(û[:, :]), dpi=600)
Plots.savefig(plt, "heat_u_t1.94_backward.png")
plt = Plots.plot(x, ū[:, 22]; label = "u(t, x)", xlabel="x", dpi=600)
Plots.savefig(plt, "heat_u_t1.90_backward.png")
rev_sol.t[22]

for (i, t) ∈ enumerate(rev_sol.t[1:10])
    pl = Plots.plot(; xlabel = "x", ylims = extrema(û[:, :]));
    Plots.plot!(pl, x, ū[:, i]; label = "Reverse")
    display(pl)
end


# ==== Generate data set ====

include("../../utils/generators.jl");

snap_kwarg =(; t_max=1., t_min=0., x_max=1., x_min=0., t_n=64, x_n=64, typ=3);
init_kwarg = (; κ=0.001, N=15);
dataset = Generator.generate_heat_dataset(1, 1, "dataset/diffusion_n256_k0.01_N15_analytical_t1_64_x1_64_up16.jld2", snap_kwarg, init_kwarg);
# dataset = Generator.read_dataset("dataset/diffusion_n256_k0.01_N15_analytical_t1_64_x1_64_up16.jld2")["training_set"];

t, uₗ, uₕ, _, _ = dataset[1];
GraphicTools.show_state(uₗ, t, x, "", "t", "x")

prob = ODEProblem(ODEFunction(f), uₗ[:, 1], extrema(t), (Δx, 0.001));
sol = solve(prob, Tsit5(), dt=0.01, saveat=t); #  dt=0.01
GraphicTools.show_state(Array(sol), t, x, "", "t", "x")
GraphicTools.show_err(uₗ, Array(sol), t, x, "", "t", "x")


for (i, t) ∈ enumerate(t)
    pl = Plots.plot(; xlabel = "x");
    Plots.plot!(pl, x, uₗ[:, i]; label = "u(t, x)", xlabel="x")
    display(pl)
end

# ==== Check results of NODE training ====

@load "./models/diffusion_k0.01_N15_analytical_t1_64_x1_64_ep500_traj.bson" K p


function check_result(K, θ)
    t, u₀, ū = Generator.get_heat_batch(2., 0., 1., 0., 64, 64, 3, (;κ = 0.01, N=15));
    _prob = ODEProblem((u, p, t) -> K(u), x, extrema(t), p, saveat=t);
    ȳ = solve(_prob, sol, u0=u₀, p=θ, sensealg=InterpolatingAdjoint(; autojacvec=ZygoteVJP()));  # BacksolveAdjoint work
    ȳ = Array(ȳ);

    Plots.plot(
        GraphicTools.show_state(ū, t, x, "", "t", "x"),
        GraphicTools.show_state(Array(ȳ), t, x, "", "t", "x"),
        GraphicTools.show_err(Array(ȳ), ū, t, x, "", "t", "x");
        layout = (1, 3),
    )
    # savefig("heat_direct_results.png");
end

check_result(K, p)
# end

x = LinRange(0., 1., 64);
t, u₀, ū = Generator.get_heat_batch(2., 0., 1., 0., 64, 64, 3, (;κ = 0.01, N=15));
_prob = ODEProblem((u, p, t) -> K(u), x, extrema(t), p, saveat=t);
ȳ = solve(_prob, sol, u0=u₀, p=p, dt=0.001, sensealg=InterpolatingAdjoint(; autojacvec=ZygoteVJP()));
ȳ = Array(ȳ);

gr(size=(600,525))
plt = GraphicTools.show_state(ū, t, x, "", "t", "x")
Plots.png("diffusion_reference_k0.01_N15_t1_x1.png");

gr(size=(600,525))
plt = GraphicTools.show_state(Array(ȳ), t, x, "", "t", "x")
Plots.savefig(plt, "diffusion_prediction_k0.01_N15_t1_x1.png");

gr(size=(600,520))
plt = GraphicTools.show_err(Array(ȳ), ū, t, x, "", "t", "x")
Plots.savefig(plt, "diffusion_difference_k0.01_N15_t1_x1.png");

gr(size=(600,600))
plt = heatmap(reshape(p, (64, 64)));
heatmap!(plt,
    dpi=600,
    aspect_ratio = :equal,
    reuse=false,
    c=:dense,
    yflip=true
)
Plots.savefig(plt, "diffusion_learned_operator.png");
