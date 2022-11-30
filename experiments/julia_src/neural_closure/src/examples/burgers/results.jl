using BSON
include("../../equations/equations.jl");
include("burgers_closure_galerkin_2.jl")

function animate_snapshot_prediction(u_pred, u, x, filename)
    t_n = 64
   
    anim = @animate for i ∈ 1:t_n
        plt = plot(x, u[:, i], label="u")
        plot!(plt, x, u_pred[:, i], linestyle=:dash, label="û")
        plot!(plt; xlims=(0., pi), ylims=(-2, 2.), dpi=600)
    end

    gif(anim, filename, fps = 15)
end
# === Old model ===

# Viscous
# BSON.@load "./models/cnn_viscous_256_2/viscous_burgers_high_dim_m10_256_500epoch_model2_j173.bson" K p
# BSON.@load "./models/downscaling/viscous_burgers_high_dim_t2_64_xpi_64_nu0.04_typ2_m10_256_up16_j173.bson" K p
t, u₀, u = Generator.get_burgers_batch(2, 0., pi, 0., 64, 64, 0.04, 2, (; m=10));

# Inviscid
BSON.@load "./models/pure_node_inviscid/old_model_inviscid_burgers_fouriers_t2_64_xpi_64_nu0_typ5_K200_256_up8_j173.bson" K p
t, u₀, u = Generator.get_burgers_batch(2, 0., pi, 0., 64, 64, 0., 5, (; m=200));
_prob = ODEProblem((u, p, t) -> K(u), reshape(u₀, :, 1), extrema(t), p; saveat=t);
û = Array(solve(_prob, Tsit5(), reltol=1e-6, abstol=1e-6, sensealg=DiffEqSensitivity.InterpolatingAdjoint(; autojacvec=ZygoteVJP())));
vx = LinRange(0, pi, 64);
display(GraphicTools.show_err(u, û[:, 1, :], t, vx, L"Error\ field\ |v_θ(t) − v(t)|", "t", "x")) # û[:, 1, 1, :] if old CNN2 model
pl_down = GraphicTools.show_state(u, t, vx, "Low-resolution with FOM model", "t", "x");
savefig(pl_down, "downscaled_inviscid_burgers.png")
pl_nn = GraphicTools.show_state(û[:, 1, :], t, vx, "ROM NODE (TF)", "t", "x"); # û[:, 1, 1, :] if old CNN2 model (additional dimension outside)
savefig(pl_nn, "pure_node_inviscid_burgers.png")
# Reference / Prediction NODE
for (i, t) ∈ enumerate(t)
    pl = Plots.plot(;title = @sprintf("Solution, t = %.3f", t), xlabel = "x", ylims = extrema(u[:, :]))
    plot!(pl, vx, u[:, i]; label = "FOM reference")
    plot!(pl, vx, û[:, 1, i]; label = "ROM NODE (TF)")
    display(pl)
    sleep(0.05)
end

begin
    x = collect(LinRange(0., pi, 64))
    t_n = 64  
    anim = @animate for (i, t) ∈ enumerate(t)
        pl = Plots.plot(;title = @sprintf("Solution, t = %.3f", t), xlabel = "x", ylabel="u", ylims = extrema(u[:, :]))
        plt = plot(x, u[:, i], label="FOM")
        plot!(plt, x, û[:, 1, i], linestyle=:dash, label="ROM NODE (TF)")
        plot!(plt; xlims=(0., pi), ylims=(-2, 2.), dpi=600)
    end
    gif(anim, "downscaling_pure_node_inviscid_burgers_high_dim_t2_64_xpi_64_nu0_typ5_K200_256_up16_fps15.gif", fps = 15)
end

# Error field
for (i, t) ∈ enumerate(t) # All
    pl = Plots.plot(;title = @sprintf("Error field, t = %.3f", t), xlabel = "x", ylims = extrema(abs.(u[:, :] - û[:, 1, 1, :])))
    plot!(pl, vx, abs.(u[:, i] - û[:, 1, 1, i]); label = L"|v_θ(t) − v(t)|")
    display(pl)
    sleep(0.05)
end

begin # sample
    pl = Plots.plot(;title = L"Error\ field\ |v_θ(t) − v(t)|", xlabel = "x", ylims = extrema(u[:, :] - û[:, 1, 1, :]))
    for i ∈ [3, 17, 33, 48, 64]
        plot!(pl, vx, u[:, i] - û[:, 1, 1, i]; label = @sprintf("t = %.2f", t[i]))
    end
    display(pl)
end

# Relative error field
begin # Sample
    pl = Plots.plot(;title = L"Relative\ error\ field\ \frac{|v_θ(t) − v(t)|}{|v(t)|}", xlabel = "x")
    for i ∈ [3, 17, 33, 48, 64]
        plot!(pl, vx, abs.(û[:, 1, 1, i] .- u[:, i]) ./ abs.(u[:, i]); label = @sprintf("t = %.2f", t[i]))
    end
    display(pl)
end

# Evolution of scalar errors on testing dataset as a function of time
begin
    t, u₀, u = Generator.get_burgers_batch(2, 0., pi, 0., 64, 64, 0.04, 2, (; m=10));
    _prob = ODEProblem((u, p, t) -> K(u), reshape(u₀, :, 1, 1), extrema(t), p; saveat=t);
    û = Array(solve(_prob, Tsit5(), reltol=1e-6, abstol=1e-6, sensealg=DiffEqSensitivity.InterpolatingAdjoint(; autojacvec=ZygoteVJP())));
    vx = LinRange(0, pi, 64);
end


# === Coarse vs downscaling
include("../../utils/graphic_tools.jl")
using AbstractFFTs
using FFTW
gr()

function downsampling(u, d)
    (Nₓ, b, Nₜ) = size(u);
    Dₜ = Int64(Nₜ / d);
    u₁ = u[:, :, 1:d:end]; # Take t value every d steps
    Dₓ = Int64(Nₓ / d);
    u₂ = sum(reshape(u₁, d, Dₓ, b, Dₜ), dims=1);
    u₃   = reshape(u₂, Dₓ, b, Dₜ) ./ d;
    return u₃
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

t, u₀, u = Generator.get_burgers_batch(2, 0., pi, 0., 1024, 1024, 0.04, 5, (; m=10));
û = downsampling(reshape(u, (1024, 1, 1024)), 16)[:, 1, :];

u₀ = û[:, 1];
t = collect(LinRange(0., 2., 64));
Δx = (pi - 0.) / (64 - 1);
k = 2 * pi * AbstractFFTs.fftfreq(64, 1. / Δx);

_prob = ODEProblem(f_fft, u₀, extrema(t), (0.04, Δx); saveat=t);
ū = Array(solve(_prob, Tsit5(), reltol=1e-6, abstol=1e-6, sensealg=DiffEqSensitivity.InterpolatingAdjoint(; autojacvec=ZygoteVJP())));

gr()
savefig(GraphicTools.show_state(û, collect(LinRange(0., 2., 64)), collect(LinRange(0., pi, 64)), "Downscaled FOM model", "t", "x"), "downscaled_fom.png")
savefig(GraphicTools.show_state(ū, collect(LinRange(0., 2., 64)), collect(LinRange(0., pi, 64)), "Baseline model", "t", "x"), "baseline_coarse_model.png")
savefig(GraphicTools.show_err(û, ū, collect(LinRange(0., 2., 64)), collect(LinRange(0., pi, 64)), "|Difference|", "t", "x"), "difference_fom_coarse.png")


# === Closure term model (viscous) ===
BSON.@load "./models/downscaling/coarse_closure_500_viscous_burgers_high_dim_t2_64_xpi_64_nu0.04_typ2_m10_256_up16_j173.bson" K p

t, u₀, u = Generator.get_burgers_batch(2, 0., pi, 0., 1024, 1024, 0., 5, (; m=10));
û = downsampling(reshape(u, (1024, 1, 1024)), 16)[:, 1, :];
u₀ = û[:, 1];
t = collect(LinRange(0., 2., 64));
vx = LinRange(0, pi, 64);
Δx = (pi - 0.) / (64 - 1);

f_godunov(u₀, (ν, Δx2), t) + K(u₀)
p, re = Flux.destructure(model);
f_nn = (u, p, t) -> K(u);
function f_closure3(u, p, t)
    f_godunov(u, (ν, Δx), t) + f_nn(u, p, t)
end

_prob = ODEProblem(f_closure3, reshape(u₀, (:, 1, 1)), extrema(t), p; saveat=t);
ū = Array(solve(_prob, Tsit5(), sensealg=DiffEqSensitivity.InterpolatingAdjoint(; autojacvec=ZygoteVJP())));

GraphicTools.show_state(û, collect(LinRange(0., 2., 64)), collect(LinRange(0., pi, 64)), "Downscaled FOM model", "t", "x")
GraphicTools.show_state(ū[:, 1, 1, :], collect(LinRange(0., 2., 64)), collect(LinRange(0., pi, 64)), "Baseline model", "t", "x")
GraphicTools.show_err(û, ū[:, 1, 1, :], collect(LinRange(0., 2., 64)), collect(LinRange(0., pi, 64)), "|Difference|", "t", "x")

for (i, t) ∈ enumerate(t)
    pl = Plots.plot(;title = @sprintf("Solution, t = %.3f", t), xlabel = "x", ylims = extrema(u[:, :]))
    plot!(pl, vx, û[:, i]; label = "FOM reference")
    plot!(pl, vx, ū[:, 1, 1, i]; label = "ROM NODE (TF)")
    display(pl)
    sleep(0.05)
end
