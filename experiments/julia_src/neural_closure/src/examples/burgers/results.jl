using BSON
include("../../equations/equations.jl");
include("closure_training.jl")

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
BSON.@load "./models/downscaling/viscous_burgers_high_dim_t2_64_xpi_64_nu0.04_typ2_m10_256_up16_j173.bson" K p
t, u₀, u = Generator.get_burgers_batch(2, 0., pi, 0., 64, 64, 0.04, 2, (; mu=10));
# û = downsampling(reshape(u, (64, 1, 64)), 1)[:, 1, :];

# Inviscid
# BSON.@load "./models/pure_node_inviscid/old_model_inviscid_burgers_fouriers_t2_64_xpi_64_nu0_typ5_K200_256_up8_j173.bson" K p
#t, u₀, u = Generator.get_burgers_batch(2, 0., pi, 0., 64, 64, 0.04, 5, (; m=200));

_prob = ODEProblem((u, p, t) -> K(u), reshape(u₀, :, 1, 1), extrema(t), p; saveat=t);
û = Array(solve(_prob, Tsit5(), reltol=1e-6, abstol=1e-6, sensealg=DiffEqSensitivity.InterpolatingAdjoint(; autojacvec=ZygoteVJP())));
vx = LinRange(0, pi, 64);
display(GraphicTools.show_err(u, û[:, 1, 1, :], t, vx, L"Error\ field\ |v_θ(t) − v(t)|", "t", "x")) # û[:, 1, 1, :] if old CNN2 model
pl_down = GraphicTools.show_state(u, t, vx, "Low-resolution with FOM model", "t", "x");
# savefig(pl_down, "downscaled_inviscid_burgers.png")
pl_nn = GraphicTools.show_state(û[:, 1, :], t, vx, "ROM NODE (TF)", "t", "x"); # û[:, 1, 1, :] if old CNN2 model (additional dimension outside)
# savefig(pl_nn, "pure_node_inviscid_burgers.png")
# Reference / Prediction NODE
for (i, t) ∈ collect(enumerate(t))[1:8:end]
    pl = Plots.plot(;title = @sprintf("Solution, t = %.3f", t), xlabel = "x", ylims = extrema(u[:, :]))
    plot!(pl, vx, u[:, i]; label = "FOM reference")
    plot!(pl, vx, û[:, 1, 1, i]; label = "ROM NODE (TF)")
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

t, u₀, u = Generator.get_burgers_batch(2, 0., pi, 0., 1024, 1024, 0.04, 4, (; σ=0.25, μ=1.));
û = downsampling(reshape(u, (1024, 1, 1024)), 16)[:, 1, :];

u₀ = û[:, 1];
t = collect(LinRange(0., 2., 64));
Δx = (pi - 0.) / (64 - 1);
k = 2 * pi * AbstractFFTs.fftfreq(64, 1. / Δx);

f_godunov = Equations.f_godunov
_prob = ODEProblem(f_fft, u₀, extrema(t), (0.04, Δx); saveat=t);
ū = Array(solve(_prob, Tsit5(), reltol=1e-6, abstol=1e-6, sensealg=DiffEqSensitivity.InterpolatingAdjoint(; autojacvec=ZygoteVJP())));

pyplot()
GraphicTools.show_state(û, collect(LinRange(0., 2., 64)), collect(LinRange(0., pi, 64)), "Down-scaled FOM model", "t", "x")
GraphicTools.show_state(ū, collect(LinRange(0., 2., 64)), collect(LinRange(0., pi, 64)), "Baseline model", "t", "x")
GraphicTools.show_err(û, ū, collect(LinRange(0., 2., 64)), collect(LinRange(0., pi, 64)), "|Difference|", "t", "x")

savefig(GraphicTools.show_state(û, collect(LinRange(0., 2., 64)), collect(LinRange(0., pi, 64)), "Downscaled FOM model", "t", "x"), "viscous_gauss_downscaled_fom.png")
savefig(GraphicTools.show_state(ū, collect(LinRange(0., 2., 64)), collect(LinRange(0., pi, 64)), "Baseline model", "t", "x"), "viscous_gauss_baseline_coarse_model.png")
savefig(GraphicTools.show_err(û, ū, collect(LinRange(0., 2., 64)), collect(LinRange(0., pi, 64)), "|Difference|", "t", "x"), "viscous_gauss_difference_fom_coarse.png")


# === Closure term model (viscous) comparison  ===
coarse = 0;
nn = 0;
closure = 0;

# for i in 1:1
    pyplot()
    t, u₀, u = Generator.get_burgers_batch(2, 0., pi, 0., 256, 256, 0.04, 2, (; m=10));
    û = downsampling(reshape(u, (256, 1, 256)), 4)[:, 1, :];
    savefig(GraphicTools.show_state(û, collect(LinRange(0., 2., 64)), collect(LinRange(0., pi, 64)), "Downscaled FOM model", "t", "x"), "viscous_burgers_downscaled.png")

    u₀ = û[:, 1];
    t = collect(LinRange(0., 2., 64));
    vx = LinRange(0, pi, 64);
    Δx = (pi - 0.) / (64 - 1);
    k = 2 * pi * AbstractFFTs.fftfreq(64, 1. / Δx);

    _prob = ODEProblem(f_fft, u₀, extrema(t), (0.04, Δx); saveat=t);
    u_d = Array(solve(_prob, Tsit5(), reltol=1e-6, abstol=1e-6, sensealg=DiffEqSensitivity.InterpolatingAdjoint(; autojacvec=ZygoteVJP())));
    GraphicTools.show_state(u_d, collect(LinRange(0., 2., 64)), collect(LinRange(0., pi, 64)), "Baseline model", "t", "x")
    GraphicTools.show_err(u_d, û, collect(LinRange(0., 2., 64)), collect(LinRange(0., pi, 64)), "|Difference|", "t", "x")
    
    
    savefig(GraphicTools.show_state(u_d, collect(LinRange(0., 2., 64)), collect(LinRange(0., pi, 64)), "Baseline model", "t", "x"), "viscous_burgers_baseline.png")
    savefig(GraphicTools.show_err(u_d, û, collect(LinRange(0., 2., 64)), collect(LinRange(0., pi, 64)), "|Difference|", "t", "x"), "diff_viscous_baseline.png")
    
    coarse += Flux.mse(u_d, û);
    Flux.mse(u_d, û)

    BSON.@load "./models/downscaling/viscous_burgers_high_dim_t2_64_xpi_64_nu0.04_typ2_m10_256_up16_j173.bson" K p
    f_nn = (u, p, t) -> K(u);
    _prob = ODEProblem(f_nn, reshape(u₀, :, 1, 1), extrema(t), p; saveat=t);
    u_nn = Array(solve(_prob, Tsit5(), reltol=1e-8, abstol=1e-8, sensealg=DiffEqSensitivity.InterpolatingAdjoint(; autojacvec=ZygoteVJP())));
    nn += Flux.mse(u_nn[:, 1, 1, :], û)
    Flux.mse(u_nn[:, 1, 1, :], û)
    GraphicTools.show_state(u_nn[:, 1, 1, :], collect(LinRange(0., 2., 64)), collect(LinRange(0., pi, 64)), "Pure NODE", "t", "x")
    GraphicTools.show_err(u_nn[:, 1, 1, :], û, collect(LinRange(0., 2., 64)), collect(LinRange(0., pi, 64)), "|Difference|", "t", "x")
    
    savefig(GraphicTools.show_state(u_nn[:, 1, 1, :], collect(LinRange(0., 2., 64)), collect(LinRange(0., pi, 64)), "Pure NODE", "t", "x"), "viscous_burgers_node.png")
    savefig(GraphicTools.show_err(u_nn[:, 1, 1, :], û, collect(LinRange(0., 2., 64)), collect(LinRange(0., pi, 64)), "|Difference|", "t", "x"), "diff_viscous_node.png")
    

    BSON.@load "./models/downscaling/coarse_closure_viscous_burgers_high_dim_t2_64_xpi_64_nu0.04_typ2_m10_256_up16_j173.bson" K p
    f_nn = (u, p, t) -> K(u);
    function f_closure4(u, p, t)
        Equations.f_fft(u, (0.04, Δx, k), t) + f_nn(u, p, t);
    end
    _prob = ODEProblem(f_closure4, reshape(u₀, (:, 1, 1)), extrema(t), p; saveat=t);
    u_cl = Array(solve(_prob, Tsit5(), sensealg=DiffEqSensitivity.InterpolatingAdjoint(; autojacvec=ZygoteVJP())));
    GraphicTools.show_state(u_cl[:, 1, 1, :], collect(LinRange(0., 2., 64)), collect(LinRange(0., pi, 64)), "Closure NODE", "t", "x")
    GraphicTools.show_err(u_cl[:, 1, 1, :], û, collect(LinRange(0., 2., 64)), collect(LinRange(0., pi, 64)), "|Difference|", "t", "x")
    
    savefig(GraphicTools.show_state(u_cl[:, 1, 1, :], collect(LinRange(0., 2., 64)), collect(LinRange(0., pi, 64)), "Closure NODE", "t", "x"), "viscous_burgers_closure.png")
    savefig(GraphicTools.show_err(u_cl[:, 1, 1, :], û, collect(LinRange(0., 2., 64)), collect(LinRange(0., pi, 64)), "|Difference|", "t", "x"), "diff_viscous_burgers_closure.png")
    
    closure += Flux.mse(u_cl[:, 1, 1, :], û);
    Flux.mse(u_cl[:, 1, 1, :], û)
# end

pyplot()
for (i, t) ∈ collect(enumerate(t))[1:8:end]
    pl = Plots.plot(;title = @sprintf("Solution, t = %.3f", t), xlabel = "x", ylims = extrema(u[:, :]))
    plot!(pl, vx, û[:, i]; label = "FOM reference")
    plot!(pl, vx, u_nn[:, 1, 1, i]; label = "ROM NODE (TF)")
    display(pl)
    sleep(0.05)
end


coarse / 10
nn / 10
closure / 10

dis_coarse = 0;
dis_nn = 0;
dis_closure = 0;

for i in 1:10
    pyplot()
    t, u₀, u = Generator.get_burgers_batch(2, 0., pi, 0., 256, 256, 0.04, 2, (; m=10));
    û = downsampling(reshape(u, (256, 1, 256)), 4)[:, 1, :];
    # GraphicTools.show_state(û, collect(LinRange(0., 2., 64)), collect(LinRange(0., pi, 64)), "Downscaled FOM model", "t", "x")

    u₀ = û[:, 1];
    t = collect(LinRange(0., 2., 64));
    vx = LinRange(0, pi, 64);
    Δx = (pi - 0.) / (64 - 1);
    k = 2 * pi * AbstractFFTs.fftfreq(64, 1. / Δx);

    _prob = ODEProblem(fflux2, u₀, extrema(t), (0., Δx); saveat=t);
    u_d = Array(solve(_prob, Euler(), dt=0.01, sensealg=DiffEqSensitivity.InterpolatingAdjoint(; autojacvec=ZygoteVJP())));
    # GraphicTools.show_state(u_d, collect(LinRange(0., 2., 64)), collect(LinRange(0., pi, 64)), "Baseline model", "t", "x")
    dis_coarse += Flux.mse(u_d, û);

    BSON.@load "./models/downscaling/viscous_burgers_high_dim_t2_64_xpi_64_nu0.04_typ2_m10_256_up16_j173.bson" K p
    f_nn = (u, p, t) -> K(u);
    _prob = ODEProblem(fflux2, reshape(u₀, :, 1, 1), extrema(t), p; saveat=t);
    u_nn = Array(solve(_prob, Euler(), dt=0.01, sensealg=DiffEqSensitivity.InterpolatingAdjoint(; autojacvec=ZygoteVJP())));
    # GraphicTools.show_state(u_nn[:, 1, 1, :], collect(LinRange(0., 2., 64)), collect(LinRange(0., pi, 64)), "Pure NODE", "t", "x")
    dis_nn += Flux.mse(u_nn[:, 1, 1, :], û)

    BSON.@load "./models/downscaling/coarse_closure_viscous_burgers_high_dim_t2_64_xpi_64_nu0.04_typ2_m10_256_up16_j173.bson" K p
    f_nn = (u, p, t) -> K(u);
    function f_closure4(u, p, t)
        f_fft(u, (0.04, Δx), t) + f_nn(u, p, t);
    end
    _prob = ODEProblem(f_closure4, reshape(u₀, (:, 1, 1)), extrema(t), p; saveat=t);
    u_cl = Array(solve(_prob, Euler(), dt=0.01, sensealg=DiffEqSensitivity.InterpolatingAdjoint(; autojacvec=ZygoteVJP())));
    # GraphicTools.show_state(u_cl[:, 1, 1, :], collect(LinRange(0., 2., 64)), collect(LinRange(0., pi, 64)), "Closure NODE", "t", "x")
    dis_closure += Flux.mse(u_cl[:, 1, 1, :], û);
end

dis_coarse / 10
dis_nn / 10
dis_closure / 10

# for i in 1:5

# p, re = Flux.destructure(model);


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


# ==== Compare offline operator computing ====

offline = [];
online = [];

B_k, L_k, N_k = Equations.offline_op(sm, Φ, ν, Δx);
@benchmark begin
    a₀_pod = Φ' * (u₀ .- sm);
    A = predict(Equations.gp,  a₀_pod, (B_k, L_k, N_k), t_pod, Tsit5());
end

## GP 2, f = fft
gg(a, p, t) = Φ' * ff(Φ * a + sm, p, t);
@benchmark begin
    a₀_pod = Φ' * (u₀ .- sm);
    A = predict(gg,  a₀_pod, (ν, Δx, k), t_pod, Tsit5());
end

# === Maulik ===

using JLD2
# rel_err_per_epochs = JLD2.load("./results/maulik_time_tf_errors_per_epoch_250ep_8b_rel.jld2")["relative_error"];
# mse_per_epochs = JLD2.load("./results/maulik_time_tf_errors_per_epoch_250ep_8b_mse.jld2")["mse"];
rel_err_per_epochs = JLD2.load("./results/maulik_time_tf_errors_per_epoch_250ep_8b_rel_mean_substracted.jld2")["relative_error"];
mse_per_epochs = JLD2.load("./results/maulik_time_tf_errors_per_epoch_250ep_8b_mse_mean_substracted.jld2")["mse"];


iters = collect(1:1:250);
avg = mean(rel_err_per_epochs; dims=2); # [:, 2:end] if size is 6
stds = std(rel_err_per_epochs; dims=2);
plot(iters, avg; ribbon = (stds, stds), xlabel = "Epoch", title = "Relative error", dpi=600, yscale=:log10, legend=:none)
savefig(plot(iters, avg; ribbon = (stds, stds), xlabel = "Epoch", ylabel="Relative error", title = "Training loss", dpi=600, yscale=:log10, legend=:none), "relative_error_maulik_time_tf_mean_substracted.png")


mse_avg = mean(mse_per_epochs; dims=2);
mse_stds = std(mse_per_epochs; dims=2);
savefig(plot(iters, mse_avg; ribbon = (mse_stds, mse_stds), xlabel = "Epoch", ylabel="MSE", title = "Training loss", dpi=600, yscale=:log10, legend=:none), "mse_maulik_time_tf_mean_substracted.png")

