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

# BSON.@load "./models/cnn_viscous_256_2/viscous_burgers_high_dim_m10_256_500epoch_model2_j173.bson" K p
BSON.@load "./models/downscaling/viscous_burgers_high_dim_t2_64_xpi_64_nu0.04_typ2_m10_256_up16_j173.bson" K p
t, u₀, u = Generator.get_burgers_batch(2, 0., pi, 0., 64, 64, 0.04, 2, (; m=10));
_prob = ODEProblem((u, p, t) -> K(u), reshape(u₀, :, 1, 1), extrema(t), p; saveat=t);
û = Array(solve(_prob, Tsit5(), reltol=1e-6, abstol=1e-6, sensealg=DiffEqSensitivity.InterpolatingAdjoint(; autojacvec=ZygoteVJP())));
vx = LinRange(0, pi, 64);

display(GraphicTools.show_err(u, û[:, 1, 1, :], t, vx, L"Error\ field\ |v_θ(t) − v(t)|", "t", "x"))
display(GraphicTools.show_state(u, t, vx, L"Low-resolution with FOM model", "t", "x"))
display(GraphicTools.show_state(û[:, 1, 1, :], t, vx, L"ROM neural closure (TF)", "t", "x"))

# Reference / Prediction NODE
for (i, t) ∈ enumerate(t)
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
        plot!(plt, x, û[:, 1, 1, i], linestyle=:dash, label="ROM NODE (TF)")
        plot!(plt; xlims=(0., pi), ylims=(-2, 2.), dpi=600)
    end
    gif(anim, "downscaling_pure_node_viscous_burgers_high_dim_t2_64_xpi_64_nu0.04_typ2_m10_256_up16_fps15.gif", fps = 15)
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