module GraphicTools

using Plots


function simple_plotter(ks, title="Simple plot", L=1.)
  pl = Plots.plot();
  for k in ks
    x = LinRange(0, L, size(k, 1))
    Plots.plot!(pl, x, k);
  end

end

function plot_regularization(noise, reg, a, b, c, d)
    plot(dpi=200)
    plot!(reg, a, markershape=:x, label="Interpolation")
    plot!(reg, b, markershape=:x, label="Full")
    plot!(reg, c, markershape=:x, label="Training")
    plot!(reg, d, markershape=:x, label="Validation")
    plot!(xlabel="Regularization", ylabel="Cost", xaxis=:log)
    plot!(xticks=(reg,["1e-12", "1e-10", "1e-8", "1e-6", "1e-5", "1e-4", "1e-3", "1e-2", "1e-1"]))
end

function show_state(u, x, y, title, xlabel, ylabel)
    """
        show_state(u, x, y, title, xlabel, ylabel=)
        Display heat map of a matrix

    # Arguments
    - `u`: Matrix
    - `x`: x axis values
    - `y`: y axis values
    - `title::String`: title of the heatmap
    - `xlabel::String`: x axis label
    - `ylabel::String`: y axis label
    """

    xₙ, yₙ = size(x)[1], size(y)[1];
    xₘᵢₙ, xₘₐₓ = x[1], x[end];
    yₘᵢₙ, yₘₐₓ = y[1], y[end];

    xformatter = x -> string(round(x / xₙ * xₘₐₓ + xₘᵢₙ, digits=2));
    yformatter = y -> string(round(y / yₙ * yₘₐₓ + yₘᵢₙ, digits=2));

    pl = heatmap(u);
    heatmap!(pl;
        title = title,
        # dpi=600,
        # aspect_ratio = :equal,
        # reuse=false,
        # c=:dense,
        # grid=:none,
        xlabel=xlabel,
        ylabel=ylabel,
        # xticks=(1:7:size(x)[1], [xformatter(x) for x in 0:7:size(x)[1]]),
        # yticks=(1:7:size(y)[1], [yformatter(y) for y in 0:7:size(y)[1]]),
    );

    return pl;
end

function show_err(u, û, x, y, title, xlabel, ylabel)
    """
        show_err(u, û, x, y, title, xlabel, ylabel)
        Display absolute difference between 2 matrices.

    # Arguments
    - `u`: Matrix
    - `û`: Matrix
    - `x`: x axis values
    - `y`: y axis values
    - `title::String`: title of the heatmap
    - `x::String`: x label
    - `y::String`: y label
    """
    xₙ, yₙ = size(x)[1], size(y)[1];
    xₘᵢₙ, xₘₐₓ = x[1], x[end];
    yₘᵢₙ, yₘₐₓ = y[1], y[end];

    xformatter = x -> string(round(x / xₙ * xₘₐₓ + xₘᵢₙ, digits=2));
    yformatter = y -> string(round(y / yₙ * yₘₐₓ + yₘᵢₙ, digits=2));

    pl = heatmap(abs.(u .- û), c=:dense);
    heatmap!(pl,
        title = title,
        # dpi=600,
        # aspect_ratio = :equal,
        # reuse=false,
        xlabel=xlabel,
        ylabel=ylabel,
        # xticks=(1:7:size(x)[1], [xformatter(x) for x in 0:7:size(x)[1]]),
        # yticks=(1:7:size(y)[1], [yformatter(y) for y in 0:7:size(y)[1]]),
    );

    return pl;
end

end
