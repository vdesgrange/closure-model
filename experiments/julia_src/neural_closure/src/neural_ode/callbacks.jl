module Callbacks

function callback_standalone(theta, loss, u_pred; plot_map=true)
    """
    DEPRECATED
    """
    display(loss)
    if plot_map
        IJulia.clear_output(true);
        display(
            plot(
                GraphicTools.show_state(u_pred, "Prediction"),
                GraphicTools.show_state(u_true, "Reference");
                layout = (1, 2),
            ),
        )
    end

  return false
end

function callback(A, loss)
  """
  DEPRECATED
  """
  println(loss);
  flush(stdout);
  false
end

end
