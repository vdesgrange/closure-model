module Objectives



function mseloss(u_pred, u_true)
  return sum(abs2, u_pred .- u_true) / prod(size(u_true))
end

function seloss(u_pred, u_true)
  return sum(abs2, u_pred .- u_true))
end

end
