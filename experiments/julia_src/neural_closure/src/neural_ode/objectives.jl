module Objectives

function l2loss(u_pred, u)
  return sqrt(sum(abs2, u_pred .- u) .^ 2)
end

end
