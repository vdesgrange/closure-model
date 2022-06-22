module Objectives

function l2(ŷ, y)
  _check_sizes(ŷ, y)
  sqrt(sum(abs2, ŷ .- y) .^ 2)
end

function energy(ŷ, y)
  _check_sizes(ŷ, y)
end

end
