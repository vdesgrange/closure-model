module PODGP

using LinearAlgebra

function pod_gp(X)
  F = svd(X; full=false);
end

function pod_eigenvalue(X)
    # Get covariance matrix
    # Solve eigenproblem
    # Sort eigen values and vectors
end

end
