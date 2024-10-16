# We need a version of the integration function that runs a single timestep and 
# takes as input the current prognostic fields. This function will then be
# passed to Enzyme for computing the Jacobian, which can subsequently be used
# in the Kalman filter. We're giving as input uveta, which will be a block
# vector of the fields u, v, and eta in that order. The restructuring of the
# arrays will be as columns stacked on top of eachother, e.g. the first column becomes
# the first bit of the vector, the second column the second bit, and so on.

using Parameters
Enzyme.API.looseTypeAnalysis!(true)
Enzyme.API.runtimeActivity!(true)

function u_mat_to_vec(u_mat)

    m, n = size(u_mat)
    u_vec = reshape(u_mat, (m*n))

    return u_vec

end

function v_mat_to_vec(v_mat)

    m, n = size(v_mat)
    v_vec = reshape(v_mat, (m*n))

    return v_vec

end

function eta_mat_to_vec(eta_mat)

    m, n = size(eta_mat)
    eta_vec = reshape(eta_mat, (m*n))

    return eta_vec

end

function compute_eigenvalues()

    evalues = []
    normalizedevecs = []

    x = randn(Complex{Float64}, 5, 1)

end

eigvals = []
normalizedeigvecs = []
unnormalizedeigvecs = []
temp = zeros(6,1)

A = randn(5,5)
x = randn(Complex{Float64}, 5,1)

for k = 1:5

    eigval = A * x ./ x
    j = 0

    while norm(eigval .- mean(eigval)) > 1e-3

        x = A * x

        projection = zeros(Complex{Float64}, 5)

        # gram-schmidt
        for l = 1:k-1

            projection[:] = projection[:] + (normalizedeigvecs[l]'x) .* normalizedeigvecs[l]

        end

        x = x - projection
        x = x ./ sqrt(x'x)

        eigval = A * x ./ x
        j += 1

        if j == 300
            break
        end
        
    end

    push!(eigvals, eigval[1])
    push!(normalizedeigvecs, x)

    projection = zeros(Complex{Float64}, 5)
    xnew = randn(Complex{Float64},5,1)

    # gram-schmidt
    for l = 1:k

        projection[:] = projection[:] + (normalizedeigvecs[l]'xnew) .* normalizedeigvecs[l]

    end

    x = xnew - projection

end