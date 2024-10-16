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