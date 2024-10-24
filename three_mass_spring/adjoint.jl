
# Function to run gradient descent with the gradient computed from the adjoint variables

function grad_descent(M, params::mso_params_ops, x0::Vector{Float64})

    T = params.T

    u_new = zeros(6, T+1)
    params.x .= x0
    params.states .= zeros(6, T+1)
    params.energy .= zeros(3, T+1)
    params.J = 0.0

    dparams = Enzyme.Compiler.make_zero(Core.Typeof(params), IdDict(), params)
    dparams.J = 1.0
    dparams.k = 0.
    dparams.r = 0.
    dparams.dt = 0.
    dparams.Q_inv = 0.

    autodiff(Reverse, integrate, Duplicated(params, dparams))

    for t = 1:T 

        u_new[:, t] = params.u[:, t] - 1 / (norm(dparams.u[1,:])) * dparams.u[:, t]

    end

    k = 1
    u_old = copy(params.u)
    u_grad_old = copy(dparams.u)
    params.u .= u_new 

    while norm(u_grad_old[1,:]) > 500

        params.x .= x0
        params.states .= zeros(6, T+1)
        params.J = 0.0

        dparams = Enzyme.Compiler.make_zero(Core.Typeof(params), IdDict(), params)
        dparams.J = 1.0
        dparams.k = 0.
        dparams.r = 0.
        dparams.dt = 0.
        dparams.Q_inv = 0.

        autodiff(Reverse, integrate, Duplicated(params, dparams))

        @show norm(dparams.u[1,:])

        gamma = 0.0
        num = 0.0
        den = 0.0

        for j = 1:T 

            num += sum(dot(params.u[:,j] - u_old[:,j], dparams.u[:,j] - u_grad_old[:,j]))
            den += norm(dparams.u[:,j] - u_grad_old[:,j])^2

        end

        gamma = (abs(num) / den)

        for t = 1:T 

            u_new[:, t] = params.u[:, t] - gamma * dparams.u[:, t]

        end

        u_old = copy(params.u)
        u_grad_old = copy(dparams.u)
        params.u .= u_new
        dparams.u = zeros(6, T+1)

        k += 1

        if k > M
            break
        end

    end

    return dparams

end

### None of the following really in use anymore, these computed gradients by manually adding
# together all the adjoint variables and instead I'm now using the other version
# defined below that solely uses Enzyme
function grad_descent(M, params::mso_params, ops::mso_operators)#, x0::Vector{Float64})

    T = params.T

    u_new = zeros(6, T+1)

    params.x .= [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    params.states .= zeros(6, T+1)
    params.energy .= zeros(3, T+1)
    params.J = 0.0

    adjoint_variables = run_adjoint(params,
        ops
    )

    u_grad_new = compute_control_deriv(adjoint_variables,
        params,
        ops
    )

    for t = 1:T
            
        u_new[:, t] = params.u[:, t] - 1 / (norm(u_grad_new[1,:])) * u_grad_new[:, t]

    end

    k = 1
    u_old = copy(params.u)
    u_grad_old = copy(u_grad_new)
    params.u .= u_new

    while norm(u_grad_old[1,:]) > 500

        params.x .= [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        params.states .= zeros(6, T+1)

        adjoint_variables = run_adjoint(params,
            ops
        )

        u_grad_new = compute_control_deriv(adjoint_variables,
            params,
            ops
        )

        @show norm(u_grad_new[1,:])

        gamma = 0.0
        num = 0.0
        den = 0.0

        for j = 1:T 

            num += sum(dot(params.u[:,j] - u_old[:,j], u_grad_new[:,j] - u_grad_old[:,j]))
            den += norm(u_grad_new[:,j] - u_grad_old[:,j])^2

        end

        gamma = (abs(num) / den)

        for t = 1:T 

            u_new[:, t] = params.u[:, t] - gamma * u_grad_new[:, t]

        end

        u_old = copy(params.u)
        u_grad_old = copy(u_grad_new)
        params.u .= u_new
        u_grad_new = zeros(6, T+1)

        k += 1

        if k > M
            break
        end

    end

end

function compute_control_deriv(adjoint_variables, params, ops)

    @unpack u = params 
    @unpack A, B, Q, Gamma = ops

    u_deriv = zeros(6, params.T+1)
    Q_inv = zeros(6,6)
    Q_inv[1,1] = 1/Q[1,1]

    for t = 1:params.T 
        
        u_deriv[:, t] .= 2 * Q_inv * u[:, t] + Gamma' * adjoint_variables[:, t+1]

    end

    return u_deriv

end