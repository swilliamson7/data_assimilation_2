# defines a function that runs the adjoint method

function run_adjoint(
    params, 
    ops)

    @unpack A, B, Gamma, E, R, Kc = ops 
    @unpack T, x, u, q, dt, states, data, energy =  params 
    @unpack data_steps = params

    states[:,1] .= x 

    kin, ptl, total = compute_energy(states[:,1], Kc)
    energy[:,1] = [kin;ptl;total]

    # run the model forward to get estimates for the states 
    temp = 0.0
    for t = 2:T+1

        x .= A * x + B * [q(temp); 0.; 0.; 0.; 0.; 0.] + Gamma * u[:, t-1]
        states[:, t] .= x 

        kin, ptl, total = compute_energy(x, Kc)
        energy[:,t] = [kin;ptl;total]

        temp += dt

    end
    
    # find the initial adjoint variable 
    if T+1 in data_steps 
        initial_adjoint = 2 * (E' * R^(-1) * (E * states[:, T+1] - E * data[:, T+1])) 
    else
        initial_adjoint = zeros(6)
    end

    adjoint_variables = zeros(6, T+1)
    adjoint_variables[:, T+1] .= initial_adjoint

    # run Enzyme backwards to find and store all adjoint variables 
    for k = T:-1:1

        adjoint_new = zeros(6)

        autodiff(Reverse,
            one_step_forward, 
            temp,
            Duplicated(zeros(6), adjoint_variables[:, k+1]),
            Duplicated(states[:,k], adjoint_new),
            params,
            ops,
            params.u[:,k]
        )

        if k in data_steps

            adjoint_new[:] = adjoint_new[:] + 2 * (E' * R^(-1) * (E * states[:,k] - E * data[:,k]))

        end

        adjoint_variables[:, k] .= adjoint_new 

        temp -= dt

    end

    # @assert all(x -> x < 100.0, params.x)

    return adjoint_variables

end

# Function to run gradient descent with the gradient computed from the adjoint variables. 
# Since the initial condition is assumed known we're minimizing w.r.t. the control 
# parameters.  
function grad_descent(M, params::mso_params, ops::mso_operators)#, x0::Vector{Float64})

    @unpack T = params

    u_new = zeros(6, T+1)

    params.x .= [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    params.states .= zeros(6, T+1)
    params.energy .= zeros(3, T+1)

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

function grad_descent(M, params::mso_params_ops, x0::Vector{Float64})

    @unpack T = params

    u_new = zeros(6, T+1)
    params.x .= x0
    params.states .= zeros(6, T+1)
    params.energy .= zeros(3, T+1)

    dparams = Enzyme.Compiler.make_zero(Core.Typeof(params), IdDict(), params)
    dparams.J = 1.0

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

        dparams = Enzyme.Compiler.make_zero(Core.Typeof(params), IdDict(), params)
        dparams.J = 1.0

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

function cp_adjoint(cstruct, chkpt_scheme, q)

    temp = 0.0 
    # cstruct.t = 2 

    @checkpoint_struct chkpt_scheme cstruct for cstruct.t in 2:cstruct.T

        t = cstruct.t 
        cstruct.x .= cstruct.A * cstruct.x + cstruct.B * [q(temp); 0.; 0.; 0.; 0.; 0.] + cstruct.Gamma * cstruct.u[:, t]
        cstruct.states[:, t] .= cstruct.x 

        kin, ptl, total = compute_energy(cstruct.x, cstruct.Kc)
        cstruct.energy[:,t] = [kin;ptl;total]

        # if t in cstruct.data_steps 

        #     cstruct.J += (cstruct.E * cstruct.x - cstruct.E * cstruct.data[:, t])' * 
        #         cstruct.R^(-1) * (cstruct.E * cstruct.x - cstruct.E * cstruct.data[:, t])
        #         + cstruct.u[:,t]' * cstruct.Q^(-1) * cstruct.u[:,t]

        # end

        temp += cstruct.dt 
        # cstruct.t += 1 

    end 

    
    return cstruct.J

end
        
