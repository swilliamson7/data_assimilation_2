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

        x .= A * x + B * [q(temp); 0.; 0.; 0.; 0.; 0.] + Gamma * u[:, t]
        states[:, t] .= x 

        kin, ptl, total = compute_energy(x, Kc)
        energy[:,t] = [kin;ptl;total]

        temp += dt

    end 

    # find the initial adjoint variable 
    if T+1 in data_steps 
        initial_adjoint = E' * R^(-1) * (E * states[:, T+1] - E * data[:, T+1])
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
            ops
        )

        if k in data_steps

            adjoint_new[:] = adjoint_new[:] + E' * R^(-1) * (E * states[:,k] - E * data[:,k])

        end

        adjoint_variables[:, k] .= adjoint_new 

        temp -= dt

    end

    @assert all(x -> x < 100.0, params.x)

    return adjoint_variables

end

# Function to run gradient descent with the gradient computed from the adjoint variables. 
# Since the initial condition is assumed known we're minimizing w.r.t. the control 
# parameters.  
function grad_descent(M, params, ops)

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
        ops=ops
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
            ops=ops
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


function compute_control_deriv(adjoint_variables, params; ops=ops)

    @unpack u = params 
    @unpack A, B, Q, Gamma = ops

    u_deriv = zeros(6, params.T)
    Q_inv = zeros(6,6)
    Q_inv[1,1] = 1/Q[1,1]

    for t = 1:params.T 
        
        u_deriv[:, t] .= 2*(Q_inv * u[:, t] + Gamma' * adjoint_variables[:, t+1])

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
        
# function gradient_for_optim(u; params = params_adjoint, ops = ops)

#     params.u .= u
#     params.x .= [1.;0.;0.;0.;0.;0.]
#     params.states .= zeros(6, params.T+1)

#     adjoint_variables = run_adjoint(params, 
#         ops
#     )

#     u_deriv = compute_control_deriv(adjoint_variables, params)

#     return u_deriv

# end

# function objective_for_optim(u; params = params_adjoint, ops = ops)

#     objective_sum = 0.0 
#     Q_inv = 1/ops.Q[1,1]

#     for t = 1:params.T
        
#         if t in params.data_steps 

#             objective_sum += 0.5 * sum((ops.E * params.states[:,t] - ops.E * params.data[:,t])' * 
#                 ops.R^(-1) * (ops.E * params.states[:,t] - ops.E * params.data[:,t]))

#         end

#         if t != T

#             objective_sum += 0.5 * (u[1, t]' * Q_inv * u[1, t])

#         end

#     end

#     return objective_sum

# end 

# function fg!(F, G, init_guess)

#     # Parameter choices 
#     T = 10000          # Total number of steps to integrate
#     r = 0.5
#     q_true(t) = 0.1 * cos(2 * pi * t / (2.5 / r))                       # known forcing function 
#     q_kf(t) = 0.5 * q_true(t)                                           # forcing seen by KF (and adjoint)
#     data_steps = [1500 + k*1000 for k in 1:5]      # steps where data will be assimilated

#     rand_forcing = 0.1 .* randn(T+1)
#     u_true = zeros(6, T+1)
#     u_true[1, :] .= rand_forcing

#     params_true = mso_params(T = T,
#     x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     u = u_true,
#     n = 0.05 .* randn(6, T+1), 
#     q = q_true,
#     data_steps = data_steps,
#     data = zeros(1,1),
#     states = zeros(6, T+1),
#     energy = zeros(3, T+1)
#     )

#     ops = build_ops(params_true)

#     # assuming data of all positions and velocities -> E is the identity operator 
#     ops.E .= Diagonal(ones(6))

#     ops.Q[1,1] = cov(params_true.u[:], corrected=false)
#     ops.R .= cov(params_true.n[:], corrected=false) .* Diagonal(ones(6))

#     # assuming random forcing on position of mass one 
#     ops.Gamma[1, 1] = 1.0 

#     states_noisy = create_data(params_true, ops)

#     params_adjoint = mso_params(T=T, 
#     x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     u = init_guess,
#     n = 0.001 .* randn(6, T+1), 
#     q = q_kf,
#     data_steps = data_steps,
#     data = states_noisy,
#     states = zeros(6, T+1),
#     energy = zeros(3, T+1)
#     )

#     @unpack A, B, Gamma, E, R, Kc, Q = ops 
#     @unpack T, x, u, q, dt, states, data, energy =  params_adjoint 
#     @unpack data_steps = params_adjoint

#     states[:,1] .= x 


#     kin, ptl, total = compute_energy(states[:,1], Kc)
#     energy[:,1] = [kin;ptl;total]

#     # run the model forward to get estimates for the states 
#     temp = 0.0 
#     for t = 2:T+1 

#         x .= A * x + B * [q(temp); 0.; 0.; 0.; 0.; 0.] + Gamma * u[:, t]
#         states[:, t] .= x 

#         kin, ptl, total = compute_energy(x, Kc)
#         energy[:,t] = [kin;ptl;total]

#         temp += dt

#     end 


#     if G !== nothing 

#         # find the initial adjoint variable 
#         if T+1 in data_steps 
#             initial_adjoint = E' * R^(-1) * (E * states[:, T+1] - E * data[:, T+1])
#         else
#             initial_adjoint = zeros(6)
#         end

#         adjoint_variables = zeros(6, T+1)
#         adjoint_variables[:, T+1] .= initial_adjoint

#         # run Enzyme backwards to find and store all adjoint variables 
#         for k = T:-1:1 

#             adjoint_new = zeros(6)

#             autodiff(Reverse,
#                 one_step_forward, 
#                 temp,
#                 Duplicated(zeros(6), adjoint_variables[:, k+1]),
#                 Duplicated(states[:,k], adjoint_new),
#                 params_adjoint,
#                 ops
#             )

#             if k in data_steps

#                 adjoint_new[:] = adjoint_new[:] + E' * R^(-1) * (E * states[:,k] - E * data[:,k])

#             end

#             adjoint_variables[:, k] .= adjoint_new 

#             temp -= dt

#         end

#         @assert all(x -> x < 100.0, params_adjoint.x)

#         @unpack u = params_adjoint 
    
#         u_deriv = zeros(6, params_adjoint.T)
#         Q_inv = zeros(6,6)
#         Q_inv[1,1] = 1/Q[1,1]
    
#         for t = 1:params_adjoint.T 
            
#             u_deriv[:, t] .= 2*(Q_inv * u[:, t] + Gamma' * adjoint_variables[:, t+1])
    
#         end

#         copyto!(G, u_deriv)
#     end

#     if F !== nothing
#         objective_sum = 0.0
#         Q_inv = 1/ops.Q[1,1]
#         for t = 1:params_adjoint.T   
#             if t in params_adjoint.data_steps 
#                 objective_sum += 0.5 * sum((ops.E * params_adjoint.states[:,t] - ops.E * params_adjoint.data[:,t])' * 
#                     ops.R^(-1) * (ops.E * params_adjoint.states[:,t] - ops.E * params_adjoint.data[:,t]))
#             end
#             if t != T
#                 objective_sum += 0.5 * (params_adjoint.u[1, t]' * Q_inv * params_adjoint.u[1, t])
#             end
#         end
#         return objective_sum
#     end

# end