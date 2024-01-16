using Optimization, OptimizationOptimJL
using Enzyme, LinearAlgebra, Statistics, Random
# using Parameters, UnPack

# @with_kw mutable struct mso_params_ops{F<:Function}

#     T::Int               # Total steps to integrate 
#     t::Int = 0           # placeholder for current step
#     dt::Float64 = 0.001  # timestep

#     x::Vector{Float64} = zeros(6)        # placeholder for state vector
#     u::Matrix{Float64} = zeros(6, T+1)   # random forcing (if any), otherwise just leave zero
#     n::Matrix{Float64} = zeros(6, T+1)   # placeholder for noise to add to the data

#     k::Float64 = 30          # spring constant
#     r::Float64 = 0.5         # Rayleigh friction coefficient

#     q::F                     # forcing function

#     J::Float64 = 0.0         # cost function evaluation 

#     data_steps::Vector{Int64}  # the timesteps where data points exist 
#     data::Matrix{Float64}

#     states::Matrix{Float64}    # placeholder for computed states
#     energy::Matrix{Float64}    # placeholder for computed energy

#     A::Matrix{Float64} = zeros(6,6)                            # Time-step (x(t+1) = A x(t))
#     B::Matrix{Float64} = diagm([1., 0., 0., 0., 0., 0.])       # Distributes known forcing
#     Gamma::Matrix{Float64} = zeros(6,6)                        # Distrubutes unknown (random) forcing

#     E::Matrix{Float64} = zeros(6,6)           # Acts on data vector, generally the identity (e.g. full info on all positions/velocities)

#     Q::Matrix{Float64} = zeros(6,6)           # Covariance matrix for unknown (random) forcing
#     Q_inv::Float64
#     R::Matrix{Float64} = zeros(6,6)           # Covariance matrix for noise in data
#     R_inv::Matrix{Float64}                    # Inverse of operator R

#     K::Matrix{Float64} = zeros(6,6)           # Placeholder for Kalman gain matrix 
#     Kc::Matrix{Float64} = zeros(6,6)     

# end

# function integrate1(mso_struct::mso_params_ops)

#     @unpack B, Gamma, E, R, R_inv, Kc, Q, Q_inv = mso_struct
#     @unpack r = mso_struct
#     @unpack T, x, u, dt, states, data, energy, q =  mso_struct
#     @unpack data_steps, J = mso_struct

#     k = mso_struct.k

#     states[:,1] .= x

#     kin, ptl, total = compute_energy(states[:,1], Kc)
#     energy[:,1] = [kin;ptl;total]

#     Ac = zeros(6,6)
#     Ac[1,4] = 1
#     Ac[2,5] = 1
#     Ac[3,6] = 1
#     Ac[4,1] = -2*k
#     Ac[4,2] = k
#     Ac[4,4] = -r
#     Ac[5,1] = k
#     Ac[5,2] = -3*k
#     Ac[5,3] = k
#     Ac[5,5] = -r
#     Ac[6,2] = k
#     Ac[6,3] = -2*k
#     Ac[6,6] = -r
#     A = diagm(ones(6)) + dt .* Ac

#     # run the model forward to get estimates for the states 
#     temp = 0.0
#     for t = 2:T+1

#         x .= A * x + B * [q(temp); 0.; 0.; 0.; 0.; 0.] + Gamma * u[:, t-1]
#         states[:, t] .= copy(x)

#         temp += dt

#         if t in data_steps

#             mso_struct.J = mso_struct.J + (E * x - E * data[:, t])' * R_inv * (E * x - E * data[:, t])

#         end

#         kin, ptl, total = compute_energy(x, Kc)
#         energy[:,t] = [kin;ptl;total]

#         # xnew[1] = xold[4] + q(temp) + u[1,t-1]
#         # xnew[2] = xold[5] + u[2,t-1]
#         # xnew[3] = xold[6] + u[3,t-1]
#         # xnew[4] = -2*k*xold[1] + k*xold[2] - r*xold[4] + u[4,t-1]
#         # xnew[5] = k*x[1] - 3*k*xold[2] + k*xold[3] - r*xold[5] + u[5,t-1]
#         # xnew[6] = k*x[2] - 2*k*xold[3] - r*xold[6] + u[6,t-1]

#         # # x .= A * x + B * [q(temp); 0.; 0.; 0.; 0.; 0.] + Gamma * u[:, t-1]

#         # states[:, t] .= copy(xnew)

#         # temp += dt

#         # if t in data_steps

#         #     mso_struct.J = mso_struct.J + (E * xnew - E * data[:, t])' * R_inv * (E * xnew - E * data[:, t]) + u[:, t]' * Q_inv * u[:, t]

#         # end

#         # kin, ptl, total = compute_energy(xnew, Kc)
#         # energy[:,t] = [kin;ptl;total]

#         # xold = xnew 
#         # xnew = zeros(6)

#     end

#     return nothing

# end

function cost_function_eval(k_guess)

    k = k_guess[1]
    # Parameter choices 
    T = 10000             # Total number of steps to integrate
    r = 0.5               # spring coefficient

    ###########################################
    q_true(t) = 0.1 * cos(2 * pi * t / (2.5 / r))      # known forcing function
    # q_kf(t) = 0.5 * q_true(t)                          # forcing seen by KF and adjoint
    q_kf(t) = q_true(t)
    ###########################################

    data_steps1 = [j for j in 3000:200:7000]         # steps where data will be assimilated
    # data_steps2 = [k for k in 7200:100:9000]
    # data_steps = [data_steps1;data_steps2]
    data_steps = data_steps1
    # data_steps = [t for t in 1:T]

    rand_forcing = 0.1 .* randn(T+1)
    u = zeros(6, T+1)
    u[1, :] .= rand_forcing

    params_true = ThreeMassSpring.mso_params(T = T,
    x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    u = u,
    k = 30.,
    n = 0.00001 .* randn(6, T+1), ###################################
    q = q_true,
    data_steps = data_steps,
    data = zeros(1,1),
    states = zeros(6, T+1),
    energy = zeros(3, T+1)
    )

    params_pred = ThreeMassSpring.mso_params(T = T,
    x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    q = q_kf,
    u = u,
    k = k,
    data_steps = data_steps,
    data = zeros(1,1),
    states = zeros(6, T+1),
    energy = zeros(3, T+1)
    )

    ops = ThreeMassSpring.build_ops(params_true)

    # assuming data of all positions and velocities -> E is the identity operator
    ops.E .= ThreeMassSpring.Diagonal(ones(6))

    ops.Q[1,1] = ThreeMassSpring.cov(params_true.u[:], corrected=false)
    ops.R .= ThreeMassSpring.cov(params_true.n[:], corrected=false) .* ThreeMassSpring.Diagonal(ones(6))

    # assuming random forcing on position of mass one
    ops.Gamma[1, 1] = 1.0

    # pure prediction model
    _ = ThreeMassSpring.create_data(params_pred, ops)

    states_noisy = ThreeMassSpring.create_data(params_true, ops)

    diag = 0.0
    Q_inv = diag

    ######################
    # R_inv = ops.R^(-1)
    R_inv = ops.E
    ######################

    params_adjoint = ThreeMassSpring.mso_params_ops(T=T,
        t = 0,
        x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        u = u,
        n = 0.0 .* randn(6, T+1),
        q = q_kf,
        J = 0.0,
        k = k[1],
        data_steps = data_steps,
        data = states_noisy,
        states = zeros(6, T+1),
        energy = zeros(3, T+1),
        A = ops.A,
        B = ops.B,
        Gamma = ops.Gamma,
        E = ops.E,
        Q = 0.0 .* ops.Q,
        Q_inv = Q_inv,
        R = ops.R,
        R_inv = R_inv,
        K = ops.K,
        Kc = ops.Kc
    )

    ThreeMassSpring.integrate1(params_adjoint)

    return params_adjoint.J

end

# optf = OptimizationFunction(cost_function_eval, AutoEnzyme())
# prob = OptimizationProblem(optf,[20],0.0)
# sol = solve(prob, BFGS())

function gradient_eval!(G, k)

    k = k[1]
    # Parameter choices 
    T = 10000             # Total number of steps to integrate
    r = 0.5               # spring coefficient 
    q_true(t) = 0.1 * cos(2 * pi * t / (2.5 / r))      # known forcing function 
    q_kf(t) = 0.5 * q_true(t)                          # forcing seen by KF and adjoint
    data_steps1 = [j for j in 3000:200:7000]         # steps where data will be assimilated
    data_steps = data_steps1

    rand_forcing = 0.1 .* randn(T+1)
    u = zeros(6, T+1)
    u[1, :] .= rand_forcing

    params_true = ThreeMassSpring.mso_params(T = T,
    x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    u = u,
    k = 30.0,
    n = 0.05 .* randn(6, T+1),
    q = q_true,
    data_steps = data_steps,
    data = zeros(1,1),
    states = zeros(6, T+1),
    energy = zeros(3, T+1)
    )

    params_pred = ThreeMassSpring.mso_params(T = T,
    x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    q = q_kf,
    u = u,
    k = k,
    data_steps = data_steps,
    data = zeros(1,1),
    states = zeros(6, T+1),
    energy = zeros(3, T+1)
    )

    ops = ThreeMassSpring.build_ops(params_pred)

    # assuming data of all positions and velocities -> E is the identity operator
    ops.E .= ThreeMassSpring.Diagonal(ones(6))

    ops.Q[1,1] = ThreeMassSpring.cov(params_true.u[:], corrected=false)
    ops.R .= ThreeMassSpring.cov(params_true.n[:], corrected=false) .* ThreeMassSpring.Diagonal(ones(6))

    # assuming random forcing on position of mass one
    ops.Gamma[1, 1] = 1.0

    # pure prediction model
    _ = ThreeMassSpring.create_data(params_pred, ops)

    states_noisy = ThreeMassSpring.create_data(params_true, ops)

    diag = 0.0
    Q_inv = diag
    R_inv = ops.R^(-1)

    params_adjoint = ThreeMassSpring.mso_params_ops(T=T,
    t = 0,
    x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    u = u,
    n = 0.001 .* randn(6, T+1),
    q = q_kf,
    J = 0.0,
    k = k,
    data_steps = data_steps,
    data = states_noisy,
    states = zeros(6, T+1),
    energy = zeros(3, T+1),
    A = ops.A,
    B = ops.B,
    Gamma = ops.Gamma,
    E = ops.E,
    Q = 0.0 .* ops.Q,
    Q_inv = Q_inv,
    R = ops.R,
    R_inv = R_inv,
    K = ops.K,
    Kc = ops.Kc
    )

    dparams = Enzyme.Compiler.make_zero(Core.Typeof(params_adjoint), IdDict(), params_adjoint)
    dparams.J = 1.0
    dparams.k = 0.
    dparams.r = 0.
    dparams.dt = 0.
    dparams.Q_inv = 0.
    @show dparams

    autodiff(Reverse, ThreeMassSpring.integrate1, Duplicated(params_adjoint, dparams))

    @show dparams

    G[1] = dparams.k

end

function cost_and_gradient_eval(F, G, k)

    k = k_guess[1]
    # Parameter choices 
    T = 10000             # Total number of steps to integrate
    r = 0.5               # spring coefficient

    ###########################################
    q_true(t) = 0.1 * cos(2 * pi * t / (2.5 / r))      # known forcing function
    # q_kf(t) = 0.5 * q_true(t)                          # forcing seen by KF and adjoint
    q_kf(t) = q_true(t)
    ###########################################

    data_steps1 = [j for j in 3000:200:7000]         # steps where data will be assimilated
    # data_steps2 = [k for k in 7200:100:9000]
    # data_steps = [data_steps1;data_steps2]
    data_steps = data_steps1
    # data_steps = [t for t in 1:T]

    rand_forcing = 0.1 .* randn(T+1)
    u = zeros(6, T+1)
    u[1, :] .= rand_forcing

    params_true = ThreeMassSpring.mso_params(T = T,
    x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    u = u,
    k = 30.,
    n = 0.0 .* randn(6, T+1), ###################################
    q = q_true,
    data_steps = data_steps,
    data = zeros(1,1),
    states = zeros(6, T+1),
    energy = zeros(3, T+1)
    )

    params_pred = ThreeMassSpring.mso_params(T = T,
    x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    q = q_kf,
    u = u,
    k = k,
    data_steps = data_steps,
    data = zeros(1,1),
    states = zeros(6, T+1),
    energy = zeros(3, T+1)
    )

    ops = ThreeMassSpring.build_ops(params_true)

    # assuming data of all positions and velocities -> E is the identity operator
    ops.E .= ThreeMassSpring.Diagonal(ones(6))

    ops.Q[1,1] = ThreeMassSpring.cov(params_true.u[:], corrected=false)
    ops.R .= ThreeMassSpring.cov(params_true.n[:], corrected=false) .* ThreeMassSpring.Diagonal(ones(6))

    # assuming random forcing on position of mass one
    ops.Gamma[1, 1] = 1.0

    # pure prediction model
    _ = ThreeMassSpring.create_data(params_pred, ops)

    states_noisy = ThreeMassSpring.create_data(params_true, ops)

    diag = 0.0
    Q_inv = diag

    ######################
    # R_inv = ops.R^(-1)
    R_inv = ops.E
    ######################

    params_adjoint = ThreeMassSpring.mso_params_ops(T=T,
        t = 0,
        x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        u = u,
        n = 0.0 .* randn(6, T+1),
        q = q_kf,
        J = 0.0,
        k = k,
        data_steps = data_steps,
        data = states_noisy,
        states = zeros(6, T+1),
        energy = zeros(3, T+1),
        A = ops.A,
        B = ops.B,
        Gamma = ops.Gamma,
        E = ops.E,
        Q = 0.0 .* ops.Q,
        Q_inv = Q_inv,
        R = ops.R,
        R_inv = R_inv,
        K = ops.K,
        Kc = ops.Kc
    )

    dparams = ThreeMassSpring.Enzyme.Compiler.make_zero(Core.Typeof(params_adjoint), IdDict(), params_adjoint)
    dparams.J = 1.0
    dparams.k = 0.
    dparams.r = 0.
    dparams.dt = 0.
    dparams.Q_inv = 0.

    Enzyme.autodiff(Reverse, ThreeMassSpring.integrate1, Duplicated(params_adjoint, dparams))

    if G !== nothing
        G = dparams.k
    end

    if F !== nothing
        return params_adjoint.J
    end

end

k_guess = [29.999]
result = Optim.optimize(cost_function_eval, k_guess)
# result = Optim.optimize(Optim.only_fg!(cost_and_gradient_eval), k_guess)

# k_values = 20.0:1.0:40.0
# cost_values = zeros(length(k_values))
# for i = 1:length(k_values)

#     J = cost_function_eval(k_values[i])
#     cost_values[i] = J

# end

# plot(k_values, cost_values)
# xlabel!("k")
# ylabel!("J(k)")