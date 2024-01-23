function build_ops(params; E = zeros(6,6), R = zeros(6,6), K = zeros(6,6))
    
    dt = params.dt
    k = params.k
    r = params.r

    Rc = -r .* diagm(ones(3))
    Kc = [-2*k k 0; k -3*k k; 0 k -2*k]
    Ac = [zeros(3,3) diagm(ones(3))
         Kc Rc
    ]
    
    A = diagm(ones(6)) + dt .* Ac
    
    ms_ops = mso_operators(A=A,
    E=E,
    R=R,
    K=K,
    Kc=Kc
    )

    return ms_ops
end

# computes the kinetic energy given a state vector x
function compute_energy(x, Kc)

    kin = 0.5 * (x[4:6]' * x[4:6])
    ptl = 0.5 * (-x[1:3]' * Kc * x[1:3])

    return kin, ptl, kin + ptl

end

function create_data(params, ops)

    @unpack T, x, u, n, q, dt, states, energy = params
    @unpack A, B, Gamma, Kc = ops

    states[:,1] .= x

    states_noisy = zeros(6,T+1)
    states_noisy[:,1] .= x + n[:,1]
    kin, ptl, total = compute_energy(states[:,1], Kc)
    energy[:, 1] = [kin;ptl;total]

    temp = 0.0
    for j = 2:T+1

        x[:] = A * x + B * [q(temp); 0.; 0.; 0.; 0.; 0.] + Gamma * u[:, j-1]
        states[:, j] .= copy(x)

        kin, ptl, total = compute_energy(x, Kc)
        energy[:,j] = [kin;ptl;total]

        states_noisy[:, j] .= copy(x) + n[:, j]
        temp += dt

    end 

    return states_noisy

end

function integrate(mso_struct::mso_params_ops)

    @unpack A, B, Gamma, E, R, R_inv, Kc, Q, Q_inv = mso_struct
    @unpack T, x, u, dt, states, data, energy, q =  mso_struct
    @unpack data_steps, J = mso_struct

    states[:,1] .= x

    kin, ptl, total = compute_energy(states[:,1], Kc)
    energy[:,1] = [kin;ptl;total]

    # run the model forward to get estimates for the states 
    temp = 0.0
    for t = 2:T+1

        x .= A * x + B * [q(temp); 0.; 0.; 0.; 0.; 0.] + Gamma * u[:, t-1]
        states[:, t] .= copy(x)

        temp += dt

        if t in data_steps

            mso_struct.J = mso_struct.J + (E * x - E * data[:, t])' * R_inv * (E * x - E * data[:, t]) + u[:, t]' * Q_inv * u[:, t]

        end

        kin, ptl, total = compute_energy(x, Kc)
        energy[:,t] = [kin;ptl;total]

    end

    return nothing 

end

function integrate(mso_struct::mso_params, ops::mso_operators)

    @unpack A, B, Gamma, E, R, Kc, Q = ops
    @unpack T, x, u, q, dt, states, data, energy =  mso_struct
    @unpack data_steps = mso_struct

    kin, ptl, total = compute_energy(states[:,1], Kc)
    energy[:,1] = [kin;ptl;total]
    states[:,1] .= x

    diag = 0.0
    Q_inv = diag

    ######################
    ops.R_inv = ops.R^(-1)
    # R_inv = ops.E
    ######################

    # run the model forward to get estimates for the states 
    temp = 0.0
    for t = 2:T+1

        x .= A * x + B * [q(temp); 0.; 0.; 0.; 0.; 0.] + Gamma * u[:, t-1]
        states[:, t] .= copy(x)

        temp += dt

        if t in data_steps

            mso_struct.J = mso_struct.J + (E * x - E * data[:, t])' * R_inv * (E * x - E * data[:, t]) + u[:, t]' * Q_inv * u[:, t]

        end

        kin, ptl, total = compute_energy(x, Kc)
        energy[:,t] = [kin;ptl;total]

    end 

    return nothing 

end

function one_step_forward(
    t,
    xout,
    xin, 
    params,
    ops,
    u)

    xout[:] = ops.A * xin + ops.B * [params.q(t); 0.; 0.; 0.; 0.; 0.] + ops.Gamma * u

    # if j in params.data_steps
    #     Jout = Jin  + (xout - data[:, j])' * (xout - data[:, j])
    # else
    #     Jout = Jin
    # end

    return nothing

end