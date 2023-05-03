# defines the function that runs the Kalman filter 

function run_kalman_filter(
    params, 
    ops, 
    data;
    data_steps = [k for k in 100:125:700]
)

    @unpack A, B, Gamma, P0, P, E, Q, R, K = ops 
    @unpack T, x, dt, states, energy =  params 

    states[:, 1] .= x
    uncertainty = [copy(P0)]

    kin, ptl, total = compute_energy(x)
    energy[:,1] = [kin;ptl;total]

    P .= P0
    temp = 0.
    for t = 2:T+1

        x .=  A * x + B * [params.q(temp); 0.; 0.; 0.; 0.; 0.]
        P .= A * P * A' + Gamma * Q * Gamma' 

        K .= P * E' * (E * P * E' + R)^(-1)

        if t in data_steps 

            x .= x + K * (E * data[:, t] - E * x)
            P .= P - K * E * P

        end 

        kin, ptl, total = compute_energy(x)
        energy[:,t] = [kin;ptl;total]

        states[:, t] .= x
        push!(uncertainty, copy(P))

        temp += dt 

    end 

    return uncertainty

end 