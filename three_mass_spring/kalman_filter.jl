# defines the function that runs the Kalman filter 

function run_kalman_filter(
    params, 
    ops
)

    @unpack A, B, Gamma, P0, P, E, Q, R, K, Kc = ops 
    @unpack T, x, dt, states, energy =  params 
    @unpack data, data_steps = params 

    states[:, 1] .= x
    uncertainty = [copy(P0)]

    kin, ptl, total = compute_energy(x, Kc)
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

        kin, ptl, total = compute_energy(x, Kc)
        energy[:,t] = [kin;ptl;total]

        states[:, t] .= x
        push!(uncertainty, copy(P))

        temp += dt 

    end 

    return uncertainty

end 