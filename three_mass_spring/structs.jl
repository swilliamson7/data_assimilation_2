@with_kw struct mso_state 

    x::Vector{Float64} = zeros(6)      # placeholder for state vector 

end

@with_kw struct mso_params{F<:Function}

    T::Int               # Total steps to integrate 
    dt::Float64 = 0.001  # timestep

    x::Vector{Float64} = zeros(6)      # placeholder for state vector 
    u::Matrix{Float64} = zeros(6, T)   # random forcing (if any), otherwise just leave zero 
    n::Matrix{Float64} = zeros(6, T)   # placeholder for noise to add to the data 

    k::Int = 30          # spring constant
    r::Float64 = 0.5     # Rayleigh friction coefficient

    q::F                 # forcing function 

    data_steps::Vector{Int64}       # the timesteps where data points exist 
    data::Matrix{Float64}

    states::Matrix{Float64}    # placeholder for computed states 
    energy::Matrix{Float64}    # placeholder for computed energy

end

# structure containing both time-stepping operators as well as operators
# related to the Kalman filter operators (e.g. the Kalman gain matrix K)
@with_kw struct mso_operators

    A::Matrix{Float64} = zeros(6,6)                            # Time-step (x(t+1) = A x(t))
    B::Matrix{Float64} = diagm([1., 0., 0., 0., 0., 0.])       # Distributes known forcing  
    Gamma::Matrix{Float64} = zeros(6,6)                        # Distrubutes unknown (random) forcing

    P0::Matrix{Float64} = zeros(6,6)          # Init. uncertainty operator (generally only non-zero when x0 not fully known)
    P::Matrix{Float64} = zeros(6,6)           # Placeholder for future uncertainty operators  

    E::Matrix{Float64} = zeros(6,6)           # Acts on data vector, generally the identity (e.g. full info on all positions/velocities)

    Q::Matrix{Float64} = zeros(6,6)           # Covariance matrix for unknown (random) forcing 
    R::Matrix{Float64} = zeros(6,6)           # Covariance matrix for noise in data 

    K::Matrix{Float64} = zeros(6,6)           # Placeholder for Kalman gain matrix           

end
    