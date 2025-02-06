
function cost_eval(param_guess, data, data_spots)

    P = ShallowWaters.Parameter(T=Float32;
    output=false,
    L_ratio=1,
    g=9.81,
    H=500,
    wind_forcing_x="double_gyre",
    Lx=3840e3,
    tracer_advection=false,
    tracer_relaxation=false,
    seasonal_wind_x=false,
    data_steps=data_steps,
    topography="flat",
    bc="nonperiodic",
    α=2,
    nx=128,
    Ndays=Ndays,
    initial_cond="ncfile",
    initpath="./data_files_forkf/128_spinup_noforcing/"
    )
    S = ShallowWaters.model_setup(P)
    m,n = size(S.Prog.η)
    @show m
    @show n

    S.Prog.u = reshape(param_guess[1:17292], 131, 132)
    S.Prog.v = reshape(param_guess[17293:34584], 132, 131)
    S.Prog.η = reshape(param_guess[34585:end], 130, 130)

    J = integrate(S, data, data_spots)

    return J

end

function gradient_eval(G, param_guess, data, data_spots)

    P = ShallowWaters.Parameter(T=Float32;
    output=false,
    L_ratio=1,
    g=9.81,
    H=500,
    wind_forcing_x="double_gyre",
    Lx=3840e3,
    tracer_advection=false,
    tracer_relaxation=false,
    seasonal_wind_x=false,
    data_steps=data_steps,
    topography="flat",
    bc="nonperiodic",
    α=2,
    nx=128,
    Ndays=Ndays,
    initial_cond="ncfile",
    initpath="./data_files_forkf/128_spinup_noforcing/"
    )
    S = ShallowWaters.model_setup(P)

    m,n = size(S.Diag.VolumeFluxes.h)
    @show m
    @show n

    S.Prog.u = reshape(param_guess[1:17292], 131, 132)
    S.Prog.v = reshape(param_guess[17293:34584], 132, 131)
    S.Prog.η = reshape(param_guess[34585:end], 130, 130)

    dS = Enzyme.Compiler.make_zero(S)
    # snaps = Int(floor(sqrt(S.grid.nt)))
    # revolve = Revolve{ShallowWaters.ModelSetup}(S.grid.nt,
    #     snaps;
    #     verbose=1,
    #     gc=true,
    #     write_checkpoints=false,
    #     write_checkpoints_filename = "",
    #     write_checkpoints_period = 224
    # )

    ddata = Enzyme.make_zero(data)
    ddata_spots = Enzyme.make_zero(data_spots)

    autodiff(Enzyme.ReverseWithPrimal, integrate,
    Duplicated(S, dS),
    Duplicated(data, ddata),
    Duplicated(data_spots, ddata_spots)
    )

    G = [vec(dS.Prog.u); vec(dS.Prog.v); vec(dS.Prog.η)]

    return nothing

end

function FG(F, G, param_guess, data, data_spots)

    G === nothing || gradient_eval(G, param_guess, data, data_spots)
    F === nothing || return cost_eval(param_guess, data, data_spots)

end