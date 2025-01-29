
function cost_eval(param_guess; data, kwargs...)

    P = ShallowWaters.Parameter(T=Float32;kwargs...)
    S = ShallowWaters.model_setup(P)

    S.Prog.u = reshape(param_guess[1:17292], 131, 132)
    S.Prog.v = reshape(param_guess[17293:34584], 132, 131)
    S.Prog.η = reshape(param_guess[34585:end], 130, 130)

    J = integrate(S, data, data_spots)

    return J

end

function gradient_eval(G, param_guess; kwargs...)

    P = ShallowWaters.Parameter(T=Float32;kwargs...)
    S = ShallowWaters.model_setup(P)

    S.Prog.u = reshape(param_guess[1:17292], 131, 132)
    S.Prog.v = reshape(param_guess[17293:34585], 132, 131)
    S.Prog.η = reshape(param_guess[34586:end], 130, 130)

    dS = Enzyme.Compiler.make_zero(S)
    snaps = Int(floor(sqrt(S.grid.nt)))
    revolve = Revolve{ShallowWaters.ModelSetup}(S.grid.nt,
        snaps;
        verbose=1,
        gc=true,
        write_checkpoints=false,
        write_checkpoints_filename = "",
        write_checkpoints_period = 224
    )

    autodiff(Enzyme.ReverseWithPrimal, checkpointed_initcond, Duplicated(S, dS))

    G = [vec(dS.Prog.u); vec(dS.Prog.v); vec(dS.Prog.η)]

    return nothing

end

function FG(F, G, param_guess)

    G === nothing || gradient_eval(G, param_guess)
    F === nothing || return cost_eval(param_guess)

end