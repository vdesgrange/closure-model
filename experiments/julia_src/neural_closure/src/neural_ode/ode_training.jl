module ODETraining

using Flux
using FluxTraining
using DiffEqFlux
using GalacticOptim
using OrdinaryDiffEq

struct ODETrainingPhase <: FluxTraining.AbstractTrainingPhase end
struct ODEValidationPhase <: FluxTraining.AbstractValidationPhase end


function FluxTraining.step!(learner, phase::ODETrainingPhase, batch)
    """
    DEPRECATED
    """
    xs, ys, ts = batch

    p, re = Flux.destructure(learner.model);
    net(u, p, t) = re(p, u);

    prob = ODEProblem{false}(net, Nothing, (Nothing, Nothing));

    function predict_neural_ode(x, t)
        tspan = (t[1], t[end]);
        _prob = remake(prob; u0=x, p=p, tspan=tspan);
        return Array(solve(_prob, Tsit5(), u0=x, p=p, saveat=t));
    end

    FluxTraining.runstep(learner, phase, (; xs=xs, ys=ys, ts=ts)) do handle, state
        state.grads = FluxTraining._gradient(learner.optimizer, learner.model, learner.params) do model
            state.ŷs = predict_neural_ode(state.xs, state.ts[1]);
            handle(LossBegin())
            state.loss = learner.lossfn(state.ŷs, state.ys)
            handle(BackwardBegin())
            return state.loss
        end
        handle(BackwardEnd())
        learner.params, learner.model = FluxTraining._update!(learner.optimizer, learner.params, learner.model, state.grads)
    end
end


function FluxTraining.step!(learner, phase::ODEValidationPhase, batch)
    """
    DEPRECATED
    """
    xs, ys, ts = batch
    prob = ODEProblem{false}(learner.model, Nothing, (Nothing, Nothing));
    p, re = Flux.destructure(learner.model);
    net(u, p, t) = re(p, u);
    prob = ODEProblem{false}(net, Nothing, (Nothing, Nothing));

    function predict_neural_ode(x, t)
        tspan = (t[1], t[end]);
        _prob = remake(prob; u0=x, p=p, tspan=tspan);
        Array(solve(_prob, Tsit5(), u0=x, p=p, saveat=t));
    end

    FluxTraining.runstep(learner, phase, (;xs=xs, ys=ys, ts=ts)) do _, state
        state.ŷs = predict_neural_ode(state.xs, state.ts[1])
        state.loss = learner.lossfn(state.ŷs, state.ys)
    end
end


function fit!(learner, nepochs::Int, (trainiter, validiter))
    """
    DEPRECATED
    """
    for i in 1:nepochs
        epoch!(learner, ODETrainingPhase(), trainiter)
        epoch!(learner, ODEValidationPhase(), validiter)
    end
end

function fit!(learner, nepochs::Int)
    """
    DEPRECATED
    """
    fit!(learner, nepochs, (learner.data.training, learner.data.validation))
end


end
