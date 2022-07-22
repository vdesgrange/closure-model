module Training

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
    xs, ys = batch

    FluxTraining.runstep(learner, phase, (; xs=xs, ys=ys)) do handle, state
        state.grads = FluxTraining._gradient(learner.optimizer, learner.model, learner.params) do model
            println(size(state.xs))
            state.ŷs = model(state.xs)
            println(typeof(state.ŷs))
            println(size(state.ŷs))
            handle(FluxTraining.LossBegin())
            state.loss = learner.lossfn(state.ŷs, state.ys)
            handle(FluxTraining.BackwardBegin())
            return state.loss
        end
        handle(FluxTraining.BackwardEnd())
        learner.params, learner.model = FluxTraining._update!(learner.optimizer, learner.params, learner.model, state.grads)
    end
end


function FluxTraining.step!(learner, phase::ODEValidationPhase, batch)
    """
    DEPRECATED
    """
    xs, ys = batch

    FluxTraining.runstep(learner, phase, (;xs=xs, ys=ys)) do _, state
        state.ŷs = learner.model(state.xs)
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
