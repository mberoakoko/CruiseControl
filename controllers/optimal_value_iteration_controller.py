import typing

import control
import dataclasses

import numpy as np
from numpy.typing import NDArray

import utils.util


@dataclasses.dataclass(frozen=True)
class ValueIterParams:
    Q: NDArray
    R: NDArray
    plant: control.StateSpace


class ValueIteraState(typing.NamedTuple):
    K: NDArray
    P: NDArray


def _value_iter_step(state: ValueIteraState, params: ValueIterParams) -> ValueIteraState:
    A, B = params.plant.A, params.plant.B
    Ki = np.linalg.inv((params.R + B.T @ state.P @ B)) @ B.T @ state.P
    return ValueIteraState(
        K=Ki,
        P=state.Q + Ki.T @ params.R @ Ki + (A + B @ Ki).T @ state.P @ (A + B @ Ki)
    )


def _cost_to_go(state: NDArray, covariance: NDArray) -> NDArray:
    return state.T @ covariance @ state


def _termination_criteria(state_1: ValueIteraState, state_2: ValueIteraState) -> bool:
    a = np.linalg.matrix_norm(state_1.P)
    b = np.linalg.matrix_norm(state_2.P)
    return np.abs(a - b) < 1e-6

type SteppingFunctionType = typing.Callable[[ValueIteraState], ValueIteraState]

def value_iteration_procedure(initial_state: ValueIteraState, params: ValueIterParams) -> typing.Iterator[ValueIteraState]:
    steping_function: SteppingFunctionType = lambda state: _value_iter_step(state, params)
    return utils.util.converge(
        values=utils.util.iterate(steping_function, start=initial_state),
        done=_termination_criteria
    )

def optimal_values(inital_state: ValueIteraState, params: ValueIterParams) -> ValueIteraState:
    stepping_function: SteppingFunctionType = lambda state: _value_iter_step(state, params)
    return utils.util.converged[ValueIteraState](
        values=utils.util.iterate(stepping_function, start=inital_state),
        done=_termination_criteria
    )