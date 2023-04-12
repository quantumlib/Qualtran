from attrs import frozen


@frozen
class PhysicalCost:
    failure_prob: float
    """Approximate probability of an error occurring during execution of the algorithm.

    This can be a bad CCZ being produced, a bad T state being produced,
    or a topological error occurring during the algorithm.
    """

    footprint: int
    """Total physical qubits required to run algorithm."""

    duration_hr: float
    """Total time in hours to run algorithm."""
