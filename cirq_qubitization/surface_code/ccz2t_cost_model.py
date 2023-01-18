import math

from attrs import frozen


@frozen
class CCZ2TCostModel:
    """Physical costs using the model from catalyzed CCZ to 2T paper.

    Args:
        t_count: Number of algorithm T gates (not including Toffoli compilations).
        toffoli_count: Number of algorithm Toffoli gates.
        n_alg_qubits: Number of algorithm logical qubits.
        error_budget: Acceptable chance of an error occurring at any point. This includes
            data storage failures as well as top-level distillation failure.
        physical_error_rate: The physical error rate of the device. This sets the suppression
            factor for increasing code distance.
        cycle_time_us: The number of microseconds it takes to execute a surface code cycle.
        distillation_l1_d: Code distance used for level 1 factories.
        distillation_l2_d: Code distance used for level 2 factories.
        routing_overhead: Additional space needed for moving magic states and data qubits around
            in order to perform operations.

    References:
        Efficient magic state factories with a catalyzed |CCZ> to 2|T> transformation.
        https://arxiv.org/abs/1812.01238
    """

    t_count: int
    toffoli_count: int
    n_alg_qubits: int
    error_budget: float

    physical_error_rate: float
    cycle_time_us: float

    distillation_l1_d: int = 15
    distillation_l2_d: int = 31
    routing_overhead: float = 0.5

    def error_at(self, *, d: int):
        """Logical error suppressed with code distance `d` for this physical error rate.

        This is an estimate, see the references section.

        The formula was originally expressed as $p_l = a (b * p_p)^((d+1)/2)$, with physical
        error rate $p_p$ and parameters $a$ and $b$. This can alternatively be expressed with
        $p_th = (1/b)$ roughly corresponding to the code threshold. This is sometimes also
        expressed with $lambda = p_th / p_p$. A lambda of 10, for example, would be p_p = 1e-3
        and p_th = 0.01. The pre-factor $a$ has no clear provenance.

        References:
            Low overhead quantum computation using lattice surgery. Fowler and Gidney (2018).
            https://arxiv.org/abs/1808.06709.
            See section XV for introduction of this formula, with citation to below.

            Surface code quantum error correction incorporating accurate error propagation.
            Fowler et. al. (2010). https://arxiv.org/abs/1004.0255.
            Note: this doesn't actually contain the formula from the above reference.
        """
        return 0.1 * (100 * self.physical_error_rate) ** ((d + 1) / 2)

    # -------------------------------------------------------------------------------
    # ----     Level 0    ---------
    # -------------------------------------------------------------------------------

    @property
    def l0_state_injection_error(self) -> float:
        """Error rate associated with the level-0 creation of a |T> state.

        By using the techniques of Ying Li (https://arxiv.org/abs/1410.7808), this can be
        done with approximately the same error rate as the underlying physical error rate.
        """
        return self.physical_error_rate

    @property
    def l0_topo_error_t_gate(self) -> float:
        """Topological error associated with level-0 distillation.

        For a level-1 code distance of `d1`, this construction uses a `d1/2` distance code
        for storing level-0 T states.
        """

        # The chance of a logical error occurring within a lattice surgery unit cell at
        # code distance d1*0.5.
        topo_error_per_unit_cell = self.error_at(d=self.distillation_l1_d // 2)

        # It takes approximately 100 L0 unit cells to get the injected state where
        # it needs to be and perform the T gate.
        return 100 * topo_error_per_unit_cell

    @property
    def l0_error(self) -> float:
        """Chance of failure of a T gate performed with an injected (level-0) T state.

        As a simplifying approximation here (and elsewhere) we assume different sources
        of error are independent, and we merely add the probabilities.
        """
        return self.l0_state_injection_error + self.l0_topo_error_t_gate

    # -------------------------------------------------------------------------------
    # ----     Level 1    ---------
    # -------------------------------------------------------------------------------

    @property
    def l1_topo_error_factory(self) -> float:
        """Topological error associated with a L1 T factory."""

        # The L1 T factory uses approximately 1000 L1 unit cells.
        return 1000 * self.error_at(d=self.distillation_l1_d)

    @property
    def l1_topo_error_t_gate(self) -> float:
        # It takes approximately 100 L1 unit cells to get the L1 state produced by the
        # factory to where it needs to be and perform the T gate.
        return 100 * self.error_at(d=self.distillation_l1_d)

    @property
    def l1_distillation_error(self) -> float:
        """The error due to level-0 faulty T states making it through distillation undetected.

        The level 1 distillation proceedure detects any two errors. There are 35 weight-three
        errors that can make it through undetected.
        """
        return 35 * self.l0_error**3

    @property
    def l1_error(self) -> float:
        """Chance of failure of a T gate performed with a T state produced from the L1 factory."""
        return self.l1_topo_error_factory + self.l1_topo_error_t_gate + self.l1_distillation_error

    # -------------------------------------------------------------------------------
    # ----     Level 2    ---------
    # -------------------------------------------------------------------------------

    @property
    def l2_error(self) -> float:
        """Chance of failure of the level two factory.

        This is the chance of failure of a CCZ gate or a pair of T gates performed with a CCZ state.
        """

        # The L2 CCZ factory and catalyzed T factory both use approximately 1000 L2 unit cells.
        l2_topo_error_factory = 1000 * self.error_at(d=self.distillation_l2_d)

        # Distillation error for this level.
        l2_distillation_error = 28 * self.l1_error**2

        return l2_topo_error_factory + l2_distillation_error

    # -------------------------------------------------------------------------------
    # ----     Distillation totals    ---------
    # -------------------------------------------------------------------------------

    @property
    def n_distillation_qubits(self) -> int:
        l1 = 4 * 8 * 2 * self.distillation_l1_d**2
        l2 = 4 * 8 * 2 * self.distillation_l2_d**2
        return 6 * l1 + l2

    @property
    def n_ccz_states(self) -> int:
        """Total number of CCZ states required.

        This includes CCZ states consumed to run catalysis.
        """
        return self.toffoli_count + math.ceil(self.t_count / 2)

    @property
    def distillation_error(self) -> float:
        """Error resulting from the magic state distillation part of the computation."""
        return self.l2_error * self.n_ccz_states

    @property
    def n_rounds(self) -> int:
        """The number of error-correction rounds to distill enough magic states.

        According to the approximations used in this cost model where we are limited by
        T state production, this sets the number of rounds for the whole computation. See
        `self.duration_hr` for this quantity in units of time.
        """

        distillation_d = max(2 * self.distillation_l1_d + 1, self.distillation_l2_d)
        catalyzations = math.ceil(self.t_count / 2)

        # Naive depth of 8.5, but can be overlapped to effective depth of 5.5
        # See section 2, paragraph 2 of the reference.
        ccz_depth = 5.5

        return math.ceil((self.n_ccz_states * ccz_depth + catalyzations) * distillation_d)

    # -------------------------------------------------------------------------------
    # ----     Data    ---------
    # -------------------------------------------------------------------------------

    def _code_distance_from_budget(self, budget: float) -> int:
        """Get the code distance that keeps one below the logical error `budget`."""

        # See: `self.error_at()`. p_l = a Λ^(-r) where r = (d+1)/2
        # Which we invert: r = ln(p_l/a) / ln(1/Λ)
        r = math.log(10 * budget) / math.log(100 * self.physical_error_rate)
        d = 2 * math.ceil(r) - 1
        if d < 3:
            return 3
        return d

    @property
    def n_logi_qubits(self) -> int:
        """Number of logical qubits including overhead.

        Note: the spreadsheet from the reference had a 50% overhead hardcoded for
        some of the cells using this quantity and variable (but set to 50% as default)
        for others.
        """
        return math.ceil((1 + self.routing_overhead) * self.n_alg_qubits)

    @property
    def data_code_distance(self) -> int:
        """The code distance for data qubits based on remaining error budget."""

        # Use "left over" budget for data qubits.
        err_budget = self.error_budget - self.distillation_error
        data_unit_cells = self.n_logi_qubits * self.n_rounds
        target_err_per_round = err_budget / data_unit_cells
        return self._code_distance_from_budget(budget=target_err_per_round)

    @property
    def n_data_qubits(self) -> int:
        n_phys_per_logical = 2 * self.data_code_distance**2
        return self.n_logi_qubits * n_phys_per_logical

    @property
    def data_error(self) -> float:
        data_unit_cells = self.n_logi_qubits * self.n_rounds
        error_per_unit_cell = self.error_at(d=self.data_code_distance)
        return data_unit_cells * error_per_unit_cell

    # -------------------------------------------------------------------------------
    # ----     Totals    ---------
    # -------------------------------------------------------------------------------

    @property
    def failure_prob(self) -> float:
        """Approximate probability of an error occurring during execution of the algorithm.

        This can be a bad CCZ being produced, a bad T state being produced,
        or a topological error occurring during the algorithm.
        """
        return self.distillation_error + self.data_error

    @property
    def n_phys_qubits(self) -> int:
        """Total physical qubits required to run algorithm."""
        return self.n_distillation_qubits + self.n_data_qubits

    @property
    def duration_hr(self) -> float:
        """Total time in hours to run algorithm, assuming no routing or Clifford bottlenecks."""
        return self.cycle_time_us * self.n_rounds / 1_000_000 / 60 / 60
