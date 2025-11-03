#  Copyright 2025 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import itertools
from typing import cast, Optional, Union

import attrs
import mpmath
import numpy as np

import qualtran.rotation_synthesis._typing as rst
import qualtran.rotation_synthesis.channels as channels
import qualtran.rotation_synthesis.math_config as mc
import qualtran.rotation_synthesis.protocols.protocol as rsp
import qualtran.rotation_synthesis.relative_norm as relative_norm
import qualtran.rotation_synthesis.rings as rings
from qualtran.rotation_synthesis.matrix import analytical_decomposition as rsad
from qualtran.rotation_synthesis.matrix import su2_ct
from qualtran.rotation_synthesis.protocols import diagonal, fallback, mixed_diagonal
from qualtran.rotation_synthesis.rings import zsqrt2

_DEFAULT_RELATIVE_NORM_SOLVER = relative_norm.CliffordTRelativeNormSolver()


def _solve(
    protocol: rsp.ApproxProblem,
    config: mc.MathConfig,
    collector: rsp.PointCollector,
    relative_norm_solver: relative_norm.CliffordTRelativeNormSolver = _DEFAULT_RELATIVE_NORM_SOLVER,
    verbose: bool = False,
) -> Optional[Union[list[channels.Channel], tuple[list[channels.Channel], list[channels.Channel]]]]:
    """Iterates over lattice points that satisify the geometric constraints defined by the protocol.

    Each valid point gets added to the collector which decides whether to terminate or not.

    Args:
        protocol: An ApproxProtocol instance that defines the geometric constraints of the problem.
        config: A math config.
        collector: A PointCollector object that decides when to terminate the loop and creates the result.
        relative_norm_solver: The relative norm solver to use.
        verbose: Whether to print debug statements.

    Returns:
        The result of calling collector.result() which is one of channel, list[channel] or None.
    """
    bad = 0
    total = 0
    for n, p in protocol.get_points(config, verbose):
        total += 1
        new = True
        m = n
        real_bound_fn = protocol.make_real_bound_fn(m, config)
        if not real_bound_fn(p):
            # if verbose:
            #     print('skip', p)
            bad += new
            if verbose and bad % 10**5 == 0:
                print(f"at {n=}", "seen", bad, "out of", total, "points, ratio =", bad / total)
            continue
        new = False
        if verbose:
            print(f"{n=} {m=} {p=}")
        q2 = 2 * zsqrt2.LAMBDA_KLIUCHNIKOV**n - (p * p.conjugate()).to_zsqrt2()[0]
        q = relative_norm_solver.solve(q2)
        if verbose:
            print(q)
        if q is None:
            continue
        assert q * q.conj() == rings.ZW.from_pair(q2, rings.ZSqrt2(0, 0))
        assert m == n, f"{m=} {n=}"
        if verbose:
            print("insert", p, q, m)
        collector.add_point(p, q, m, config)
        if collector.is_done():
            return collector.result()
        if verbose:
            print(collector.status())
    return None


def diagonal_unitary_approx(
    theta: rst.Real,
    eps: rst.Real,
    max_n: int,
    config: mc.MathConfig,
    relative_norm_solver: relative_norm.CliffordTRelativeNormSolver = _DEFAULT_RELATIVE_NORM_SOLVER,
    verbose: bool = False,
) -> Optional[channels.UnitaryChannel]:
    r"""Approximates $e^{i\theta}$ using the diagonal protocol.

    Args:
        theta: Target angle.
        eps: Target error.
        max_n: Maximum number of T gates to check.
        config: A math config.
        relative_norm_solver: The relative norm solver to use.
        verbose: whether to print debug statements or not.
    Returns:
        A Unitary channel.
    References:
        [Shorter quantum circuits via single-qubit gate approximation](https://arxiv.org/abs/2203.10064)
        section 3.2
    """
    theta = config.number(theta)
    eps = config.number(eps)
    theta %= 2 * config.pi
    protocol = diagonal.Diagonal(theta, eps, max_n)
    res = _solve(
        protocol,
        config,
        rsp.SimplePointCollector(target_count=1),
        relative_norm_solver=relative_norm_solver,
        verbose=verbose,
    )
    if res is not None:
        ch = res[0]
        assert isinstance(ch, channels.UnitaryChannel)
        return ch
    return None


def fallback_protocol(
    theta: rst.Real,
    eps: rst.Real,
    success_probability: rst.Real,
    max_n: int,
    config: mc.MathConfig,
    eps_rotation: Optional[rst.Real] = None,
    num_valid_points: int = 1,
    relative_norm_solver: relative_norm.CliffordTRelativeNormSolver = _DEFAULT_RELATIVE_NORM_SOLVER,
    verbose: bool = False,
) -> Optional[channels.ProjectiveChannel]:
    r"""Approximates $e^{i\theta}$ using the fallback protocol.

    Args:
        theta: Target angle.
        eps: Target error.
        success_probability: The target success probability of the projective measurement,
        max_n: Maximum number of T gates to check.
        config: A math config.
        eps_rotation: The error budget to use for the projective step, this parameter can be
            used to control the error budget split. If None, use most of `eps`.
        num_valid_points: The number of valid points to check, this parameter can be
            used to indirectly control the split of the error budget.
        relative_norm_solver: The relative norm solver to use.
        verbose: whether to print debug statements or not.
    Returns:
        A ProjectiveChannel.
    References:
        [Shorter quantum circuits via single-qubit gate approximation](https://arxiv.org/abs/2203.10064)
        section 3.3
    """
    theta = config.number(theta)
    eps = config.number(eps)
    theta %= 2 * config.pi
    # In https://arxiv.org/abs/2203.10064 section section 3.3 they suggest using eps/2, however
    # the dataset they provided uses more than half to get the best result.
    eps_rotation = eps_rotation if eps_rotation is not None else eps
    eps_rotation = config.number(eps_rotation)

    protocol: rsp.ApproxProblem = fallback.Fallback(
        theta, success_probability, eps_rotation, max_n, offset_angle=True
    )
    best_result = None
    best_cost = None
    projections = cast(
        list[channels.UnitaryChannel],
        _solve(
            protocol,
            config,
            rsp.SimplePointCollector(target_count=num_valid_points),
            relative_norm_solver=relative_norm_solver,
            verbose=verbose,
        ),
    )
    for rotation in projections:
        v = rotation.q.value(config.sqrt2) / zsqrt2.radius_at_n(
            zsqrt2.LAMBDA_KLIUCHNIKOV, rotation.n, config
        )
        arg_v = config.arctan2(v.imag, v.real)
        abs_v2 = (v * v.conjugate()).real
        # the expected error is (1-|v|^2) eps_rotation + |v|^2 * eps_correction
        err_proj = channels.ProjectiveChannel.diamond_distance_to_rz_on_measurement_success(
            rotation.p, theta, config
        )
        assert err_proj <= eps_rotation, f"err_proj={err_proj:e} eps_rotation={eps_rotation:e}"
        eps_correction = min((eps - err_proj) / max(abs_v2, 1 - success_probability), 0.1)
        protocol = diagonal.Diagonal(theta - arg_v, eps_correction, 400)
        correction_cands = cast(
            list[channels.UnitaryChannel],
            _solve(
                protocol,
                config,
                rsp.SimplePointCollector(target_count=1),
                relative_norm_solver=relative_norm_solver,
                verbose=verbose,
            ),
        )
        for correction in correction_cands:
            cand = channels.ProjectiveChannel(rotation, correction)
            cost = cand.diamond_norm_distance_to_rz(theta, config)
            if cost > eps:
                continue
            if best_cost is None or (cost < best_cost):
                best_cost = cost
                best_result = cand
    return best_result


def mixed_diagonal_protocol(
    theta: rst.Real,
    eps: rst.Real,
    max_n: int,
    config: mc.MathConfig,
    num_valid_points: int = 1,
    search_area_scaler: float = 1,
    relative_norm_solver: relative_norm.CliffordTRelativeNormSolver = _DEFAULT_RELATIVE_NORM_SOLVER,
    verbose: bool = False,
) -> Optional[channels.ProbabilisticChannel]:
    r"""Approximates $e^{i\theta}$ using the mixed diagonal protocol.

    Args:
        theta: Target angle.
        eps: Target error.
        max_n: Maximum number of T gates to check.
        config: A math config.
        num_valid_points: The number of valid points to check, this parameter can be
            used to indirectly control the split of the error budget.
        search_area_scaler: A scaler that allows the algorithm to expand the search area.
            This parameter can be used to indirectly control the error budget split.
        relative_norm_solver: The relative norm solver to use.
        verbose: whether to print debug statements or not.
    Returns:
        A ProbabilisticChannel channel.

    References:
        [Shorter quantum circuits via single-qubit gate approximation](https://arxiv.org/abs/2203.10064)
        section 3.4
    """
    theta = config.number(theta)
    eps = config.number(eps)
    theta %= 2 * config.pi
    protocol = mixed_diagonal.MixedDiagonal(theta, search_area_scaler * eps, max_n)
    under_rotations, over_rotations = cast(
        tuple[list[channels.UnitaryChannel], list[channels.UnitaryChannel]],
        _solve(
            protocol,
            config,
            rsp.SplitRegionCollector((config.sin(theta), -config.cos(theta), 0), num_valid_points),
            relative_norm_solver=relative_norm_solver,
            verbose=verbose,
        ),
    )
    best_choice: Optional[channels.ProbabilisticChannel] = None
    for u1, u2 in itertools.product(under_rotations, over_rotations):
        cand = channels.ProbabilisticChannel.from_unitary_channels(
            attrs.evolve(u1, twirl=True), attrs.evolve(u2, twirl=True), theta, config
        )
        if cand.diamond_norm_distance_to_rz(theta, config) > eps:
            continue
        if best_choice is None or (
            cand.expected_num_ts(config) < best_choice.expected_num_ts(config)
        ):
            best_choice = cand
    return best_choice


def mixed_fallback_protocol(
    theta: rst.Real,
    eps: rst.Real,
    success_probability: rst.Real,
    max_n: int,
    config: mc.MathConfig,
    eps_under_rotation: Optional[rst.Real] = None,
    eps_over_rotation: Optional[rst.Real] = None,
    num_valid_points: int = 1,
    fallback_max_n: int = 400,
    fallback_min_eps: rst.Real = mpmath.mpf("1e-32"),
    relative_norm_solver: relative_norm.CliffordTRelativeNormSolver = _DEFAULT_RELATIVE_NORM_SOLVER,
    verbose: bool = False,
) -> Optional[channels.ProbabilisticChannel]:
    r"""Approximates $e^{i\theta}$ using the mixed fallback protocol.

    Args:
        theta: Target angle.
        eps: Target error.
        success_probability: The target success probability of the projective measurement,
        max_n: Maximum number of T gates to check.
        config: A math config.
        eps_under_rotation: The error budget to use for the under rotation, this parameter can be
            used to control the error budget split. If None, use most of `eps`.
        eps_over_rotation: The error budget to use for the over rotation, this parameter can be
            used to control the error budget split. If None, use most of `eps`.
        num_valid_points: The number of valid points to check, this parameter can be
            used to indirectly control the split of the error budget.
        fallback_max_n: The maximum number of T gates to use for the correction step.
        fallback_min_eps: The minimum eps to use for the correction step,
        relative_norm_solver: The relative norm solver to use.
        verbose: whether to print debug statements or not.

    Returns:
        A ProbabilisticChannel channel.

    References:
        [Shorter quantum circuits via single-qubit gate approximation](https://arxiv.org/abs/2203.10064)
        section 3.5
    """
    theta = config.number(theta)
    eps = config.number(eps)
    theta %= 2 * config.pi
    eps_under_rotation = eps / 2 if eps_under_rotation is None else eps_under_rotation
    eps_over_rotation = eps / 2 if eps_over_rotation is None else eps_over_rotation

    eps_over_rotation = config.number(eps_over_rotation)
    eps_under_rotation = config.number(eps_under_rotation)

    delta_under_rotation = config.arcsin(config.sqrt(eps_under_rotation))
    new_eps_under_rotation = 2 * config.sin(delta_under_rotation / 2)
    delta_over_rotation = config.arcsin(config.sqrt(eps_over_rotation))
    new_eps_over_rotation = 2 * config.sin(delta_over_rotation / 2)

    protocol = fallback.Fallback(
        theta - delta_under_rotation / 2,
        success_probability,
        new_eps_under_rotation,
        max_n,
        offset_angle=False,
        filter_by_dist=False,
    )
    cands = cast(
        list[channels.UnitaryChannel],
        _solve(
            protocol,
            config,
            rsp.SimplePointCollector(num_valid_points),
            relative_norm_solver=relative_norm_solver,
            verbose=verbose,
        ),
    )
    if cands is None:
        raise ValueError(
            "no candidates for an under rotation, consider increasing "
            "max_n or using more digits of precision"
        )

    protocol = fallback.Fallback(
        theta + delta_over_rotation / 2,
        success_probability,
        new_eps_over_rotation,
        max_n,
        offset_angle=False,
        filter_by_dist=False,
    )
    over_cands = cast(
        list[channels.UnitaryChannel],
        _solve(
            protocol,
            config,
            rsp.SimplePointCollector(num_valid_points),
            relative_norm_solver=relative_norm_solver,
            verbose=verbose,
        ),
    )
    if over_cands is None:
        raise ValueError(
            "no candidates for an over rotation, consider increasing "
            "max_n or using more digits of precision"
        )

    cands += over_cands

    under_rotations = []
    over_rotations = []
    for r in cands:
        if r.rotation_angle(config) % (2 * config.pi) >= theta:
            over_rotations.append(r)
        else:
            under_rotations.append(r)

    best_result = None
    best_cost = None
    for under, over in itertools.product(under_rotations, over_rotations):
        assert isinstance(under, channels.UnitaryChannel)
        assert isinstance(over, channels.UnitaryChannel)
        delta1 = under.rotation_angle(config) - theta
        delta2 = over.rotation_angle(config) - theta
        eps_proj = 2 * max(config.sin(delta1) ** 2, config.sin(delta2) ** 2)
        if eps - eps_proj < fallback_min_eps:
            continue

        rem = (eps - eps_proj) / (1 - success_probability)
        rem = min(rem, 1e-1)
        correction_under = mixed_diagonal_protocol(
            theta - under.failure_angle(config),
            rem,
            fallback_max_n,
            config,
            num_valid_points=num_valid_points,
            search_area_scaler=1,
            verbose=verbose,
        )
        if correction_under is None:
            continue
        correction_over = mixed_diagonal_protocol(
            theta - over.failure_angle(config),
            rem,
            fallback_max_n,
            config,
            num_valid_points=num_valid_points,
            search_area_scaler=1,
            verbose=verbose,
        )

        if correction_over is None:
            continue

        under_channel = channels.ProjectiveChannel(under, correction_under)
        over_channel = channels.ProjectiveChannel(over, correction_over)

        cand = channels.ProbabilisticChannel.from_projective_channels(
            under_channel, over_channel, theta, config
        )
        cost = cand.diamond_norm_distance_to_rz(theta, config)
        if cost > eps:
            continue
        if best_cost is None or (cost < best_cost):
            best_cost = cost
            best_result = cand
    return best_result


def magnitude_approx(
    unitary: np.ndarray,
    eps: rst.Real,
    max_n: int,
    config: mc.MathConfig,
    eps_split: Optional[tuple[rst.Real, rst.Real, rst.Real]] = None,
    relative_norm_solver: relative_norm.CliffordTRelativeNormSolver = _DEFAULT_RELATIVE_NORM_SOLVER,
    verbose: bool = False,
) -> Optional[channels.UnitaryChannel]:
    r"""Approximates a unitary using the magnitude approximation protocol.

    Any $SU(2)$ unitary can be written as a product of 3 rotations ZXZ. This method computes these
    rotation angles (and ignores the global phase if the given unitary is not in $SU(2)$) and
    approximates each of them independently before merging the results to obtain a single unitary.

    Args:
        unitary: the target unitary, this can be 2x2 numpy array of mpmath.mpc objects.
        eps: Target error.
        max_n: Maximum number of T gates to check.
        config: A math config.
        eps_split: Optional splitting of the error budget for three rotations.
            If None, `eps` is split into $0.14\epsilon$ for the X-rotation and $0.43\epsilon$ for
            each of the Z-rotations.
        relative_norm_solver: The relative norm solver to use.
        verbose: whether to print debug statements or not.
    Returns:
        A UnitaryChannels or None.
    References:
        [Shorter quantum circuits via single-qubit gate approximation](https://arxiv.org/abs/2203.10064)
        section 3.1
    """
    eps = config.number(eps)
    det = unitary[0, 0] * unitary[1, 1] - unitary[1, 0] * unitary[0, 1]
    unitary = unitary / config.sqrt(det)
    if eps_split is None:
        eps_split = 0.43 * eps, 0.14 * eps, 0.43 * eps
    phi1, theta, phi2 = rsad.su_unitary_to_zxz_angles(unitary, config)
    rx_approx = diagonal_unitary_approx(
        -theta / 2,
        eps=eps_split[1],
        max_n=max_n,
        config=config,
        relative_norm_solver=relative_norm_solver,
        verbose=verbose,
    )
    if rx_approx is None:
        return None
    x_rotation = (su2_ct.HSqrt2 @ rx_approx.to_matrix() @ su2_ct.HSqrt2.adjoint()).numpy(config)

    angles_for_approx_x_rot = rsad.su_unitary_to_zxz_angles(x_rotation, config)

    rz1_approx = diagonal_unitary_approx(
        -(phi1 - angles_for_approx_x_rot[0]) / 2,
        eps_split[0],
        max_n=max_n,
        config=config,
        relative_norm_solver=relative_norm_solver,
        verbose=verbose,
    )
    if rz1_approx is None:
        return None
    rz2_approx = diagonal_unitary_approx(
        -(phi2 - angles_for_approx_x_rot[2]) / 2,
        eps_split[2],
        max_n=max_n,
        config=config,
        relative_norm_solver=relative_norm_solver,
        verbose=verbose,
    )
    if rz2_approx is None:
        return None
    return channels.UnitaryChannel.from_unitaries(
        rz1_approx, su2_ct.HSqrt2, rx_approx, su2_ct.HSqrt2.adjoint(), rz2_approx
    )
