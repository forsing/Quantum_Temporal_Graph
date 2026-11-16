# Quantum_Temporal_Graph
# Finding Causal Signal in Complex Systems



from __future__ import annotations

import csv
import hashlib
import itertools
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit
from scipy.optimize import minimize
from scipy.special import gammaincc
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


random.seed(39)
np.random.seed(39)



FILES = (
    Path("/Users/4c/Desktop/GHQ/data/loto7_4720_k91.csv"),
    Path("/Users/4c/Desktop/GHQ/data/loto7_4720_k91_loto_2982.csv"),
    Path("/Users/4c/Desktop/GHQ/data/loto7_4720_k91_loto_plus_1738.csv"),
)



N = 39
K = 7

MAX_LAG = 3
SCREEN_LIMIT = 6
MAX_TEMPORAL_PARENTS = 2
SCREEN_ALPHA = 0.20
BY_LEVEL = 0.10

REGULARIZATIONS = (10.0, 1.0, 0.1)

TRAIN_FRACTION = 0.70
VALIDATION_END = 0.85

GRADIENT_TOLERANCE = 2e-6
MAX_BFGS_ITERATIONS = 700
MAX_CG_ITERATIONS = 900

MAX_CONSECUTIVE = 4
DIAGNOSTICS = False


# ------------------------------------------------------------
# MATEMATICKA KONSTRUKCIJA
#
# 1. Vremenski graf:
#    - diskretni uslovni G² testovi;
#    - ogranicena PC selekcija na prvom delu treninga;
#    - MCI tip uslovljavanja: roditelji cilja i izvora;
#    - potvrda u dva kasnija dela treninga;
#    - BY korekcija nominalnih p-vrednosti.
#
# Ovo je ogranicena adaptacija principa PC/MCI.
# Nije puna implementacija biblioteke Tigramite.
# G² p-vrednosti su asimptotske. Rezultati se koriste kao
# prediktivne veze, ne kao dokaz fizicke uzrocnosti.
#
# 2. Amplitudni model:
#    Istorijske veze odredjuju kontekst RY rotacija.
#    Zavisnosti unutar ciljne sedmorke predstavljene su stablom
#    naucenim iskljucivo iz dostupnog treninga.
#
#    Za cvor j, njegovog roditelja a i istorijski kontekst c:
#
#      angle = pi/2 + atan(z[j, c, a])
#
#      amplitude(0) = cos(angle/2)
#      amplitude(1) = sin(angle/2)
#
#    Kvantno stablo daje amplitudu celog binarnog stanja.
#    Raspodela se tacno uslovljava na sedam ukljucenih brojeva.
#
# 3. Lokalna simulacija:
#    Koristi se faktorizovana amplitudna reprezentacija.
#    Ne alocira se veliki statevector.
#    Sedmorka ima kanonsko kodiranje u sedam registara
#    po sest qubita. Unutrasnji racun koristi indikatore
#    prisustva brojeva.
#
#    Qiskit proverava stvarno kolo kontrolisanih RY rotacija
#    na malom stablu, njegove amplitude i postselekciju.
#
# 4. Normalizacija:
#    Sum-product algoritam po stablu sa brojacem 0..7.
#
# 5. NEXT:
#    Max-product algoritam po stablu.
#    Ako njegov maksimum krsi ogranicenje uzastopnog niza,
#    koristi se tacna branch-and-bound pretraga.
#
# Nema projekcije Statevector -> 7x39.
# Nema klasifikatora, slucajnog uzorkovanja ili korekcije
# rezultata radi dobijanja razlicitih NEXT kombinacija.
# ------------------------------------------------------------


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_draws(path):
    if not path.is_file():
        raise FileNotFoundError(f"CSV nije pronadjen: {path}")

    rows = []

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(8192)
        handle.seek(0)

        try:
            dialect = csv.Sniffer().sniff(
                sample, delimiters=",;\t "
            )
        except csv.Error as exc:
            raise ValueError(
                "Nije prepoznat razdelnik CSV-a."
            ) from exc

        for line_number, row in enumerate(
            csv.reader(handle, dialect), start=1
        ):
            fields = [field.strip() for field in row if field.strip()]

            if not fields:
                continue

            if len(fields) != K:
                raise ValueError(
                    f"Red {line_number}: ocekujem {K} brojeva."
                )

            try:
                values = [int(field) for field in fields]
            except ValueError as exc:
                raise ValueError(
                    f"Red {line_number}: neispravan ceo broj."
                ) from exc

            if (
                len(set(values)) != K
                or min(values) < 1
                or max(values) > N
            ):
                raise ValueError(
                    f"Red {line_number}: neispravna kombinacija."
                )

            rows.append(sorted(value - 1 for value in values))

    draws = np.asarray(rows, dtype=np.int64)

    if len(draws) < 300:
        raise ValueError(
            "Za ovaj postupak potrebno je najmanje 300 redova."
        )

    return draws


def binary_draws(draws, n):
    result = np.zeros((len(draws), n), dtype=np.uint8)
    result[np.arange(len(draws))[:, None], draws] = 1
    return result


# ------------------------------------------------------------
# USLOVNI VREMENSKI GRAF
# ------------------------------------------------------------


@njit(cache=False)
def g_squared(data, times, target, source, lag, conditions):
    strata = 1 << len(conditions)
    counts = np.zeros((strata, 2, 2), dtype=np.float64)

    for t in times:
        state = 0

        for q in range(len(conditions)):
            variable = conditions[q, 0]
            delay = conditions[q, 1]
            state = (state << 1) | int(data[t - delay, variable])

        x = int(data[t - lag, source])
        y = int(data[t, target])
        counts[state, x, y] += 1.0

    statistic = 0.0
    degrees = 0

    for state in range(strata):
        total = counts[state].sum()

        if total <= 0.0:
            continue

        row0 = counts[state, 0, 0] + counts[state, 0, 1]
        row1 = counts[state, 1, 0] + counts[state, 1, 1]
        col0 = counts[state, 0, 0] + counts[state, 1, 0]
        col1 = counts[state, 0, 1] + counts[state, 1, 1]

        if min(row0, row1, col0, col1) <= 0.0:
            continue

        degrees += 1

        for x in range(2):
            row_total = row0 if x == 0 else row1

            for y in range(2):
                observed = counts[state, x, y]

                if observed <= 0.0:
                    continue

                col_total = col0 if y == 0 else col1
                expected = row_total * col_total / total
                statistic += 2.0 * observed * math.log(
                    observed / expected
                )

    return max(statistic, 0.0), degrees


def conditional_p(data, times, target, edge, conditions=()):
    conditions = np.asarray(
        sorted(set(conditions)), dtype=np.int64
    ).reshape(-1, 2)

    statistic, degrees = g_squared(
        data,
        times,
        target,
        edge[0],
        edge[1],
        conditions,
    )

    if degrees == 0 or len(times) < max(40, 8 * degrees):
        return 1.0

    return float(gammaincc(0.5 * degrees, 0.5 * statistic))


def by_adjustment(p_values):
    flat = np.asarray(p_values, dtype=np.float64).ravel()
    count = len(flat)
    order = np.argsort(flat, kind="stable")

    harmonic = np.sum(1.0 / np.arange(1, count + 1))
    adjusted_sorted = (
        flat[order] * count * harmonic
        / np.arange(1, count + 1)
    )

    adjusted_sorted = np.minimum.accumulate(
        adjusted_sorted[::-1]
    )[::-1]

    result = np.empty_like(flat)
    result[order] = np.minimum(adjusted_sorted, 1.0)
    return result.reshape(np.shape(p_values))


def discover_temporal_graph(data):
    n = data.shape[1]
    first_complete_time = 2 * MAX_LAG

    times = np.arange(
        first_complete_time, len(data), dtype=np.int64
    )

    if len(times) < 150:
        return tuple(() for _ in range(n))

    first_end = int(0.40 * len(times))
    second_end = int(0.70 * len(times))

    screening_times = times[:first_end]
    confirmation_a = times[first_end:second_end]
    confirmation_b = times[second_end:]

    edges = [
        (source, lag)
        for lag in range(1, MAX_LAG + 1)
        for source in range(n)
    ]

    edge_index = {edge: i for i, edge in enumerate(edges)}
    preliminary = []

    for target in range(n):
        ranking = []

        for edge in edges:
            p = conditional_p(
                data, screening_times, target, edge
            )
            ranking.append((p, edge))

        ranking.sort(key=lambda item: (item[0], item[1]))

        candidates = [
            edge
            for p, edge in ranking[:SCREEN_LIMIT]
            if p <= SCREEN_ALPHA
        ]

        unconditional = {edge: p for p, edge in ranking}
        survivors = []

        # PC-stable tip koraka sa jednim uslovom.
        # Skup kandidata se ne menja usred ovog koraka.
        for edge in candidates:
            worst_p = unconditional[edge]

            for other in candidates:
                if other == edge:
                    continue

                p = conditional_p(
                    data,
                    screening_times,
                    target,
                    edge,
                    (other,),
                )

                worst_p = max(worst_p, p)

                if worst_p > SCREEN_ALPHA:
                    break

            if worst_p <= SCREEN_ALPHA:
                survivors.append((worst_p, edge))

        survivors.sort(key=lambda item: (item[0], item[1]))

        preliminary.append(
            tuple(
                edge
                for _, edge in survivors[:MAX_TEMPORAL_PARENTS]
            )
        )

    p_values = np.ones(
        (n, len(edges)), dtype=np.float64
    )

    for target in range(n):
        for edge in preliminary[target]:
            source, lag = edge

            conditions = [
                other
                for other in preliminary[target]
                if other != edge
            ]

            # Jedan prethodno odabran roditelj izvora,
            # pomeren na vreme samog izvora.
            if preliminary[source]:
                variable, source_lag = preliminary[source][0]
                conditions.append(
                    (variable, lag + source_lag)
                )

            conditions = tuple(
                sorted(set(c for c in conditions if c != edge))
            )

            p_a = conditional_p(
                data, confirmation_a, target, edge, conditions
            )
            p_b = conditional_p(
                data, confirmation_b, target, edge, conditions
            )

            # Veza mora proci oba hronoloska dela.
            p_values[target, edge_index[edge]] = max(p_a, p_b)

    adjusted = by_adjustment(p_values)
    result = []

    for target in range(n):
        retained = [
            edge
            for edge in preliminary[target]
            if adjusted[target, edge_index[edge]] <= BY_LEVEL
        ]

        result.append(tuple(sorted(retained)))

    return tuple(result)


# ------------------------------------------------------------
# STABLO ZAJEDNICKE RASPODELE
# ------------------------------------------------------------


def mutual_information_matrix(targets):
    n = targets.shape[1]
    result = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i):
            codes = (
                2 * targets[:, i].astype(np.int64)
                + targets[:, j]
            )

            table = np.bincount(
                codes, minlength=4
            ).reshape(2, 2).astype(np.float64)

            table /= len(targets)
            rows = table.sum(axis=1)
            columns = table.sum(axis=0)

            information = 0.0

            for a in range(2):
                for b in range(2):
                    p = table[a, b]

                    if p > 0.0:
                        information += p * math.log(
                            p / (rows[a] * columns[b])
                        )

            result[i, j] = result[j, i] = max(
                information, 0.0
            )

    return result


def learn_output_tree(targets):
    """
    Stablo je faktorizacija zajednickog ishoda.
    Njegove grane se ne nazivaju uzrocima.
    Numericka blizina oznaka brojeva nije kriterijum.
    """
    information = mutual_information_matrix(targets)
    n = information.shape[0]

    root = int(np.argmax(information.sum(axis=1)))
    parent = np.full(n, -1, dtype=np.int64)
    used = np.zeros(n, dtype=bool)
    keys = information[root].copy()

    used[root] = True
    parent[:] = root
    parent[root] = -1
    order = [root]

    for _ in range(n - 1):
        candidates = np.where(used, -np.inf, keys)
        node = int(np.argmax(candidates))
        used[node] = True
        order.append(node)

        for other in range(n):
            if not used[other] and information[node, other] > keys[other]:
                keys[other] = information[node, other]
                parent[other] = node

    return parent, np.asarray(order, dtype=np.int64)


@dataclass
class Structure:
    temporal: tuple
    parent: np.ndarray
    order: np.ndarray
    child_ptr: np.ndarray
    children: np.ndarray
    prefix_ptr: np.ndarray
    parameter_ptr: np.ndarray
    k: int

    def arguments(self):
        return (
            self.parent,
            self.order,
            self.child_ptr,
            self.children,
            self.prefix_ptr,
            self.parameter_ptr,
            self.k,
        )

    @property
    def parameter_count(self):
        return int(self.parameter_ptr[-1])

    @property
    def temporal_edge_count(self):
        return sum(len(edges) for edges in self.temporal)


def assemble_structure(temporal, parent, order, k):
    n = len(parent)
    child_lists = [
        [j for j in range(n) if parent[j] == i]
        for i in range(n)
    ]

    child_ptr = np.zeros(n + 1, dtype=np.int64)
    prefix_ptr = np.zeros(n + 1, dtype=np.int64)
    parameter_ptr = np.zeros(n + 1, dtype=np.int64)

    children = []

    for j in range(n):
        children.extend(child_lists[j])
        child_ptr[j + 1] = len(children)

        prefix_ptr[j + 1] = (
            prefix_ptr[j] + len(child_lists[j]) + 1
        )

        context_count = 3 ** len(temporal[j])
        parameter_ptr[j + 1] = (
            parameter_ptr[j] + 2 * context_count
        )

    return Structure(
        tuple(temporal),
        np.asarray(parent, dtype=np.int64),
        np.asarray(order, dtype=np.int64),
        child_ptr,
        np.asarray(children, dtype=np.int64),
        prefix_ptr,
        parameter_ptr,
        k,
    )


def learn_structure(history, k):
    temporal = discover_temporal_graph(history)
    parent, order = learn_output_tree(history[1:])

    return assemble_structure(
        temporal, parent, order, k
    )


def make_contexts(history, times, temporal):
    """
    Kontekst za ishod t koristi samo redove pre t.

    Ternarne oznake:
      0: istorija jos nije dostupna;
      1: broj nije bio prisutan;
      2: broj je bio prisutan.

    Tako se zadrzavaju i najraniji istorijski prelazi.
    """
    result = np.zeros(
        (len(times), history.shape[1]), dtype=np.int64
    )

    for node, edges in enumerate(temporal):
        multiplier = 1

        for source, lag in edges:
            indices = times - lag
            known = indices >= 0

            states = np.zeros(len(times), dtype=np.int64)
            states[known] = (
                1 + history[indices[known], source].astype(np.int64)
            )

            result[:, node] += multiplier * states
            multiplier *= 3

    return result


@dataclass
class Batch:
    contexts: np.ndarray
    counts: np.ndarray
    weights: np.ndarray

    @property
    def size(self):
        return float(self.weights.sum())


def make_batch(contexts, targets, structure):
    unique, inverse = np.unique(
        contexts, axis=0, return_inverse=True
    )

    groups = len(unique)
    n = targets.shape[1]

    weights = np.bincount(
        inverse, minlength=groups
    ).astype(np.float64)

    counts = np.zeros(
        (groups, n, 2, 2), dtype=np.float64
    )

    for node in range(n):
        parent = structure.parent[node]

        if parent < 0:
            parent_state = np.zeros(
                len(targets), dtype=np.int64
            )
        else:
            parent_state = targets[:, parent].astype(np.int64)

        np.add.at(
            counts[:, node],
            (
                inverse,
                parent_state,
                targets[:, node].astype(np.int64),
            ),
            1.0,
        )

    return Batch(
        np.ascontiguousarray(unique),
        np.ascontiguousarray(counts),
        weights,
    )


# ------------------------------------------------------------
# KVANTNE AMPLITUDE I TACAN NLL
# ------------------------------------------------------------


@njit(cache=False)
def gate_factors(parameters, context, parameter_ptr):
    n = len(context)

    log_factors = np.empty((n, 2, 2), dtype=np.float64)
    factors = np.empty((n, 2, 2), dtype=np.float64)
    derivatives = np.empty((n, 2, 2), dtype=np.float64)

    for node in range(n):
        for parent_state in range(2):
            index = (
                parameter_ptr[node]
                + 2 * context[node]
                + parent_state
            )

            z = parameters[index]
            radius = math.hypot(1.0, z)
            common = math.log(2.0) + math.log(radius)
            transformed = math.asinh(z)

            for state in range(2):
                sign = 2.0 * state - 1.0

                log_p = sign * transformed - common
                log_factors[node, parent_state, state] = log_p
                factors[node, parent_state, state] = math.exp(log_p)

                derivatives[node, parent_state, state] = (
                    sign / radius - (z / radius) / radius
                )

    return log_factors, factors, derivatives


@njit(cache=False)
def forward_tree(factors, order, child_ptr, children, prefix_ptr, k):
    n = len(order)

    messages = np.zeros(
        (n, 2, k + 1), dtype=np.float64
    )
    prefixes = np.zeros(
        (prefix_ptr[-1], 2, k + 1), dtype=np.float64
    )

    for position in range(n - 1, -1, -1):
        node = order[position]
        start = prefix_ptr[node]

        prefixes[start, 0, 0] = 1.0

        if k >= 1:
            prefixes[start, 1, 1] = 1.0

        child_count = child_ptr[node + 1] - child_ptr[node]

        for r in range(child_count):
            child = children[child_ptr[node] + r]

            for state in range(2):
                for a in range(k + 1):
                    left = prefixes[start + r, state, a]

                    if left == 0.0:
                        continue

                    for b in range(k + 1 - a):
                        prefixes[start + r + 1, state, a + b] += (
                            left * messages[child, state, b]
                        )

        last = start + child_count

        for parent_state in range(2):
            for count in range(k + 1):
                messages[node, parent_state, count] = (
                    factors[node, parent_state, 0]
                    * prefixes[last, 0, count]
                    + factors[node, parent_state, 1]
                    * prefixes[last, 1, count]
                )

    return messages, prefixes


@njit(cache=False)
def group_loss(
    parameters,
    context,
    observed_counts,
    weight,
    parent,
    order,
    child_ptr,
    children,
    prefix_ptr,
    parameter_ptr,
    k,
    need_gradient,
):
    n = len(parent)

    log_factors, factors, derivatives = gate_factors(
        parameters, context, parameter_ptr
    )

    messages, prefixes = forward_tree(
        factors, order, child_ptr, children, prefix_ptr, k
    )

    root = order[0]
    normalization = messages[root, 0, k]

    if normalization <= 0.0 or not math.isfinite(normalization):
        raise ValueError("Neispravna normalizacija stabla.")

    log_z = math.log(normalization)
    loss = weight * log_z
    gradient = np.zeros(len(parameters), dtype=np.float64)

    for node in range(n):
        for a in range(2):
            for b in range(2):
                loss -= (
                    observed_counts[node, a, b]
                    * log_factors[node, a, b]
                )

    if not need_gradient:
        return loss, gradient, log_z

    adjoint = np.zeros_like(messages)
    adjoint[root, 0, k] = weight / normalization

    for position in range(n):
        node = order[position]
        start = prefix_ptr[node]
        child_count = child_ptr[node + 1] - child_ptr[node]
        last = start + child_count

        adj_prefix = np.zeros((2, k + 1), dtype=np.float64)

        for a in range(2):
            index = (
                parameter_ptr[node] + 2 * context[node] + a
            )

            for b in range(2):
                derivative_factor = 0.0

                for count in range(k + 1):
                    derivative_factor += (
                        adjoint[node, a, count]
                        * prefixes[last, b, count]
                    )

                    adj_prefix[b, count] += (
                        adjoint[node, a, count]
                        * factors[node, a, b]
                    )

                expected_count = (
                    factors[node, a, b] * derivative_factor
                )

                gradient[index] += (
                    expected_count - observed_counts[node, a, b]
                ) * derivatives[node, a, b]

        for r in range(child_count - 1, -1, -1):
            child = children[child_ptr[node] + r]
            previous_adjoint = np.zeros_like(adj_prefix)

            for state in range(2):
                for a in range(k + 1):
                    previous_value = prefixes[start + r, state, a]

                    for b in range(k + 1 - a):
                        influence = adj_prefix[state, a + b]

                        adjoint[child, state, b] += (
                            influence * previous_value
                        )

                        previous_adjoint[state, a] += (
                            influence * messages[child, state, b]
                        )

            adj_prefix = previous_adjoint

    return loss, gradient, log_z


@njit(cache=False)
def batch_loss(
    parameters,
    contexts,
    counts,
    weights,
    parent,
    order,
    child_ptr,
    children,
    prefix_ptr,
    parameter_ptr,
    k,
    need_gradient,
):
    losses = np.empty(len(contexts), dtype=np.float64)
    gradient = np.zeros(len(parameters), dtype=np.float64)

    for group in range(len(contexts)):
        value, local_gradient, _ = group_loss(
            parameters,
            contexts[group],
            counts[group],
            weights[group],
            parent,
            order,
            child_ptr,
            children,
            prefix_ptr,
            parameter_ptr,
            k,
            need_gradient,
        )

        losses[group] = value

        if need_gradient:
            gradient += local_gradient

    gradient /= weights.sum()
    return losses, gradient


@dataclass
class Model:
    structure: Structure
    parameters: np.ndarray
    regularization: float


def warm_parameters(structure, warm):
    result = np.zeros(
        structure.parameter_count, dtype=np.float64
    )

    if warm is None:
        return result

    old = warm.structure

    for node in range(len(structure.parent)):
        if (
            structure.parent[node] == old.parent[node]
            and structure.temporal[node] == old.temporal[node]
        ):
            new_slice = slice(
                structure.parameter_ptr[node],
                structure.parameter_ptr[node + 1],
            )
            old_slice = slice(
                old.parameter_ptr[node],
                old.parameter_ptr[node + 1],
            )

            result[new_slice] = warm.parameters[old_slice]

    return result


def train_model(structure, batch, regularization, warm=None):
    start = warm_parameters(structure, warm)
    anchor = start.copy()

    anchor_losses, _ = batch_loss(
        anchor,
        batch.contexts,
        batch.counts,
        batch.weights,
        *structure.arguments(),
        False,
    )

    usage = np.zeros_like(start)

    for node in range(len(structure.parent)):
        for a in range(2):
            indices = (
                structure.parameter_ptr[node]
                + 2 * batch.contexts[:, node]
                + a
            )

            np.add.at(
                usage,
                indices,
                batch.counts[:, node, a, :].sum(axis=1),
            )

    coordinate_scale = 1.0 / np.sqrt(
        0.25 * usage / batch.size
        + regularization / len(start)
        + 1e-8
    )
    coordinate_scale = np.minimum(coordinate_scale, 40.0)

    best_value = math.inf
    best_norm = math.inf
    best_parameters = start.copy()

    def objective(coordinates):
        nonlocal best_value, best_norm, best_parameters

        parameters = coordinates * coordinate_scale

        losses, gradient = batch_loss(
            parameters,
            batch.contexts,
            batch.counts,
            batch.weights,
            *structure.arguments(),
            True,
        )

        value = float(
            np.sum(losses - anchor_losses) / batch.size
            + 0.5 * regularization
            * np.mean((parameters - anchor) * (parameters + anchor))
        )

        gradient += regularization * parameters / len(parameters)
        norm = float(np.max(np.abs(gradient)))

        if not math.isfinite(value) or not np.all(np.isfinite(gradient)):
            raise FloatingPointError("Nekonacan cilj ili gradijent.")

        if (
            value < best_value
            or (
                abs(value - best_value) < 1e-13
                and norm < best_norm
            )
        ):
            best_value = value
            best_norm = norm
            best_parameters = parameters.copy()

        return value, gradient * coordinate_scale

    coordinates = start / coordinate_scale

    for method, iterations in (
        ("BFGS", MAX_BFGS_ITERATIONS),
        ("CG", MAX_CG_ITERATIONS),
    ):
        result = minimize(
            objective,
            coordinates,
            method=method,
            jac=True,
            options={
                "maxiter": iterations,
                "gtol": (
                    GRADIENT_TOLERANCE
                    * min(1.0, float(coordinate_scale.min()))
                    * 0.25
                ),
                "disp": False,
            },
        )

        objective(result.x)

        if best_norm <= GRADIENT_TOLERANCE:
            return Model(
                structure,
                best_parameters.copy(),
                regularization,
            )

        coordinates = best_parameters / coordinate_scale

        if DIAGNOSTICS:
            print(
                f"{method}: gradijent={best_norm:.6g}; "
                f"iteracije={result.nit}."
            )

    raise RuntimeError(
        f"Optimizacija nije konvergirala; gradijent={best_norm:.6g}."
    )


def evaluate_model(model, batch):
    losses, _ = batch_loss(
        model.parameters,
        batch.contexts,
        batch.counts,
        batch.weights,
        *model.structure.arguments(),
        False,
    )
    return float(losses.sum() / batch.size)


# ------------------------------------------------------------
# TACAN DEKODER
# ------------------------------------------------------------


@njit(cache=False)
def forbidden_run(mask, maximum):
    intersection = mask

    for shift in range(1, maximum + 1):
        intersection &= mask >> shift

    return intersection != 0


@njit(cache=False)
def tree_maximum(log_factors, order, child_ptr, children, k):
    n = len(order)

    scores = np.full((n, 2, k + 1), -np.inf)
    masks = np.zeros((n, 2, k + 1), dtype=np.int64)

    for position in range(n - 1, -1, -1):
        node = order[position]

        local_scores = np.full((2, k + 1), -np.inf)
        local_masks = np.zeros((2, k + 1), dtype=np.int64)

        local_scores[0, 0] = 0.0

        if k >= 1:
            local_scores[1, 1] = 0.0
            local_masks[1, 1] = np.int64(1) << node

        for edge in range(child_ptr[node], child_ptr[node + 1]):
            child = children[edge]
            updated = np.full_like(local_scores, -np.inf)
            updated_masks = np.zeros_like(local_masks)

            for state in range(2):
                for a in range(k + 1):
                    if not math.isfinite(local_scores[state, a]):
                        continue

                    for b in range(k + 1 - a):
                        if not math.isfinite(scores[child, state, b]):
                            continue

                        value = (
                            local_scores[state, a]
                            + scores[child, state, b]
                        )
                        mask = (
                            local_masks[state, a]
                            | masks[child, state, b]
                        )

                        if (
                            value > updated[state, a + b]
                            or (
                                value == updated[state, a + b]
                                and mask < updated_masks[state, a + b]
                            )
                        ):
                            updated[state, a + b] = value
                            updated_masks[state, a + b] = mask

            local_scores = updated
            local_masks = updated_masks

        for a in range(2):
            for count in range(k + 1):
                for b in range(2):
                    value = (
                        log_factors[node, a, b]
                        + local_scores[b, count]
                    )
                    mask = local_masks[b, count]

                    if (
                        value > scores[node, a, count]
                        or (
                            value == scores[node, a, count]
                            and mask < masks[node, a, count]
                        )
                    ):
                        scores[node, a, count] = value
                        masks[node, a, count] = mask

    root = order[0]
    return masks[root, 0, k], scores[root, 0, k]


def binary_potentials(log_factors, parent):
    n = len(parent)
    constant = 0.0
    fields = np.zeros(n)
    interactions = np.zeros((n, n))

    for node in range(n):
        table = log_factors[node]
        constant += table[0, 0]
        fields[node] += table[0, 1] - table[0, 0]

        p = parent[node]

        if p >= 0:
            fields[p] += table[1, 0] - table[0, 0]

            interaction = (
                table[1, 1] - table[1, 0]
                - table[0, 1] + table[0, 0]
            )

            interactions[node, p] += interaction
            interactions[p, node] += interaction

    return constant, fields, interactions


@njit(cache=False)
def bounded_search(
    position,
    remaining,
    score,
    mask,
    order,
    interactions,
    positive_suffix,
    gains,
    best_score,
    best_mask,
    maximum_run,
):
    n = len(order)

    if remaining == 0:
        if (
            score > best_score[0]
            or (
                score == best_score[0]
                and mask < best_mask[0]
            )
        ):
            best_score[0] = score
            best_mask[0] = mask
        return

    if n - position < remaining:
        return

    # Gornja granica:
    # doprinos vec izabranih + najpovoljniji preostali cvorovi
    # + precenjen pozitivan doprinos buducih parova.
    top = np.full(remaining, -np.inf)

    for j in range(position, n):
        value = (
            gains[position, j]
            + 0.5 * positive_suffix[position, j]
        )

        for r in range(remaining):
            if value > top[r]:
                for s in range(remaining - 1, r, -1):
                    top[s] = top[s - 1]
                top[r] = value
                break

    upper = score + top.sum()
    tolerance = 1e-10 * (1.0 + abs(upper))

    if upper < best_score[0] - tolerance:
        return

    bit = np.int64(1) << order[position]
    included_mask = mask | bit

    if not forbidden_run(included_mask, maximum_run):
        for j in range(n):
            gains[position + 1, j] = (
                gains[position, j] + interactions[position, j]
            )

        bounded_search(
            position + 1,
            remaining - 1,
            score + gains[position, position],
            included_mask,
            order,
            interactions,
            positive_suffix,
            gains,
            best_score,
            best_mask,
            maximum_run,
        )

    if n - position - 1 >= remaining:
        gains[position + 1] = gains[position]

        bounded_search(
            position + 1,
            remaining,
            score,
            mask,
            order,
            interactions,
            positive_suffix,
            gains,
            best_score,
            best_mask,
            maximum_run,
        )


def constrained_maximum(log_factors, structure, maximum_run):
    mask, score = tree_maximum(
        log_factors,
        structure.order,
        structure.child_ptr,
        structure.children,
        structure.k,
    )

    if not forbidden_run(mask, maximum_run):
        return int(mask), float(score)

    constant, fields, interactions = binary_potentials(
        log_factors, structure.parent
    )

    priority = fields + 0.5 * np.maximum(
        interactions, 0.0
    ).sum(axis=1)

    order = np.lexsort(
        (np.arange(len(fields)), -priority)
    ).astype(np.int64)

    ordered_fields = fields[order]
    ordered_interactions = np.ascontiguousarray(
        interactions[np.ix_(order, order)]
    )

    n = len(order)
    positive_suffix = np.zeros((n + 1, n))

    for position in range(n - 1, -1, -1):
        positive_suffix[position] = (
            positive_suffix[position + 1]
            + np.maximum(ordered_interactions[:, position], 0.0)
        )

    gains = np.zeros((n + 1, n))
    gains[0] = ordered_fields

    best_score = np.array([-math.inf])
    best_mask = np.array([0], dtype=np.int64)

    bounded_search(
        0,
        structure.k,
        0.0,
        np.int64(0),
        order,
        ordered_interactions,
        positive_suffix,
        gains,
        best_score,
        best_mask,
        maximum_run,
    )

    require(
        math.isfinite(best_score[0]),
        "Nije pronadjena dozvoljena NEXT kombinacija.",
    )

    return int(best_mask[0]), float(constant + best_score[0])


def mask_log_probability(mask, log_factors, parent, log_z):
    value = -log_z

    for node in range(len(parent)):
        state = (mask >> node) & 1
        p = parent[node]
        parent_state = 0 if p < 0 else (mask >> p) & 1
        value += log_factors[node, parent_state, state]

    return float(value)


def predict_next(model, context):
    structure = model.structure

    log_factors, factors, _ = gate_factors(
        model.parameters, context, structure.parameter_ptr
    )

    # Za NEXT proverava se i cela raspodela broja ukljucenih
    # elemenata 0..N, ne samo koeficijent za sedmorku.
    messages, _ = forward_tree(
        factors,
        structure.order,
        structure.child_ptr,
        structure.children,
        structure.prefix_ptr,
        len(structure.parent),
    )

    cardinality_mass = messages[structure.order[0], 0]

    require(
        abs(cardinality_mass.sum() - 1.0) < 1e-10,
        "Normalizacija punog amplitudnog stabla nije prosla.",
    )

    z = float(cardinality_mass[structure.k])

    require(
        z > 0.0 and math.isfinite(z),
        "Neispravna masa sektora sa sedam brojeva.",
    )

    log_z = math.log(z)

    mask, unnormalized_score = constrained_maximum(
        log_factors, structure, MAX_CONSECUTIVE
    )

    numbers = [
        j + 1
        for j in range(len(structure.parent))
        if (mask >> j) & 1
    ]

    verified = mask_log_probability(
        mask, log_factors, structure.parent, log_z
    )

    require(
        len(numbers) == structure.k,
        "NEXT nema odgovarajuci broj elemenata.",
    )
    require(
        not forbidden_run(np.int64(mask), MAX_CONSECUTIVE),
        "NEXT krsi ogranicenje uzastopnog niza.",
    )
    require(
        abs(verified - (unnormalized_score - log_z)) < 1e-8,
        "Skor dekodera nije saglasan sa amplitudnim modelom.",
    )
    require(
        verified <= 1e-9,
        "Verovatnoca NEXT kombinacije veca je od jedan.",
    )

    return tuple(numbers), (
        log_factors.copy(),
        structure.parent.copy(),
        log_z,
    )


# ------------------------------------------------------------
# INTERNE PROVERE
# ------------------------------------------------------------


def internal_checks():
    # Uslovni test: potpuna povezanost i uklanjanje povezanosti
    # nakon uslovljavanja na identicnu poznatu promenljivu.
    data = np.zeros((401, 3), dtype=np.uint8)
    data[:, 0] = np.arange(401) % 2
    data[1:, 1] = data[:-1, 0]
    data[:, 2] = data[:, 0]

    times = np.arange(1, 401, dtype=np.int64)

    p_unconditional = conditional_p(
        data, times, 1, (0, 1)
    )
    p_conditional = conditional_p(
        data, times, 1, (0, 1), ((2, 1),)
    )

    require(
        p_unconditional < 1e-20 and p_conditional == 1.0,
        "Provera uslovnog testa nije prosla.",
    )

    n = 6
    k = 2

    parent = np.array([-1, 0, 0, 1, 1, 2], dtype=np.int64)
    order = np.arange(n, dtype=np.int64)
    temporal = tuple(((j, 1),) for j in range(n))

    structure = assemble_structure(
        temporal, parent, order, k
    )

    context = np.array([1, 2, 0, 1, 2, 1], dtype=np.int64)
    parameters = 0.45 * np.sin(
        0.7 * np.arange(structure.parameter_count) + 0.2
    )

    targets = np.zeros((1, n), dtype=np.uint8)
    targets[0, [1, 4]] = 1

    batch = make_batch(context[None, :], targets, structure)

    _, analytic = batch_loss(
        parameters,
        batch.contexts,
        batch.counts,
        batch.weights,
        *structure.arguments(),
        True,
    )

    numerical = np.zeros_like(parameters)

    for index in range(len(parameters)):
        step = 1e-5
        plus = parameters.copy()
        minus = parameters.copy()
        plus[index] += step
        minus[index] -= step

        fp, _ = batch_loss(
            plus,
            batch.contexts,
            batch.counts,
            batch.weights,
            *structure.arguments(),
            False,
        )
        fm, _ = batch_loss(
            minus,
            batch.contexts,
            batch.counts,
            batch.weights,
            *structure.arguments(),
            False,
        )

        numerical[index] = (
            fp.sum() - fm.sum()
        ) / (2.0 * step * batch.size)

    require(
        np.max(np.abs(analytic - numerical)) < 3e-7,
        "Analiticki gradijent nije prosao proveru.",
    )

    log_factors, factors, _ = gate_factors(
        parameters, context, structure.parameter_ptr
    )

    messages, _ = forward_tree(
        factors,
        structure.order,
        structure.child_ptr,
        structure.children,
        structure.prefix_ptr,
        n,
    )

    cardinality_mass = messages[order[0], 0]

    require(
        abs(cardinality_mass.sum() - 1.0) < 1e-11,
        "Normalizacija stabla nije prosla proveru.",
    )

    circuit = QuantumCircuit(n)

    for node in order:
        base = (
            structure.parameter_ptr[node] + 2 * context[node]
        )
        p = parent[node]

        angle0 = math.pi / 2.0 + math.atan(parameters[base])

        if p < 0:
            circuit.ry(angle0, int(node))
        else:
            angle1 = (
                math.pi / 2.0 + math.atan(parameters[base + 1])
            )

            circuit.x(int(p))
            circuit.cry(angle0, int(p), int(node))
            circuit.x(int(p))
            circuit.cry(angle1, int(p), int(node))

    amplitudes = Statevector.from_instruction(circuit).data

    observed_cardinality = np.zeros(n + 1)

    for mask in range(1 << n):
        probability = math.exp(
            mask_log_probability(mask, log_factors, parent, 0.0)
        )

        require(
            abs(abs(amplitudes[mask]) ** 2 - probability) < 1e-11,
            "Qiskit amplitude nisu saglasne sa stablom.",
        )

        observed_cardinality[mask.bit_count()] += probability

    require(
        np.max(
            np.abs(observed_cardinality - cardinality_mass)
        ) < 1e-11,
        "Qiskit provera sektora fiksne kardinalnosti nije prosla.",
    )

    mask, score = constrained_maximum(log_factors, structure, 4)

    brute_score = -math.inf

    for combo in itertools.combinations(range(n), k):
        candidate_mask = sum(1 << j for j in combo)

        if forbidden_run(np.int64(candidate_mask), 4):
            continue

        candidate_score = mask_log_probability(
            candidate_mask, log_factors, parent, 0.0
        )
        brute_score = max(brute_score, candidate_score)

    require(
        abs(score - brute_score) < 1e-11,
        "Max-product dekoder nije prosao proveru.",
    )

    # Posebno aktiviranje branch-and-bound puta.
    fallback_structure = assemble_structure(
        tuple(() for _ in range(n)),
        parent,
        order,
        5,
    )

    fallback_parameters = np.zeros(
        fallback_structure.parameter_count
    )

    for node in range(n):
        start = fallback_structure.parameter_ptr[node]
        fallback_parameters[start:start + 2] = (
            2.0 if node < 5 else -2.0
        )

    fallback_factors, _, _ = gate_factors(
        fallback_parameters,
        np.zeros(n, dtype=np.int64),
        fallback_structure.parameter_ptr,
    )

    unfiltered_mask, _ = tree_maximum(
        fallback_factors,
        order,
        fallback_structure.child_ptr,
        fallback_structure.children,
        5,
    )

    require(
        forbidden_run(unfiltered_mask, 4),
        "Kontrolni primer ne aktivira ograniceni dekoder.",
    )

    fallback_mask, fallback_score = constrained_maximum(
        fallback_factors, fallback_structure, 4
    )

    brute_score = -math.inf

    for combo in itertools.combinations(range(n), 5):
        candidate_mask = sum(1 << j for j in combo)

        if not forbidden_run(np.int64(candidate_mask), 4):
            brute_score = max(
                brute_score,
                mask_log_probability(
                    candidate_mask, fallback_factors, parent, 0.0
                ),
            )

    require(
        abs(fallback_score - brute_score) < 1e-10
        and not forbidden_run(np.int64(fallback_mask), 4),
        "Branch-and-bound dekoder nije prosao proveru.",
    )

    # Hronologija konteksta.
    history = (
        np.arange(60).reshape(10, 6) // 7
    ).astype(np.uint8) % 2

    selected_times = np.arange(1, 6, dtype=np.int64)

    first = make_contexts(history, selected_times, temporal)
    changed = history.copy()
    changed[6:] = 1 - changed[6:]
    second = make_contexts(changed, selected_times, temporal)

    require(
        np.array_equal(first, second),
        "Kontekst koristi buduce podatke.",
    )

    # Grupisanje mora dati isti NLL kao pojedinacni uzorci.
    repeated_targets = np.repeat(targets, 3, axis=0)
    repeated_targets[1] = 0
    repeated_targets[1, [0, 5]] = 1

    repeated_contexts = np.repeat(context[None, :], 3, axis=0)

    grouped = make_batch(
        repeated_contexts, repeated_targets, structure
    )

    grouped_losses, grouped_gradient = batch_loss(
        parameters,
        grouped.contexts,
        grouped.counts,
        grouped.weights,
        *structure.arguments(),
        True,
    )

    individual_loss = 0.0
    individual_gradient = np.zeros_like(parameters)

    for row in range(3):
        single = make_batch(
            repeated_contexts[row:row + 1],
            repeated_targets[row:row + 1],
            structure,
        )

        losses, gradient = batch_loss(
            parameters,
            single.contexts,
            single.counts,
            single.weights,
            *structure.arguments(),
            True,
        )

        individual_loss += losses.sum()
        individual_gradient += gradient / 3.0

    require(
        abs(grouped_losses.sum() - individual_loss) < 1e-10
        and np.max(
            np.abs(grouped_gradient - individual_gradient)
        ) < 1e-10,
        "Tacno objedinjavanje uzoraka nije proslo proveru.",
    )


# ------------------------------------------------------------
# TRI NEZAVISNA CSV-A
# ------------------------------------------------------------


def prepare_stage(history, target_end):
    structure = learn_structure(
        history[:target_end + 1], K
    )

    all_times = np.arange(
        1, len(history) + 1, dtype=np.int64
    )

    contexts = make_contexts(
        history, all_times, structure.temporal
    )

    training = make_batch(
        contexts[:target_end],
        history[1:target_end + 1],
        structure,
    )

    return structure, contexts, training


def process_csv(path):
    draws = read_draws(path)
    history = binary_draws(draws, N)

    transition_count = len(history) - 1
    train_end = int(TRAIN_FRACTION * transition_count)
    validation_end = int(VALIDATION_END * transition_count)

    require(
        0 < train_end < validation_end < transition_count,
        "Neispravna hronoloska podela.",
    )

    structure, contexts, training = prepare_stage(
        history, train_end
    )

    print(
        f"Potvrdjene vremenske veze u treningu: "
        f"{structure.temporal_edge_count}."
    )

    validation = make_batch(
        contexts[train_end:validation_end],
        history[train_end + 1:validation_end + 1],
        structure,
    )

    best_model = None
    best_validation = math.inf
    warm = None

    for regularization in REGULARIZATIONS:
        print(
            f"Ucenje amplitudnog stabla: reg={regularization:g}.",
            flush=True,
        )

        try:
            candidate = train_model(
                structure, training, regularization, warm=warm
            )
            value = evaluate_model(candidate, validation)
            warm = candidate

        except (RuntimeError, ValueError, FloatingPointError) as exc:
            print(
                f"PROVERA: reg={regularization:g} odbacen: {exc}"
            )
            continue

        if value < best_validation:
            best_validation = value
            best_model = candidate

    if best_model is None:
        raise RuntimeError(
            "Nijedan validacioni kandidat nije prosao provere."
        )

    print(
        f"Validacija NLL: {best_validation:.6f} | "
        f"reg: {best_model.regularization:g}"
    )

    print("Ucenje modela za test.", flush=True)

    test_structure, test_contexts, test_training = prepare_stage(
        history, validation_end
    )

    test_model = train_model(
        test_structure,
        test_training,
        best_model.regularization,
        warm=best_model,
    )

    test_batch = make_batch(
        test_contexts[validation_end:transition_count],
        history[validation_end + 1:],
        test_structure,
    )

    print(f"Test NLL: {evaluate_model(test_model, test_batch):.6f}")

    print("Zavrsno ucenje: svi istorijski prelazi.", flush=True)

    final_structure, final_contexts, final_training = prepare_stage(
        history, transition_count
    )

    final_model = train_model(
        final_structure,
        final_training,
        best_model.regularization,
        warm=test_model,
    )

    print(
        f"Potvrdjene vremenske veze u zavrsnom modelu: "
        f"{final_structure.temporal_edge_count}."
    )

    next_numbers, distribution = predict_next(
        final_model, final_contexts[transition_count]
    )

    print(f"NEXT: {list(next_numbers)}")

    fingerprint = hashlib.sha256(
        np.ascontiguousarray(draws).tobytes()
    ).hexdigest()

    return {
        "name": path.name,
        "next": next_numbers,
        "fingerprint": fingerprint,
        "distribution": distribution,
    }


def diagnostic_difference(first, second):
    largest = 0.0

    for step in (1, 2, 4, 5, 7, 8, 10, 11, 14, 16, 17, 19):
        for start in range(N):
            indices = (start + step * np.arange(K)) % N
            mask = sum(1 << int(j) for j in indices)

            a = mask_log_probability(mask, *first)
            b = mask_log_probability(mask, *second)
            largest = max(largest, abs(a - b))

    return largest


def main():
    print(
        "Quantum Temporal Graph V1: uslovne vremenske veze "
        "i kontrolisane RY amplitude."
    )
    print(
        "Tacan NLL cele sedmorke; "
        "normalizacija i maksimum preko stabla."
    )
    print(
        "Lokalna simulacija; otkrivanje grafa "
        "i optimizacija parametara su klasicni."
    )
    print("NEXT: najvise cetiri uzastopna broja.")
    print("Pokretanje internih provera.", flush=True)

    internal_checks()

    print("Interne provere: prosle.")

    results = []
    failures = 0

    for path in FILES:
        print()
        print(f"CSV: {path.name}", flush=True)

        try:
            current = process_csv(path)

            for previous in results:
                if current["next"] != previous["next"]:
                    continue

                same_input = (
                    current["fingerprint"] == previous["fingerprint"]
                )

                difference = diagnostic_difference(
                    current["distribution"],
                    previous["distribution"],
                )

                print(
                    f"PROVERA: isti NEXT za "
                    f"{previous['name']} i {current['name']}."
                )

                input_status = "isti" if same_input else "razliciti"

                print(
                    f"CSV sadrzaji: {input_status}; "
                    "maksimalna razlika logP na provernim kombinacijama: "
                    f"{difference:.6g}."
                )

            results.append(current)

        except KeyboardInterrupt:
            raise

        except Exception as exc:
            failures += 1
            print(f"STOP za ovaj CSV: {exc}")

            if DIAGNOSTICS:
                import traceback
                traceback.print_exc()

    if failures:
        print()
        print(
            "Obrada nije kompletna: neki CSV-ovi nisu prosli provere."
        )


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print()
        print("Obrada prekinuta korisnickim zahtevom.")

    except Exception as exc:
        print()
        print(f"STOP: {exc}")

        if DIAGNOSTICS:
            import traceback
            traceback.print_exc()



"""
Quantum Temporal Graph V1: uslovne vremenske veze i kontrolisane RY amplitude.
Tacan NLL cele sedmorke; normalizacija i maksimum preko stabla.
Lokalna simulacija; otkrivanje grafa i optimizacija parametara su klasicni.
NEXT: najvise cetiri uzastopna broja.
Pokretanje internih provera.
Interne provere: prosle.

CSV: loto7_4720_k91.csv
Potvrdjene vremenske veze u treningu: 0.
Ucenje amplitudnog stabla: reg=10.
Ucenje amplitudnog stabla: reg=1.
Ucenje amplitudnog stabla: reg=0.1.
Validacija NLL: 16.546221 | reg: 10
Ucenje modela za test.
Test NLL: 16.565149
Zavrsno ucenje: svi istorijski prelazi.
Potvrdjene vremenske veze u zavrsnom modelu: 0.
NEXT: [8, 11, 22, 23, 33, 34, 38]

CSV: loto7_4720_k91_loto_2982.csv
Potvrdjene vremenske veze u treningu: 0.
Ucenje amplitudnog stabla: reg=10.
Ucenje amplitudnog stabla: reg=1.
Ucenje amplitudnog stabla: reg=0.1.
Validacija NLL: 16.571115 | reg: 10
Ucenje modela za test.
Test NLL: 16.566918
Zavrsno ucenje: svi istorijski prelazi.
Potvrdjene vremenske veze u zavrsnom modelu: 0.
NEXT: [7, 8, 9, 23, 24, 26, 34]

CSV: loto7_4720_k91_loto_plus_1738.csv
Potvrdjene vremenske veze u treningu: 0.
Ucenje amplitudnog stabla: reg=10.
Ucenje amplitudnog stabla: reg=1.
Ucenje amplitudnog stabla: reg=0.1.
Validacija NLL: 16.572382 | reg: 10
Ucenje modela za test.
Test NLL: 16.564538
Zavrsno ucenje: svi istorijski prelazi.
Potvrdjene vremenske veze u zavrsnom modelu: 0.
NEXT: [2, 4, 11, 12, 18, 23, 27]
"""



"""
Učenje vremenskih zavisnosti koje ostaju značajne kada se uzmu u obzir ostale informacije iz istorije.

Izbor ulaza, otkrivanje strukture zavisnosti, skrivene zajedničke faktore, vremenska kašnjenja i promene režima. Velika prediktivna važnost neke karakteristike sama ne dokazuje uzročnost. 

PCMCI — uslovne zavisnosti kroz vreme. Proveravam da li prošla informacija doprinosi objašnjenju naredne vrednosti i nakon uključivanja drugih relevantnih prošlih informacija. Podržava različite testove za diskretne i kontinuirane podatke. 


- Učim koje veze između prethodnih kola i narednog izvlačenja donose dodatnu informaciju.
- Proveravam njihovu stabilnost kroz uzastopne istorijske periode, uz kontrolu višestrukih testiranja.
- Naučenu strukturu koristim za izgradnju zajedničke raspodele sedmorke, iz koje biram NEXT.

Tačno sedam izvučenih brojeva već stvara zavisnosti unutar kola. Te veze se ne sme automatski tumačiti kao prediktivni signal.

PCMCI otkriva strukturu; nije sam po sebi dekoder NEXT kombinacije. Kvantni amplitudni model uslovljen tom strukturom bio bi moja nadogradnja, a korist proveravam hronološkom validacijom.


Izdvajam uslovne veze kroz prethodna kola, proveravam njihovu stabilnost i koristim ih u amplitudnom modelu cele sedmorke.


Uslovno kvantno stablo: vremenske veze određuju kontekst rotacija, a stablo predstavlja zajedničke amplitude sedmorke. Normalizacija i izbor maksimuma računaju se po stablu. Ako podaci ne potvrde nijednu vremensku vezu, kod će to jasno ispisati, bez dodavanja izmišljenih veza.


Definisao sam i zaštitu od curenja podataka: izbor veza i provera njihove stabilnosti koriste odvojene hronološke delove treninga. Validacija i test ostaju van tog postupka. Pri računanju modela mogu se tačno objediniti uzorci sa istim kontekstom, što smanjuje posao bez izbacivanja istorijskih prelaza.


Uslovne vremenske veze određuju kontekst kvantnih rotacija. Zajednička raspodela i njen maksimum računaju se preko stabla. Ovo je moja adaptacija principa PC/MCI, bez tvrdnje da otkrivene veze dokazuju uzročnost.
"""
