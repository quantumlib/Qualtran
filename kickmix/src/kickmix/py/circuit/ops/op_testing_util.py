from __future__ import annotations

from typing import Any

import kickmix as km

U64_FALSE = 0b0000000000000000000000000000000000000000000000000000000000000000
U64_TRUE = 0b1111111111111111111111111111111111111111111111111111111111111111
U64_BIT5 = 0b1111111111111111111111111111111100000000000000000000000000000000
U64_BIT4 = 0b1111111111111111000000000000000011111111111111110000000000000000
U64_BIT3 = 0b1111111100000000111111110000000011111111000000001111111100000000
U64_BIT2 = 0b1111000011110000111100001111000011110000111100001111000011110000
U64_BIT1 = 0b1100110011001100110011001100110011001100110011001100110011001100
U64_BIT0 = 0b1010101010101010101010101010101010101010101010101010101010101010


def store64(sim: km.Simulator, index: km.q | km.b | km.xb | km.xbool | bool, val64: int) -> int:
    if isinstance(index, bool):
        return U64_TRUE if index else U64_FALSE
    if isinstance(index, km.xb):
        sim.write_across_shots(km.b(index.id), val64)
        return val64
    if isinstance(index, km.xbool):
        return U64_TRUE if index == km.xbool(True) else U64_FALSE
    sim.write_across_shots(index, val64)
    return val64


def read64(sim: km.Simulator, index: km.q | km.b | km.xb | km.xbool | bool) -> int:
    if isinstance(index, bool):
        return U64_TRUE if index else U64_FALSE
    if isinstance(index, km.xb):
        return sim.read_across_shots(km.b(index.id), out=int)
    if isinstance(index, km.xbool):
        return U64_TRUE if index == km.xbool(True) else U64_FALSE
    return sim.read_across_shots(index, out=int)


def x_cases_q(index: int) -> list[Any]:
    return [km.q(index)]


def x_cases_bc(index: int) -> list[Any]:
    return [km.xb(index), km.xbool(False), km.xbool(True)]


def z_cases(index: int) -> list[Any]:
    return [km.q(index), km.b(index), False, True]


def x_list_cases(index: int, size: int, *, include_x_bits: bool = True) -> list[Any]:
    r = range(index, index + size)
    result: list[Any] = []
    result.append([km.q(k) for k in r])
    result.append(km.array(result[-1]))
    if include_x_bits:
        result.append(km.array([km.xb(k) for k in r]))
        result.append(km.array([km.xbool(hash((k, index, size)) & 1 == 0) for k in r]))
    mixed = []
    for k in r:
        h = hash((k, index, size)) % 4
        if h == 0 and include_x_bits:
            mixed.append(km.xbool(False))
        elif h == 1 and include_x_bits:
            mixed.append(km.xbool(True))
        elif h == 2 and include_x_bits:
            mixed.append(km.xb(k))
        else:
            mixed.append(km.q(k))
    result.append(km.array(mixed))
    result.extend(x_cases_q(index))
    if include_x_bits:
        result.extend(x_cases_bc(index))
    return result


def z_list_cases(index: int, size: int) -> list[Any]:
    r = range(index, index + size)
    result: list[Any] = []
    result.append([km.q(k) for k in r])
    result.append(km.array(result[-1]))
    result.append(km.array([km.b(k) for k in r]))
    result.append(km.array([hash((k, index, size)) & 1 == 0 for k in r]))
    mixed = []
    for k in r:
        h = hash((k, index, size)) % 4
        if h == 0:
            mixed.append(False)
        elif h == 1:
            mixed.append(True)
        elif h == 2:
            mixed.append(km.q(k))
        else:
            mixed.append(km.b(k))
    result.append(km.array(mixed))
    result.extend(z_cases(index))
    return result


def broadcast_logic(*xs: Any) -> list[Any]:
    for x in xs:
        if hasattr(x, '__len__'):
            n = len(x)
            break
    else:
        n = 1

    result = [[x] * n if isinstance(x, (km.b, km.xb, km.q, km.xbool, bool)) else x for x in xs]
    for x in result:
        assert len(x) == n
    return result
