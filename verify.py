#!/usr/bin/env python3
"""Independent checks of the values in R_values.txt.

The file has one line "n K(n) R(n)" for each n.  Every number is compared with results
obtained in other ways.  None of the checks uses the code of rational_partitions.py.

Check 1  (key counts, every n in the file)
    K(n) is recomputed in two ways: as the number of distinct non-integral values of a/b
    with 1 <= a, b <= n, and from K(n) = 2 (phi(1) + ... + phi(n)) - n - 1.

Check 2  (exact, n <= 8)
    R(n) is counted from the definition with exact fractions: a recursion over the sorted
    list of keys, with no lattice and no generating function.

Check 3  (modulo 2^64, every n in the file)
    The product of 1/(1 - z^(a/b)) over the keys of n is recomputed with numpy arrays of
    unsigned 64-bit integers, whose arithmetic is arithmetic modulo 2^64.  This is a second
    implementation of the elimination of primes.

Check 4  (modulo 2^64, every n in the file)
    A different formula.  Let P_K(z) be the product of 1/(1 - z^k) for k <= K.  By Moebius
    inversion over the squarefree d,
        prod_{a/b in C_N} 1/(1 - z^(a/b))
            = (1/P_N(z)) * prod_{d <= N} prod_{j <= N/d} P_{floor(N/d)}(z^(1/j))^mu(d).
    Here the fractions are not reduced and the coprimality condition is replaced by the
    Moebius function, so the set C_N of keys itself is tested.

Usage
    python3 verify.py                  all checks for the whole file
    python3 verify.py --max 60         the same for n <= 60
    python3 verify.py --jobs 2         checks 3 and 4 on two processes

Requires numpy.
"""
import argparse
import multiprocessing
import sys
import time
from fractions import Fraction
from functools import lru_cache, reduce
from math import gcd

import numpy as np

MODULUS = 2 ** 64                         # numpy uint64 arithmetic wraps around modulo 2^64


def read_values(path):
    """Return ({n: K(n)}, {n: R(n)})."""
    key_counts, values = {}, {}
    for line in open(path):
        if line.strip() and not line.startswith("#"):
            n, keys, value = line.split()
            key_counts[int(n)], values[int(n)] = int(keys), int(value)
    return key_counts, values


# ---------------------------------------------------------------- check 1

def key_count_by_values(N):
    """Number of distinct non-integral values a/b with 1 <= a, b <= N."""
    return len({x for x in (Fraction(a, b) for a in range(1, N + 1) for b in range(1, N + 1)) if x.denominator > 1})


def totients(nmax):
    """[phi(0), phi(1), ..., phi(nmax)] by a sieve."""
    phi = list(range(nmax + 1))
    for p in range(2, nmax + 1):
        if phi[p] == p:                       # p is prime
            for k in range(p, nmax + 1, p):
                phi[k] -= phi[k] // p
    return phi


# ---------------------------------------------------------------- check 2

def count_from_definition(N):
    """R(N), counted with exact fractions."""
    parts = sorted({x for x in (Fraction(a, b) for a in range(1, N + 1) for b in range(1, N + 1)) if x.denominator > 1},
                   reverse=True)

    @lru_cache(maxsize=None)
    def count(i, rest):
        """Number of multisets from parts[i:] with sum rest."""
        if rest == 0:
            return 1
        if i == len(parts):
            return 0
        total = 0
        while rest >= 0:
            total += count(i + 1, rest)
            rest -= parts[i]
        return total

    return count(0, Fraction(N))


# ---------------------------------------------------------------- residues

def lcm(a, b):
    return a * b // gcd(a, b)


def prime_list(n):
    return [p for p in range(2, n + 1) if all(p % d for d in range(2, int(p ** 0.5) + 1))]


def top_prime(n):
    """Largest prime factor of n >= 2."""
    return max(p for p in prime_list(n) if n % p == 0)


def moebius(n):
    result = 1
    for p in prime_list(n):
        if n % (p * p) == 0:
            return 0
        if n % p == 0:
            result = -result
    return result


def geometric(g, step):
    """g <- g / (1 - x^step) modulo 2^64, for a uint64 array g of coefficients."""
    n = len(g)
    if step >= n:
        return
    full = (n // step) * step
    rows = g[:full].reshape(-1, step)         # row i holds the coefficients i*step, ..., i*step + step - 1
    rows[:] = np.cumsum(rows, axis=0, dtype=np.uint64)     # running sums down each residue class
    if full < n:
        g[full:] += rows[-1, :n - full]


def one_minus(g, step):
    """g <- g * (1 - x^step) modulo 2^64."""
    if step < len(g):
        g[step:] -= g[:-step].copy()


def eliminate(N, M, factors_of):
    """The integral part, up to z^M and modulo 2^64, of a product over denominators 2 <= c <= N.

    factors_of(c) lists pairs (k, sign): the factor (1 - z^(k/c))^(-sign) is multiplied in.
    The denominators are grouped by their largest prime factor."""
    f, D = np.zeros(M + 1, dtype=np.uint64), 1
    f[0] = 1
    for p in reversed(prime_list(N)):
        block = [c for c in range(2, N + 1) if top_prime(c) == p]
        D2 = reduce(lcm, block, D)
        g = np.zeros(D2 * M + 1, dtype=np.uint64)
        g[::D2 // D] = f
        for c in block:
            for k, sign in factors_of(c):
                (geometric if sign > 0 else one_minus)(g, k * (D2 // c))
        q = 1
        while D2 % (q * p) == 0:
            q *= p
        f, D = np.ascontiguousarray(g[::q]), D2 // q
    assert D == 1
    return f


def residue_product(N):
    """R(N) modulo 2^64, from the product over the keys of N."""
    def factors_of(b):
        return [(a, 1) for a in range(1, N + 1) if gcd(a, b) == 1]
    return int(eliminate(N, N, factors_of)[N])


def residue_boxes(N):
    """R(N) modulo 2^64, from the Moebius formula."""
    squarefree = [(d, moebius(d)) for d in range(1, N + 1) if moebius(d) != 0]

    def factors_of(j):
        return [(k, mu) for d, mu in squarefree if d * j <= N for k in range(1, N // d + 1)]
    f = eliminate(N, N, factors_of)
    for d, mu in squarefree:                      # the factors with j = 1; d = 1 cancels 1/P_N
        if d > 1:
            for k in range(1, N // d + 1):
                (geometric if mu > 0 else one_minus)(f, k)
    return int(f[N])


def _task(n):
    start = time.time()
    return n, residue_product(n), residue_boxes(n), time.time() - start


def main():
    parser = argparse.ArgumentParser(description="Independent checks of R_values.txt.")
    parser.add_argument("--file", default="R_values.txt")
    parser.add_argument("--max", type=int, default=None, help="check only n <= MAX")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    key_counts, values = read_values(args.file)
    nmax = min(max(values), args.max or max(values))
    assert sorted(values) == list(range(1, max(values) + 1))

    phi, totient_sum = totients(nmax), 0
    for n in range(1, nmax + 1):
        totient_sum += phi[n]
        assert key_counts[n] == key_count_by_values(n) == 2 * totient_sum - n - 1, ("check 1 fails", n)
    print("check 1 passed: K(n) for n <= %d" % nmax, flush=True)

    for n in range(1, min(nmax, 8) + 1):
        assert count_from_definition(n) == values[n], n
    print("check 2 passed: R(n) from the definition for n <= %d" % min(nmax, 8), flush=True)

    def compare(results):
        for n, product, boxes, seconds in results:
            expected = values[n] % MODULUS
            assert product == expected, ("check 3 fails", n)
            assert boxes == expected, ("check 4 fails", n)
            print("n = %3d: checks 3 and 4 agree with the file  [%.1f s]" % (n, seconds), file=sys.stderr, flush=True)

    order = range(nmax, 0, -1)                    # the expensive values first
    if args.jobs > 1:
        with multiprocessing.Pool(args.jobs) as pool:
            compare(pool.imap_unordered(_task, order))
    else:
        compare(map(_task, order))
    print("check 3 passed: product over the keys, modulo 2^64, for n <= %d" % nmax)
    print("check 4 passed: Moebius formula, modulo 2^64, for n <= %d" % nmax)


if __name__ == "__main__":
    main()
