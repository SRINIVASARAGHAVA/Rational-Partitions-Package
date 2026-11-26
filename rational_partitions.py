#!/usr/bin/env python3
"""Exact values of the diagonal rational partition numbers R(n) and of the key counts K(n).

Definition
    The keys of an integer N >= 1 are the fractions a/b in lowest terms with
    1 <= a <= N and 2 <= b <= N.  They form the set C_N, and K(N) is their number.
    A rational partition of m of order N is a multiset of keys of N whose sum is m.
    R_N(m) is the number of them, and R(n) = R_n(n).

    Examples:  1 has no keys, so K(1) = 0 and R(1) = 0.
               The only key of 2 is 1/2, so K(2) = 1 and R(2) = 1.
               The keys of 3 are 1/3, 1/2, 2/3, 3/2, so K(3) = 4, and R(3) = 16.
               The keys of 4 are 1/4, 1/3, 1/2, 2/3, 3/4, 4/3, 3/2, so K(4) = 7, and R(4) = 226.

    K(N) = 2 (phi(1) + ... + phi(N)) - N - 1, where phi is Euler's function.

Method
    R_N(m) is the coefficient of z^m in the product of 1/(1 - z^(a/b)) over C_N.
    The denominators are grouped by their largest prime factor p, and the primes are
    taken in decreasing order.  The series is kept on a lattice (1/D)Z.  Once all
    denominators with largest prime factor p are multiplied in, no later denominator is
    divisible by p.  So an exponent that is not p-integral can never become an integer,
    and its term is dropped.  This keeps D small (D <= 27720 for N <= 101).

Usage
    python3 rational_partitions.py 101             K(n) and R(n) for 1 <= n <= 101, written to R_values.txt
    python3 rational_partitions.py 101 --jobs 4    the same on four processes
    python3 rational_partitions.py --keys 4        the keys of 4
    python3 rational_partitions.py --row 12 20     R_12(0), ..., R_12(20)
    python3 rational_partitions.py --selftest      comparison with a direct computation

Only the Python standard library is used.  All arithmetic is exact.

Reference
    K. Srinivasa Raghava, Asymptotics and limit laws for diagonal rational partitions.
"""
import argparse
import multiprocessing
import sys
import time
from fractions import Fraction
from functools import reduce
from math import gcd
from operator import add


def lcm(a, b):
    return a * b // gcd(a, b)


def primes_upto(n):
    """All primes p <= n, in increasing order."""
    sieve = [True] * (n + 1)
    primes = []
    for p in range(2, n + 1):
        if sieve[p]:
            primes.append(p)
            for k in range(p * p, n + 1, p):
                sieve[k] = False
    return primes


def largest_prime_factor(n):
    best, d = 1, 2
    while d * d <= n:
        while n % d == 0:
            best, n = d, n // d
        d += 1
    return n if n > 1 else best


def keys(N):
    """The keys of N in increasing order: the fractions a/b in lowest terms, 1 <= a <= N, 2 <= b <= N."""
    return sorted(Fraction(a, b) for b in range(2, N + 1) for a in range(1, N + 1) if gcd(a, b) == 1)


def key_count(N):
    """K(N), the number of keys of N."""
    return sum(1 for b in range(2, N + 1) for a in range(1, N + 1) if gcd(a, b) == 1)


def divide_by_one_minus_power(g, step):
    """Replace the coefficient list g of a series G(x) by that of G(x) / (1 - x^step)."""
    size, lo = len(g), step
    while lo < size:
        hi = min(lo + step, size)
        g[lo:hi] = map(add, g[lo:hi], g[lo - step:hi - step])
        lo = hi


def counts(N, M):
    """Return the list [R_N(0), R_N(1), ..., R_N(M)]."""
    f, D = [1] + [0] * M, 1                  # f[i] is the coefficient of z^(i/D)
    for p in reversed(primes_upto(N)):
        block = [b for b in range(2, N + 1) if largest_prime_factor(b) == p]
        D2 = reduce(lcm, block, D)
        g = [0] * (D2 * M + 1)               # the same series on the finer lattice (1/D2)Z
        g[::D2 // D] = f
        for b in block:
            for a in range(1, N + 1):
                if gcd(a, b) == 1 and a * (D2 // b) < len(g):
                    divide_by_one_minus_power(g, a * (D2 // b))
        q = 1                                # the power of p in D2
        while D2 % (q * p) == 0:
            q *= p
        f, D = g[::q], D2 // q               # keep the p-integral exponents
    assert D == 1
    return f


def diagonal(n):
    """R(n) = R_n(n)."""
    return counts(n, n)[n]


def counts_direct(N, M):
    """[R_N(0), ..., R_N(M)] computed on the full lattice (1/lcm(1..N))Z.  For small N only."""
    L = reduce(lcm, range(1, N + 1), 1)
    g = [1] + [0] * (L * M)
    for b in range(2, N + 1):
        for a in range(1, N + 1):
            if gcd(a, b) == 1 and a * (L // b) < len(g):
                divide_by_one_minus_power(g, a * (L // b))
    return g[::L]


def _task(n):
    start = time.time()
    return n, diagonal(n), time.time() - start


def diagonal_values(nmax, jobs=1, report=None):
    """Return {n: R(n)} for 1 <= n <= nmax."""
    order = range(nmax, 0, -1)               # the expensive values first
    values = {}
    if jobs > 1:
        with multiprocessing.Pool(jobs) as pool:
            for n, value, seconds in pool.imap_unordered(_task, order):
                values[n] = value
                if report:
                    report(n, value, seconds)
    else:
        for n in order:
            n, value, seconds = _task(n)
            values[n] = value
            if report:
                report(n, value, seconds)
    return values


def write_values(values, path):
    nmax = max(values)
    with open(path, "w") as out:
        out.write("# Key counts K(n) and diagonal rational partition numbers R(n) for 1 <= n <= %d.\n" % nmax)
        out.write("# The keys of n are the fractions a/b in lowest terms with 1 <= a <= n and 2 <= b <= n.\n")
        out.write("# K(n) is the number of keys of n.  R(n) is the number of multisets of keys of n whose sum is n.\n")
        out.write("# Format: n K(n) R(n)\n")
        for n in range(1, nmax + 1):
            out.write("%d %d %d\n" % (n, key_count(n), values[n]))


def selftest(nmax=12):
    known = {1: 0, 2: 1, 3: 16, 4: 226, 5: 8047, 6: 223310, 7: 19427673, 8: 1208928320}
    assert counts(1, 3) == counts_direct(1, 3) == [1, 0, 0, 0]
    for n in range(1, nmax + 1):
        fast = counts(n, n)
        assert fast == counts_direct(n, n), n
        assert n not in known or fast[n] == known[n], n
        print("N = %2d: R_N(m) for 0 <= m <= N agrees with the direct computation; K(%d) = %d, R(%d) = %d"
              % (n, n, key_count(n), n, fast[n]))
    assert [key_count(n) for n in range(1, 5)] == [0, 1, 4, 7]
    assert [str(k) for k in keys(4)] == ["1/4", "1/3", "1/2", "2/3", "3/4", "4/3", "3/2"]
    totient_sum = 0
    for n in range(1, 201):                  # K(n) = 2 (phi(1) + ... + phi(n)) - n - 1
        totient_sum += sum(1 for k in range(1, n + 1) if gcd(k, n) == 1)
        assert key_count(n) == len(keys(n)) == 2 * totient_sum - n - 1, n
    print("K(n) = 2 (phi(1) + ... + phi(n)) - n - 1 for n <= 200")
    print("self-test passed")


def main():
    parser = argparse.ArgumentParser(description="Exact diagonal rational partition numbers R(n) and key counts K(n).")
    parser.add_argument("nmax", nargs="?", type=int, help="compute K(n) and R(n) for 1 <= n <= nmax")
    parser.add_argument("--out", default="R_values.txt", help="output file (default: R_values.txt)")
    parser.add_argument("--jobs", type=int, default=1, help="number of processes (default: 1)")
    parser.add_argument("--keys", type=int, metavar="N", help="print the keys of N")
    parser.add_argument("--row", nargs=2, type=int, metavar=("N", "M"), help="print R_N(0), ..., R_N(M)")
    parser.add_argument("--selftest", action="store_true", help="compare with a direct computation for N <= 12")
    args = parser.parse_args()
    if args.selftest:
        selftest()
    elif args.keys:
        print("K(%d) = %d" % (args.keys, key_count(args.keys)))
        print(" ".join(str(k) for k in keys(args.keys)))
    elif args.row:
        for m, value in enumerate(counts(*args.row)):
            print(m, value)
    elif args.nmax and args.nmax >= 1:
        def report(n, value, seconds):
            print("R(%d) has %d digits  [%.1f s]" % (n, len(str(value)), seconds), file=sys.stderr, flush=True)
        values = diagonal_values(args.nmax, args.jobs, report)
        write_values(values, args.out)
        print("wrote %s" % args.out)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
