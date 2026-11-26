#!/usr/bin/env python3
"""Asymptotic formulas for the rational partition numbers R(n), compared with the exact values.

Notation
    The keys of n are the fractions a/b in lowest terms with 1 <= a <= n and 2 <= b <= n.
    For beta > 0 let

        F_n(beta) = - sum over the keys x of log(1 - exp(-beta x)).

    For a prime l <= n let q be the largest power of l not exceeding n, let k = floor(n/q) and
    L_k = lcm(1, ..., k).  The block of l is the set of keys whose denominator is divisible by q.
    For x in the block, w(x) = q L_k x is an integer.  Put

        P_l(beta) = (1/l) * sum_{r=0}^{l-1} prod_{x in the block of l}
                            (1 - exp(-beta x)) / (1 - exp(2 pi i r w(x)/l - beta x)).

    Give every key x an independent multiplicity that is geometric with ratio exp(-beta x).
    Then P_l(beta) is the probability that the sum of the parts taken from the block of l has
    a denominator that is not divisible by q.  This is necessary for the total to be an integer.
    Finally

        Psi_n(beta) = beta n + F_n(beta) + sum over the primes l <= n of log P_l(beta),

    beta* is the minimiser of Psi_n, sigma^2 = Psi_n''(beta*), and D_n = lcm(1..n) / (product of
    the primes l <= n).

Formulas
    (1) Theorem (proved):    log R(n) = T(n) + O(n^(3/2) / log^3 n), where
            T(n) = sqrt(2) n^(3/2) - kappa n^(3/2)/log n - kappa2 n^(3/2)/log^2 n,
            kappa = 0.0026471306...,  kappa2 = 0.0010771199...

    (2) Bound (proved):      log R(n) <= Psi_n(beta) for every beta > 0, and
                             log R(n) >= Psi_n(beta*) - C n.

    (3) Sharp formula without the second-order term (a conjecture):
            R(n) ~ exp(Psi_n(beta*)) / (D_n sqrt(2 pi sigma^2)).

    (4) Sharp formula, with the next term of the local limit expansion (numerical evidence):
            R(n) ~ exp(Psi_n(beta*)) / (D_n sqrt(2 pi sigma^2)) * (1 + c2),
            c2 = Psi''''/(8 sigma^4) - 5 Psi'''^2/(24 sigma^6)      (derivatives at beta*).

    Conjecture 8.1 of the paper is formula (3) with the sum over l restricted to the primes
    above sqrt(n) (then D_n is replaced by the product of the largest prime powers q <= n of
    the primes l <= sqrt(n)).  It is evaluated as well.

Usage
    python3 asymptotics.py              comparison with R_values.txt for all n, and a table
    python3 asymptotics.py 100          all formulas for one value of n
    python3 asymptotics.py --selftest   checks of the program

Requires numpy.

Reference
    K. Srinivasa Raghava, Asymptotics and limit laws for diagonal rational partitions
    (Theorem 1.1, Proposition 4.1 and Conjecture 8.1).
"""
import os
import sys
from functools import reduce
from math import exp, gcd, log, pi, sqrt

import numpy as np

KAPPA = 0.0026471306
KAPPA2 = 0.0010771199


def lcm(a, b):
    return a * b // gcd(a, b)


def primes_upto(n):
    return [p for p in range(2, n + 1) if all(p % d for d in range(2, int(p ** 0.5) + 1))]


def theorem_value(n):
    """T(n): the first three terms of log R(n)."""
    return sqrt(2) * n ** 1.5 - KAPPA * n ** 1.5 / log(n) - KAPPA2 * n ** 1.5 / log(n) ** 2


def minimise(function, start):
    """Minimiser of a strictly convex function on (0, infinity); function returns (value, f', f'').
    Newton's method, kept inside an interval on which f' changes sign."""
    lo = hi = start
    while function(lo)[1] > 0:
        lo /= 2
    while function(hi)[1] < 0:
        hi *= 2
    beta = 0.5 * (lo + hi)
    for _ in range(200):
        value, d1, d2 = function(beta)
        lo, hi = (beta, hi) if d1 < 0 else (lo, beta)
        new = beta - d1 / d2
        if not lo < new < hi:
            new = 0.5 * (lo + hi)
        if abs(new - beta) < 1e-14 * beta:
            break
        beta = new
    return beta


def block_of(n, l, pairs):
    """Keys x of the block of the prime l, and the residues of w(x) = q L_k x modulo l."""
    q = l
    while q * l <= n:
        q *= l
    k = n // q
    L = reduce(lcm, range(1, k + 1), 1)
    block = [(a, b) for a, b in pairs if b % q == 0]
    x = np.array([a / b for a, b in block])
    w = np.array([(a * L * q // b) % l for a, b in block])
    return q, x, w


class Model:
    """The function Psi_n.  With all_primes=False only the primes above sqrt(n) are used,
    as in Conjecture 8.1 of the paper."""

    def __init__(self, n, all_primes=True):
        self.n = n
        pairs = [(a, b) for b in range(2, n + 1) for a in range(1, n + 1) if gcd(a, b) == 1]
        self.keys = np.array([a / b for a, b in pairs])
        self.blocks = []
        self.log_lattice = 0.0                     # log D_n, or log Q_n for the paper's version
        for l in primes_upto(n):
            q, x, w = block_of(n, l, pairs)
            if all_primes or l * l > n:
                phases = np.exp(2j * pi * ((np.arange(l)[:, None] * w[None, :]) % l) / l)
                self.blocks.append((l, x, phases))
                self.log_lattice += log(q // l)
            else:
                self.log_lattice += log(q)

    def free(self, beta):
        """beta n + F_n(beta) and its first two derivatives."""
        q = np.exp(-beta * self.keys)
        value = beta * self.n - np.log1p(-q).sum()
        d1 = self.n - (self.keys * q / (1 - q)).sum()
        d2 = (self.keys ** 2 * q / (1 - q) ** 2).sum()
        return value, d1, d2

    @staticmethod
    def log_probability(beta, l, x, phases):
        """log P_l(beta) and its first two derivatives."""
        q = np.exp(-beta * x)
        z = phases * q                                                    # shape (l, size of the block)
        terms = np.exp(np.log1p(-q).sum() - np.log(1 - z).sum(axis=1))
        a = (x * (q / (1 - q) - z / (1 - z))).sum(axis=1)
        b = (x ** 2 * (z / (1 - z) ** 2 - q / (1 - q) ** 2)).sum(axis=1)
        P = terms.sum().real / l
        P1 = (terms * a).sum().real / l
        P2 = (terms * (a * a + b)).sum().real / l
        return log(P), P1 / P, P2 / P - (P1 / P) ** 2

    def psi(self, beta):
        """Psi_n(beta), Psi_n'(beta), Psi_n''(beta)."""
        value, d1, d2 = self.free(beta)
        for l, x, phases in self.blocks:
            v, e1, e2 = self.log_probability(beta, l, x, phases)
            value, d1, d2 = value + v, d1 + e1, d2 + e2
        return value, d1, d2

    def evaluate(self):
        """beta*, Psi_n(beta*), sigma^2, the logarithm of formula (3), and the correction c2 of formula (4)."""
        beta = minimise(self.psi, sqrt(self.n / 2))
        value, d1, sigma2 = self.psi(beta)
        h = 2e-3 * beta                  # third and fourth derivatives by five-point differences of the second
        p2, p1, m1, m2 = (self.psi(beta + j * h)[2] for j in (2, 1, -1, -2))
        d3 = (-p2 + 8 * p1 - 8 * m1 + m2) / (12 * h)
        d4 = (-p2 + 16 * p1 - 30 * sigma2 + 16 * m1 - m2) / (12 * h ** 2)
        c2 = d4 / (8 * sigma2 ** 2) - 5 * d3 ** 2 / (24 * sigma2 ** 3)
        return beta, value, sigma2, value - self.log_lattice - 0.5 * log(2 * pi * sigma2), c2


def read_exact(path="R_values.txt"):
    """{n: log R(n)} for the n >= 2 of the file."""
    exact = {}
    if os.path.exists(path):
        for line in open(path):
            if line.strip() and not line.startswith("#"):
                n, keys, value = line.split()
                if int(value) > 0:
                    exact[int(n)] = log(int(value))
    return exact


def formulas(n):
    beta, psi, sigma2, sharp, c2 = Model(n).evaluate()
    paper = Model(n, all_primes=False).evaluate()[3]
    return dict(n=n, leading=sqrt(2) * n ** 1.5, theorem=theorem_value(n), beta=beta, psi=psi, sigma2=sigma2,
                paper=paper, sharp=sharp, c2=c2)


def selftest():
    # P_l by the formula with roots of unity, against a convolution over the residues modulo l
    for n, beta in ((12, 2.3), (30, 3.9), (50, 5.0), (64, 5.7), (100, 7.06)):
        pairs = [(a, b) for b in range(2, n + 1) for a in range(1, n + 1) if gcd(a, b) == 1]
        for l in primes_upto(n):
            q, x, w = block_of(n, l, pairs)
            phases = np.exp(2j * pi * ((np.arange(l)[:, None] * w[None, :]) % l) / l)
            value = exp(Model.log_probability(beta, l, x, phases)[0])
            law = np.zeros(l)
            law[0] = 1.0                                   # law of the sum of w(x) X_x modulo l
            for xi, wi in zip(x, w):
                t = exp(-beta * xi)
                step = np.zeros(l)
                for j in range(l):                         # X modulo l has law (1-t) t^j / (1 - t^l)
                    step[(wi * j) % l] += (1 - t) * t ** j / (1 - t ** l)
                law = np.array([sum(law[i] * step[(r - i) % l] for i in range(l)) for r in range(l)])
            assert abs(value - law[0]) < 1e-12, (n, l, value, law[0])
    print("P_l(beta): the formula with roots of unity agrees with a convolution modulo l")
    # derivatives of Psi by finite differences
    for n in (7, 30, 80):
        model = Model(n)
        beta, h = sqrt(n / 2) * 1.07, 1e-5
        v0, d1, d2 = model.psi(beta)
        vp, vm = model.psi(beta + h), model.psi(beta - h)
        assert abs((vp[0] - vm[0]) / (2 * h) - d1) < 1e-6 * (1 + abs(d1))
        assert abs((vp[1] - vm[1]) / (2 * h) - d2) < 1e-6 * d2
    print("derivatives of Psi_n agree with finite differences")
    # n = 2: Psi(beta) = 2 beta - log(1 - exp(-beta)), minimum log(27/4) at beta = log(3/2), sigma^2 = 6
    beta, psi, sigma2, sharp, c2 = Model(2).evaluate()
    assert abs(beta - log(1.5)) < 1e-12 and abs(psi - log(6.75)) < 1e-12 and abs(sigma2 - 6) < 1e-9
    print("n = 2: closed form reproduced")
    print("self-test passed")


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        selftest()
        return
    exact = read_exact()
    if len(sys.argv) == 2:
        f = formulas(int(sys.argv[1]))
        n = f["n"]
        print("n = %d" % n)
        print("  sqrt(2) n^(3/2)                               %.4f" % f["leading"])
        print("  (1) three terms of the theorem                %.4f" % f["theorem"])
        print("  (2) upper bound Psi_n(beta*)                  %.4f   (beta* = %.6f, sigma^2 = %.4f)"
              % (f["psi"], f["beta"], f["sigma2"]))
        print("      Conjecture 8.1 of the paper, logarithm    %.6f" % f["paper"])
        print("  (3) sharp formula without c2, logarithm       %.6f" % f["sharp"])
        print("  (4) sharp formula, logarithm                  %.6f   (c2 = %.3e)" % (f["sharp"] + log(1 + f["c2"]), f["c2"]))
        if n in exact:
            print("  exact log R(n)                                %.6f" % exact[n])
        return
    if not exact:
        sys.exit("R_values.txt not found")
    nmax = max(exact)
    errors = {}
    print("  n    log R(n)   theorem (1)   bound (2)   Conj. 8.1/R - 1   (3)/R - 1    (4)/R - 1")
    for n in range(2, nmax + 1):
        f = formulas(n)
        assert exact[n] <= f["psi"] + 1e-9, n                             # the proved upper bound
        errors[n] = (exp(f["paper"] - exact[n]) - 1, exp(f["sharp"] - exact[n]) - 1,
                     exp(f["sharp"] - exact[n]) * (1 + f["c2"]) - 1)
        if n % 10 == 0 or n == nmax:
            print("%3d  %10.4f  %10.4f  %10.4f      %+.2e       %+.2e    %+.2e"
                  % ((n, exact[n], f["theorem"], f["psi"]) + errors[n]), flush=True)
    print()
    print("largest relative error    Conj. 8.1    (3)        (4)")
    for lo, hi in ((2, 29), (30, 47), (48, nmax)):
        if lo <= hi:
            worst = [max(abs(errors[n][i]) for n in range(lo, hi + 1)) for i in range(3)]
            print("%3d <= n <= %3d          %.1e    %.1e    %.1e" % ((lo, hi) + tuple(worst)))


if __name__ == "__main__":
    main()
