"""Wilson score interval and Fisher's exact test for 2x2 tables (standard library only)."""
from math import comb, sqrt

Z95 = 1.959963984540054


def wilson(k, n, z=Z95):
    """95% Wilson score interval for k successes in n trials."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def _hypergeom(a, row1, row2, col1):
    return comb(row1, a) * comb(row2, col1 - a) / comb(row1 + row2, col1)


def fisher(k1, n1, k2, n2):
    """Fisher's exact test of k1/n1 against k2/n2.

    Returns (two_sided, greater): two_sided sums every table no more likely than the observed one
    (the convention of scipy.stats.fisher_exact); greater is P(first group has >= k1 successes).
    """
    col1 = k1 + k2
    lo, hi = max(0, col1 - n2), min(n1, col1)
    probs = {a: _hypergeom(a, n1, n2, col1) for a in range(lo, hi + 1)}
    observed = probs[k1]
    two_sided = sum(p for p in probs.values() if p <= observed * (1 + 1e-7))
    greater = sum(p for a, p in probs.items() if a >= k1)
    return min(1.0, two_sided), min(1.0, greater)
