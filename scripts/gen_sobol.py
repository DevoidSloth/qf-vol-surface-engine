"""Generate the Sobol direction-number table for cpp/include/vse/sobol_data.hpp.

The direction numbers are those of

    S. Joe and F. Y. Kuo, "Constructing Sobol sequences with better
    two-dimensional projections", SIAM J. Sci. Comput. 30, 2635-2654 (2008),

published by the authors under the BSD licence reproduced in the generated
header. They are not interchangeable with a set of arbitrary odd initialisers:
the whole point of the Joe-Kuo construction is that the two-dimensional
projections of the sequence are well distributed, and a Sobol sequence built
from unfortunate initialisers can be *worse* than pseudo-random on exactly the
low-dimensional projections a Brownian bridge concentrates the variance into.

Source data is fetched from the Sobol.jl repository, which redistributes the
tables verbatim with their licence:

    python scripts/gen_sobol.py [--dimensions 1024]

Cached copies under scripts/_cache are reused, so the generation is reproducible
offline once it has been run.
"""

from __future__ import annotations

import argparse
import pathlib
import ssl
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
CACHE = HERE / "_cache"
OUT = HERE.parent / "cpp" / "include" / "vse" / "sobol_data.hpp"
BASE = "https://raw.githubusercontent.com/stevengj/Sobol.jl/master/src/"


def fetch(name: str) -> str:
    CACHE.mkdir(exist_ok=True)
    cached = CACHE / name
    if cached.exists():
        return cached.read_text()
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(BASE + name, timeout=180, context=ctx) as r:
        text = r.read().decode()
    cached.write_text(text)
    return text


def split_polynomial(a: int) -> tuple[int, int]:
    """Degree and interior coefficients of a primitive polynomial.

    Two encodings are in circulation and confusing them silently produces a
    sequence that looks plausible and is not a Sobol sequence at all -- its
    dimensions past the first come out with a mean of 0.286 rather than 0.5.

    The source data encodes the FULL polynomial
        p(z) = a_0 + a_1 z + ... + a_s z^s,   a_0 = a_s = 1
    as the integer whose i-th bit is a_i. Joe and Kuo's own tables, and the
    recurrence that consumes them, use only the INTERIOR coefficients
    a_1..a_{s-1}. So the degree is the index of the highest set bit, and the
    interior encoding is the bits strictly between the two ends.

    Checked against the published table: dimension 3 gives (s=2, a=1),
    dimension 7 gives (s=4, a=4), dimension 9 gives (s=5, a=4).
    """
    s = -1
    while a >> (s + 1):
        s += 1
    interior = (a >> 1) & ((1 << (s - 1)) - 1) if s >= 1 else 0
    return s, interior


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dimensions",
        type=int,
        default=1024,
        help="how many Sobol dimensions to emit (default 1024)",
    )
    args = ap.parse_args()
    n = args.dimensions

    a_text = fetch("sobol_a.csv")
    m_text = fetch("sobol_minit.csv")

    poly = [int(line) for line in a_text.split() if line.strip()]
    rows = [[int(v) for v in line.split(",") if v.strip()] for line in m_text.strip().split("\n")]

    # Dimension 1 has no polynomial (its direction numbers are all ones), so
    # entry j of these tables belongs to dimension j + 2.
    count = n - 1
    if count > len(poly):
        raise SystemExit(f"only {len(poly) + 1} dimensions available, {n} requested")

    flat: list[int] = []
    offsets: list[int] = []
    degrees: list[int] = []
    interior: list[int] = []
    for j in range(count):
        d, a_interior = split_polynomial(poly[j])
        offsets.append(len(flat))
        degrees.append(d)
        interior.append(a_interior)
        for i in range(d):
            flat.append(rows[i][j])

    lic = [
        '// Direction numbers from S. Joe and F. Y. Kuo, "Constructing Sobol',
        '// sequences with better two-dimensional projections", SIAM J. Sci. Comput.',
        "// 30, 2635-2654 (2008), redistributed under the following licence:",
        "//",
        "//   Copyright (c) 2008, Frances Y. Kuo and Stephen Joe. All rights reserved.",
        "//",
        "//   Redistribution and use in source and binary forms, with or without",
        "//   modification, are permitted provided that the following conditions are",
        "//   met: redistributions of source code must retain the above copyright",
        "//   notice, this list of conditions and the following disclaimer;",
        "//   redistributions in binary form must reproduce them in the documentation",
        "//   and/or other materials provided with the distribution; neither the names",
        "//   of the copyright holders nor the names of the University of New South",
        "//   Wales and the University of Waikato and its contributors may be used to",
        "//   endorse or promote products derived from this software without specific",
        "//   prior written permission.",
        "//",
        '//   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY',
        "//   EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE",
        "//   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR",
        "//   PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE",
        "//   LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR",
        "//   CONSEQUENTIAL DAMAGES HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY.",
    ]

    def emit(name: str, values: list[int], ctype: str, per_line: int = 16) -> list[str]:
        out = [f"inline constexpr {ctype} {name}[] = {{"]
        for i in range(0, len(values), per_line):
            chunk = ", ".join(str(v) for v in values[i : i + per_line])
            out.append("    " + chunk + ",")
        out.append("};")
        return out

    lines = [
        "// GENERATED by scripts/gen_sobol.py -- do not edit by hand.",
        "//",
        *lic,
        "#pragma once",
        "",
        "#include <cstdint>",
        "",
        "namespace vse::sobol_data {",
        "",
        f"inline constexpr int kMaxDimensions = {n};",
        "",
        "/// Interior coefficients a_1..a_{s-1} of each primitive polynomial, for",
        "/// dimensions 2 .. kMaxDimensions. NOT the full polynomial: the leading and",
        "/// constant terms are always 1 and the recurrence does not use them.",
        *emit("kPolynomial", interior, "std::uint32_t"),
        "",
        "/// Degree of each polynomial, which is how many initial direction numbers",
        "/// that dimension needs before the recurrence takes over.",
        *emit("kDegree", degrees, "std::uint8_t", 32),
        "",
        "/// Offset of each dimension's initial direction numbers in kInitial.",
        *emit("kOffset", offsets, "std::uint32_t"),
        "",
        "/// Initial direction numbers, concatenated.",
        *emit("kInitial", flat, "std::uint32_t"),
        "",
        "}  // namespace vse::sobol_data",
        "",
    ]

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}")
    print(
        f"  {n} dimensions, {len(flat)} initial direction numbers, "
        f"max degree {max(degrees)}, {OUT.stat().st_size // 1024} KB"
    )


if __name__ == "__main__":
    main()
