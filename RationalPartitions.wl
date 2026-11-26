(* ::Package:: *)
(* RationalPartitions.wl — exact counts and saddle-point estimates
   for the diagonal rational partitions of K. Srinivasa Raghava.

   Load with Get["/path/to/RationalPartitions.wl"].
   All exact counting functions work independently of R_values.txt. *)

BeginPackage["RationalPartitions`"];

RationalPartitionKeys::usage =
  "RationalPartitionKeys[n] lists the reduced fractions a/b with 1 <= a <= n and 2 <= b <= n in increasing order.";
RationalPartitionKeyCount::usage =
  "RationalPartitionKeyCount[n] gives the number K(n) of allowed fractions.";
RationalPartitionCounts::usage =
  "RationalPartitionCounts[n,m] gives the exact list {R_n(0),...,R_n(m)} for positive integer n and nonnegative integer m.";
RationalPartitionCount::usage =
  "RationalPartitionCount[n] gives the exact diagonal count R(n). RationalPartitionCount[n,m] gives R_n(m) for the fixed key set of order n.";
RationalPartitionExamples::usage =
  "RationalPartitionExamples[n,m,k] gives up to k distinct partitions of m into keys of n. The default k is 10. Use All for all partitions of a small instance.";
RationalPartitionGeneratingFunction::usage =
  "RationalPartitionGeneratingFunction[n,z] gives the exact rational generating function Sum[R_n(m) z^m,{m,0,Infinity}] for a small fixed order n and indeterminate z.";
ReadRationalPartitionTable::usage =
  "ReadRationalPartitionTable[path] reads the supplied three-column reference file into an association indexed by n, with fields KeyCount and Count.";
RationalPartitionLogAsymptotic::usage =
  "RationalPartitionLogAsymptotic[n] evaluates the three-term large-n expansion for natural log R(n), using the tabulated decimal coefficients. Requires integer n >= 2.";
RationalPartitionSaddlePoint::usage =
  "RationalPartitionSaddlePoint[n] gives numerical saddle-point quantities for log R(n) using all primes. RationalPartitionSaddlePoint[n,\"LargePrimes\"] uses primes above Sqrt[n]. Requires integer n >= 2.";

Begin["`Private`"];

RationalPartitionKeys[n_Integer?Positive] :=
  Sort @ Flatten @ Table[
    Select[Range[n], CoprimeQ[#, b] &]/b, {b, 2, n}];

RationalPartitionKeyCount[n_Integer?Positive] :=
  2 Total[EulerPhi[Range[n]]] - n - 1;

(* For G(x)/(1-x^step), the coefficient at j is G_j plus the
   previous output at j-step. Grouping coefficients in rows of
   width step lets Accumulate apply this recurrence to whole rows. *)
geometricFactor[coeffs_List, step_Integer] := Module[
  {size = Length[coeffs], complete, rows, tail},
  If[step >= size, Return[coeffs]];
  complete = Quotient[size, step] step;
  rows = Accumulate[Partition[coeffs, step]];
  tail = Drop[coeffs, complete];
  Join[Flatten[rows], tail + Take[Last[rows], Length[tail]]]
];

RationalPartitionCounts[n_Integer?Positive, m_Integer?NonNegative] := Module[
  {coeffs = UnitVector[m + 1, 1], lattice = 1, denominators, remainingLCM,
   b, refinedLattice, refined, stepUnit, retainedLattice},
  If[m == 0 || n == 1, Return[coeffs]];
  (* Group by decreasing largest prime and, within a group, process lower
     powers first. Refine only when the next denominator actually needs it. *)
  denominators = SortBy[Range[2, n], Function[d,
    With[{factor = Last[FactorInteger[d]]}, {-First[factor], Last[factor], d}]]];
  remainingLCM = Reverse[FoldList[LCM, 1, Reverse[denominators]]];
  Do[
    b = denominators[[i]];
    refinedLattice = LCM[lattice, b];
    If[refinedLattice != lattice,
      refined = ConstantArray[0, m refinedLattice + 1];
      refined[[1 ;; -1 ;; Quotient[refinedLattice, lattice]]] = coeffs;
      coeffs = refined; refined = Null];
    stepUnit = Quotient[refinedLattice, b];
    Do[
      If[CoprimeQ[a, b], coeffs = geometricFactor[coeffs, a stepUnit]],
      {a, Min[n, m b]}];
    (* If the remaining denominators divide L, only exponents on both
       (1/refinedLattice) Z and (1/L) Z can still reach an integer.
       Their common lattice has denominator GCD[refinedLattice, L]. *)
    retainedLattice = GCD[refinedLattice, remainingLCM[[i + 1]]];
    If[retainedLattice != refinedLattice,
      coeffs = coeffs[[1 ;; -1 ;; Quotient[refinedLattice, retainedLattice]]]];
    lattice = retainedLattice,
    {i, Length[denominators]}];
  coeffs
];

RationalPartitionCount[n_Integer?Positive, m_Integer?NonNegative] :=
  Last[RationalPartitionCounts[n, m]];
RationalPartitionCount[n_Integer?Positive] := RationalPartitionCount[n, n];

RationalPartitionExamples[n_Integer?Positive, m_Integer?NonNegative,
    limit : (All | _Integer?NonNegative) : 10] := Which[
  limit === 0, {},
  m == 0, {{}},
  n == 1, {},
  True, IntegerPartitions[m, All, RationalPartitionKeys[n], limit]
];

(* Write each multiplicity as b q + r with 0 <= r < b. The q terms
   contribute the denominator product over (1 - z^a). The residue
   numerator is a polynomial of degree at most Sum[(b-1) a/b]. *)
RationalPartitionGeneratingFunction[n_Integer?Positive, z_Symbol] := Module[
  {parts, numerators, degreeBound, coeffs, denominator},
  If[n == 1, Return[1]];
  parts = RationalPartitionKeys[n];
  numerators = Numerator[parts];
  degreeBound = Floor[Total[numerators - parts]];
  coeffs = RationalPartitionCounts[n, degreeBound];
  Do[
    If[a <= degreeBound,
      coeffs = coeffs - Join[ConstantArray[0, a], Drop[coeffs, -a]]],
    {a, numerators}];
  denominator = Times @@ (1 - z^numerators);
  Factor[FromDigits[Reverse[coeffs], z]/denominator]
];

ReadRationalPartitionTable::format = "Expected contiguous rows starting at 1, each containing three nonnegative decimal integers, in `1`.";
ReadRationalPartitionTable[path_String] := Module[{lines, fields, rows},
  lines = ReadList[path, String];
  If[lines === $Failed, Return[$Failed]];
  lines = Select[StringTrim /@ lines, # != "" && !StringStartsQ[#, "#"] &];
  fields = StringSplit /@ lines;
  If[fields === {} || !AllTrue[fields, Length[#] == 3 &] ||
      !AllTrue[Flatten[fields], StringMatchQ[#, RegularExpression["[0-9]+"]] &],
    Message[ReadRationalPartitionTable::format, path]; Return[$Failed]];
  rows = Map[FromDigits, fields, {2}];
  If[rows[[All, 1]] =!= Range[Length[rows]],
    Message[ReadRationalPartitionTable::format, path]; Return[$Failed]];
  Association @ Map[#[[1]] -> <|"KeyCount" -> #[[2]], "Count" -> #[[3]]|> &, rows]
];

(* The constants are the rounded decimal coefficients supplied with the paper. *)
RationalPartitionLogAsymptotic[n_Integer /; n >= 2] :=
  N[n^(3/2) (Sqrt[2] - (26471306/10^10)/Log[n] -
      (10771199/10^10)/Log[n]^2), 16];

RationalPartitionLogAsymptotic::domain = "The order must be an integer at least 2.";
RationalPartitionLogAsymptotic[n_] :=
  (Message[RationalPartitionLogAsymptotic::domain]; $Failed);

(* For each prime use one representative from each conjugate pair of phases. *)
saddleBlock[keys_, n_, prime_, wp_] :=
  Module[{power, common, parts, residues, roots, phases},
    power = prime^(IntegerLength[n, prime] - 1);
    parts = Select[keys, Divisible[Denominator[#], power] &];
    common = Apply[LCM, Range[Quotient[n, power]]];
    residues = Mod[power common parts, prime];
    roots = N[Exp[2 Pi I Range[0, prime - 1]/prime], wp];
    phases = Table[roots[[1 + Mod[r residues, prime]]],
      {r, 1, Quotient[prime, 2]}];
    {prime, N[parts, wp], phases}
  ];

(* Return log P and its first two beta derivatives.  P is the zero-residue
   probability.  The r=0 term of the unnormalised phase sum is exactly one. *)
saddleBlockTerms[beta_, {prime_, parts_, phases_}] :=
  Module[{t, logNumerator, first, second, sums, z, weight, a, b,
    multiplicity, normalizer, slope},
    t = Exp[-beta parts];
    logNumerator = Total[Log[1 - t]];
    first = Total[parts t/(1 - t)];
    second = Total[parts^2 t/(1 - t)^2];
    multiplicity = If[prime == 2, 1, 2];
    sums = {1, 0, 0};
    Do[
      z = phase t;
      weight = Exp[logNumerator - Total[Log[1 - z]]];
      a = first - Total[parts z/(1 - z)];
      b = Total[parts^2 z/(1 - z)^2] - second;
      sums += multiplicity Re[weight {1, a, a^2 + b}],
      {phase, phases}];
    normalizer = First[sums];
    slope = sums[[2]]/normalizer;
    {Log[normalizer/prime], slope, sums[[3]]/normalizer - slope^2}
  ];

saddlePsiTerms[beta_, n_, keys_, blocks_] :=
  Module[{t = Exp[-beta keys], freeTerms},
    freeTerms = {beta n - Total[Log[1 - t]],
      n - Total[keys t/(1 - t)], Total[keys^2 t/(1 - t)^2]};
    freeTerms + Total[saddleBlockTerms[beta, #] & /@ blocks]
  ];

RationalPartitionSaddlePoint::domain =
  "Use an integer order of at least 2 and the prime scope All or \"LargePrimes\".";
RationalPartitionSaddlePoint::conv =
  "The saddle point calculation did not converge for order `1`.";

RationalPartitionSaddlePoint[n_Integer /; n >= 2, scope : (All | "LargePrimes") : All] :=
  Module[{wp = 32, keys, numericKeys, primes, selected, blocks, lattice,
    terms, lo, hi, beta, jets, next, h, v, vm2, vm1, vp1, vp2, d3, d4,
    correction, logGaussian, tolerance = 10^-17},
    keys = RationalPartitionKeys[n];
    numericKeys = N[keys, wp];
    primes = Prime[Range[PrimePi[n]]];
    selected = If[scope === All, primes, Select[primes, #^2 > n &]];
    blocks = saddleBlock[keys, n, #, wp] & /@ selected;
    lattice = (LCM @@ Range[n])/(Times @@ selected);
    terms[b_] := saddlePsiTerms[b, n, numericKeys, blocks];
    lo = hi = N[Sqrt[n/2], wp];
    While[terms[lo][[2]] > 0, lo /= 2];
    While[terms[hi][[2]] < 0, hi *= 2];
    beta = (lo + hi)/2;
    Do[
      jets = terms[beta];
      If[Abs[jets[[2]]] <= tolerance n, Break[]];
      If[jets[[2]] < 0, lo = beta, hi = beta];
      next = beta - jets[[2]]/jets[[3]];
      If[! TrueQ[lo < next < hi], next = (lo + hi)/2];
      beta = SetPrecision[next, wp],
      {100}];
    jets = terms[beta];
    v = jets[[3]];
    If[! TrueQ[v > 0 && Abs[jets[[2]]] <= tolerance n],
      Message[RationalPartitionSaddlePoint::conv, n]; Return[$Failed]];
    (* Differentiate the analytic variance with centred five-point stencils. *)
    h = beta/10000;
    {vm2, vm1, vp1, vp2} = terms[beta + # h][[3]] & /@ {-2, -1, 1, 2};
    d3 = (vm2 - 8 vm1 + 8 vp1 - vp2)/(12 h);
    d4 = (-vm2 + 16 vm1 - 30 v + 16 vp1 - vp2)/(12 h^2);
    correction = d4/(8 v^2) - 5 d3^2/(24 v^3);
    logGaussian = jets[[1]] - Log[lattice] - Log[2 Pi v]/2;
    <|"PrimeScope" -> scope, "Beta" -> N[beta, 14],
      "PsiMinimum" -> N[jets[[1]], 14], "Variance" -> N[v, 14],
      "LatticeDenominator" -> lattice, "LogGaussianEstimate" -> N[logGaussian, 14],
      "C2" -> N[correction, 12],
      "LogCorrectedEstimate" -> If[1 + correction > 0,
        N[logGaussian + Log[1 + correction], 13], Missing["NonpositiveCorrection"]]|>
  ];

RationalPartitionSaddlePoint[n_, scope_ : All] :=
  (Message[RationalPartitionSaddlePoint::domain]; $Failed);

End[];
EndPackage[];
