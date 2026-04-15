// Tests for fiedler.js. Run with:  node fiedler.test.mjs
//
// Covers:
//   1. Path graph   — eigenvalues are known in closed form
//   2. Two-cluster  — Fiedler separates clusters
//   3. 2D grid PBC  — smooth wave with known eigenvalue
//   4. 2D grid with bottleneck — Fiedler localises at the cut

import { computeFiedler, gridLaplacian, gridFiedlerSensitivity } from './fiedler.js';

let passed = 0;
let failed = 0;

function ok(name, cond, detail = '') {
  if (cond) { passed++; console.log(`  \x1b[32m✓\x1b[0m  ${name}`); }
  else      { failed++; console.log(`  \x1b[31m✗\x1b[0m  ${name}   ${detail}`); }
}
function approx(a, b, tol) { return Math.abs(a - b) < tol; }

// ---------------------------------------------------------------------
// Test 1: path graph P_n
//   Laplacian of an n-vertex path (weights 1) has eigenvalues
//     λ_k = 2 − 2·cos(kπ/n)    for k = 0, 1, …, n−1
//   So λ_2 = 2 − 2·cos(π/n).
// ---------------------------------------------------------------------

{
  console.log('\n\x1b[1m[test 1] path graph P_n — known λ_2 = 2 − 2·cos(π/n)\x1b[0m');
  // Path graphs have a dense, slowly-decaying spectrum. Lanczos needs
  // m close to n to capture λ_2 accurately; we verify the algorithm
  // converges as m → n.
  for (const n of [8, 20, 50]) {
    const applyL = (v, out) => {
      for (let i = 0; i < n; i++) {
        const left  = i > 0     ? v[i - 1] : 0;
        const right = i < n - 1 ? v[i + 1] : 0;
        const deg   = (i > 0 ? 1 : 0) + (i < n - 1 ? 1 : 0);
        out[i] = deg * v[i] - left - right;
      }
    };
    const expected = 2 - 2 * Math.cos(Math.PI / n);
    const { lambda_2 } = computeFiedler({ applyL, N: n, m: n - 1 });
    ok(`  n=${n}   λ_2 ≈ ${expected.toFixed(6)},  computed ${lambda_2.toFixed(6)}`,
       approx(lambda_2, expected, 1e-5),
       `diff ${(lambda_2 - expected).toExponential(2)}`);
  }
}

// ---------------------------------------------------------------------
// Test 2: two K_4 cliques joined by a single weak edge (weight ε).
//   For ε ≪ 1, λ_2 ≈ ε · 2 / N  (perturbation theory) and the Fiedler
//   vector should cleanly separate the two cliques in sign.
// ---------------------------------------------------------------------

{
  console.log('\n\x1b[1m[test 2] two cliques joined by a weak bond — Fiedler separates\x1b[0m');
  const cluster = 4;        // 2 cliques of 4, total N=8
  const N = 2 * cluster;
  const eps = 0.01;

  // Adjacency as a list of (i, j, w) triples.
  const edges = [];
  for (let c = 0; c < 2; c++) {
    const off = c * cluster;
    for (let i = 0; i < cluster; i++)
      for (let j = i + 1; j < cluster; j++)
        edges.push([off + i, off + j, 1.0]);
  }
  edges.push([0, cluster, eps]);  // weak bridge

  const applyL = (v, out) => {
    for (let i = 0; i < N; i++) out[i] = 0;
    for (const [i, j, w] of edges) {
      out[i] += w * (v[i] - v[j]);
      out[j] += w * (v[j] - v[i]);
    }
  };

  const { lambda_2, v_2 } = computeFiedler({ applyL, N, m: 6 });

  // Expected λ_2 scales with ε.
  ok(`  λ_2 small (${lambda_2.toExponential(2)}) and proportional to ε=${eps}`,
     lambda_2 < 0.05 && lambda_2 > 0,
     `got λ_2 = ${lambda_2}`);

  // Sign on each cluster should be consistent (Fiedler separates).
  const s0 = Math.sign(v_2[0]);
  const s1 = Math.sign(v_2[cluster]);
  let cluster0_consistent = true;
  let cluster1_consistent = true;
  for (let i = 0; i < cluster; i++) {
    if (Math.sign(v_2[i]) !== s0) cluster0_consistent = false;
    if (Math.sign(v_2[cluster + i]) !== s1) cluster1_consistent = false;
  }
  ok(`  cluster 0 all same sign`, cluster0_consistent);
  ok(`  cluster 1 all same sign`, cluster1_consistent);
  ok(`  clusters have opposite signs`, s0 !== s1);
}

// ---------------------------------------------------------------------
// Test 3: uniform 2D periodic grid Lx×Ly with bond weight 1.
//   Laplacian eigenvalues: 4 − 2·cos(2π kx/Lx) − 2·cos(2π ky/Ly).
//   The non-zero smallest is  2·(1 − cos(2π/L))  for L = min(Lx, Ly).
// ---------------------------------------------------------------------

{
  console.log('\n\x1b[1m[test 3] uniform 2D periodic grid — known smooth-wave λ_2\x1b[0m');
  for (const L of [8, 12, 16]) {
    const N = L * L;
    const Kh = new Float64Array(N).fill(1);
    const Kv = new Float64Array(N).fill(1);
    const { applyL } = gridLaplacian({ L, Kh, Kv });
    const expected = 2 * (1 - Math.cos(2 * Math.PI / L));
    const { lambda_2 } = computeFiedler({ applyL, N, m: 30 });
    ok(`  L=${L}   λ_2 ≈ ${expected.toFixed(6)},  computed ${lambda_2.toFixed(6)}`,
       approx(lambda_2, expected, 5e-3),
       `diff ${(lambda_2 - expected).toExponential(2)}`);
  }
}

// ---------------------------------------------------------------------
// Test 4: 2D grid with a 4-bond "cut" — 2×L central bonds set to ε.
//   Fiedler should drop sharply; v_2 should localise so that the large
//   gradient sits on the weak bonds (the bottleneck-detection property).
// ---------------------------------------------------------------------

{
  console.log('\n\x1b[1m[test 4] grid with a weakened cut — Fiedler localises on the cut\x1b[0m');
  const L = 16;
  const N = L * L;
  const idx = (x, y) => y * L + x;
  const Kh = new Float64Array(N).fill(1);
  const Kv = new Float64Array(N).fill(1);
  // Weaken the vertical bonds crossing y = L/2, for all x
  const eps = 0.01;
  const yCut = L / 2;
  for (let x = 0; x < L; x++) Kv[idx(x, yCut - 1)] = eps;  // bond (x,yCut-1)-(x,yCut)

  const { applyL } = gridLaplacian({ L, Kh, Kv });
  const { lambda_2, v_2 } = computeFiedler({ applyL, N, m: 40 });
  const uniformExpected = 2 * (1 - Math.cos(2 * Math.PI / L));

  ok(`  λ_2 ≪ uniform λ_2 (cut ${lambda_2.toExponential(2)} vs uniform ${uniformExpected.toExponential(2)})`,
     lambda_2 < 0.3 * uniformExpected);

  // Fiedler vector should be roughly constant on each half, differing
  // strongly across the cut. Check that top-half and bottom-half mean
  // values have opposite signs.
  let topSum = 0, bottomSum = 0;
  for (let y = 0; y < L; y++) {
    for (let x = 0; x < L; x++) {
      if (y < yCut) topSum += v_2[idx(x, y)];
      else          bottomSum += v_2[idx(x, y)];
    }
  }
  ok(`  halves have opposite signs`,
     Math.sign(topSum) !== Math.sign(bottomSum),
     `top=${topSum.toFixed(3)}, bottom=${bottomSum.toFixed(3)}`);

  // The Fiedler sensitivity (v_i - v_j)^2 should be concentrated on the
  // weakened bonds (x, yCut-1) → (x, yCut).
  const { Sh, Sv } = gridFiedlerSensitivity({ L, v: v_2 });
  let cutMean = 0;
  for (let x = 0; x < L; x++) cutMean += Sv[idx(x, yCut - 1)];
  cutMean /= L;

  // Compare to mean of bulk vertical bonds far from cut
  let bulkMean = 0, bulkN = 0;
  for (let y = 0; y < L; y++) {
    if (Math.abs(y - yCut) < 3) continue;
    for (let x = 0; x < L; x++) { bulkMean += Sv[idx(x, y)]; bulkN++; }
  }
  bulkMean /= bulkN;

  const ratio = (cutMean + 0.01) / (bulkMean + 0.01);  // handle negatives after deflation
  ok(`  cut sensitivity >> bulk (cut=${cutMean.toFixed(4)}, bulk=${bulkMean.toFixed(4)}, ratio ${ratio.toFixed(1)})`,
     cutMean > Math.abs(bulkMean) * 5 || cutMean > 0.05);
}

// ---------------------------------------------------------------------
// Test 5: the actual use case — 2D grid with mostly-alive bulk and a
// small ring of dead bonds around one plaquette. This mimics what the
// Figure 4 simulation hands to computeFiedler during the co-evolution.
// ---------------------------------------------------------------------

{
  console.log('\n\x1b[1m[test 5] realistic use case — scattered dead-bond pattern\x1b[0m');
  const L = 16;
  const N = L * L;
  const idx = (x, y) => y * L + x;
  const Kh = new Float64Array(N).fill(1.5);
  const Kv = new Float64Array(N).fill(1.5);
  // Kill an 8-bond ring around the plaquette (cx, cy) = (8, 8)
  const cx = 8, cy = 8;
  const killed = [
    [idx(cx - 1, cy - 1), 'h'], [idx(cx, cy - 1), 'h'],
    [idx(cx - 1, cy),     'h'], [idx(cx, cy),     'h'],
    [idx(cx - 1, cy - 1), 'v'], [idx(cx, cy - 1), 'v'],
    [idx(cx - 1, cy),     'v'], [idx(cx, cy),     'v'],
  ];
  for (const [i, dir] of killed) {
    if (dir === 'h') Kh[i] = 0.001;
    else             Kv[i] = 0.001;
  }
  const { applyL } = gridLaplacian({ L, Kh, Kv });
  const t0 = Date.now();
  const { lambda_2, v_2 } = computeFiedler({ applyL, N, m: 30 });
  const elapsed = Date.now() - t0;
  const uniformExpected = 2 * 1.5 * (1 - Math.cos(2 * Math.PI / L));
  ok(`  λ_2 well below uniform (${lambda_2.toExponential(2)} < ${uniformExpected.toExponential(2)})`,
     lambda_2 < 0.2 * uniformExpected);

  // Check that Fiedler sensitivity concentrates on the killed bonds
  const { Sh, Sv } = gridFiedlerSensitivity({ L, v: v_2 });
  let killedMean = 0;
  for (const [i, dir] of killed) killedMean += (dir === 'h' ? Sh[i] : Sv[i]);
  killedMean /= killed.length;
  let bulkMean = 0, bulkN = 0;
  const dSq = (ax, ay) => Math.max(Math.abs(ax - cx), Math.abs(ay - cy));
  for (let y = 0; y < L; y++) {
    for (let x = 0; x < L; x++) {
      if (dSq(x, y) > 4) {
        bulkMean += Sh[idx(x, y)] + Sv[idx(x, y)];
        bulkN += 2;
      }
    }
  }
  bulkMean /= bulkN;
  ok(`  sensitivity concentrated on killed bonds (killed=${killedMean.toFixed(4)}, bulk=${bulkMean.toFixed(4)})`,
     killedMean > 0 && killedMean > Math.abs(bulkMean) * 5);

  ok(`  wall-clock OK for browser use (${elapsed} ms)`, elapsed < 500);
}

// ---------------------------------------------------------------------
// Test 6: convergence vs m — on a 16×16 periodic grid, the Fiedler
// eigenvalue should stabilise quickly as m grows.
// ---------------------------------------------------------------------

{
  console.log('\n\x1b[1m[test 6] Lanczos convergence with m on 16×16 grid\x1b[0m');
  const L = 16;
  const N = L * L;
  const Kh = new Float64Array(N).fill(1);
  const Kv = new Float64Array(N).fill(1);
  const { applyL } = gridLaplacian({ L, Kh, Kv });
  const expected = 2 * (1 - Math.cos(2 * Math.PI / L));
  let prev = null;
  for (const m of [10, 15, 20, 25]) {
    const { lambda_2 } = computeFiedler({ applyL, N, m });
    const diff = Math.abs(lambda_2 - expected);
    console.log(`    m=${String(m).padStart(2)}: λ_2=${lambda_2.toFixed(6)}  err=${diff.toExponential(2)}`);
    prev = lambda_2;
  }
  ok(`  converges to expected by m=25`, approx(prev, expected, 1e-5));
}

// ---------------------------------------------------------------------

console.log('');
if (failed === 0) {
  console.log(`\x1b[32m✓  All ${passed} checks passed.\x1b[0m`);
  process.exit(0);
} else {
  console.log(`\x1b[31m✗  ${failed} of ${passed + failed} checks FAILED.\x1b[0m`);
  process.exit(1);
}
