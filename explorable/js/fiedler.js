/*
 * fiedler.js — Fiedler vector computation for weighted graph Laplacians.
 *
 * Given a weighted graph with Laplacian matrix L (with constants in the
 * null space), compute:
 *     λ_2  — the smallest non-zero eigenvalue (algebraic connectivity)
 *     v_2  — the corresponding eigenvector (Fiedler vector)
 *
 * Algorithm: Lanczos tridiagonalization with full reorthogonalization
 * and explicit deflation of the constant null space, followed by
 * shift-and-invert iteration on the resulting small tridiagonal matrix.
 * Lanczos lifts the difficult eigenvalue problem on N dimensions to a
 * tractable tridiagonal problem on m ≪ N dimensions; the reverse map
 * recovers the eigenvector.
 *
 * This converges fast even on smooth 2D grids where naive power
 * iteration would stall — the spectral gap problem that blocked the
 * in-place browser implementation of Figure 4.
 *
 * Public API:
 *
 *   computeFiedler({ applyL, N, m, seed })
 *     applyL: (v, out) => void — computes  out = L · v
 *     N:      dimension of the vector space
 *     m:      Krylov subspace dimension (default 24)
 *     seed:   RNG seed for reproducibility (default 1)
 *   Returns { lambda_2, v_2 }.
 *
 *   gridLaplacian({ L, Kh, Kv })
 *     Convenience: applyL for a 2D periodic L×L grid with horizontal
 *     bond weights Kh[y*L+x] (between (x,y) and ((x+1)%L, y)) and
 *     vertical bond weights Kv[y*L+x] (between (x,y) and (x, (y+1)%L)).
 */

// ---------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function deflateConst(v) {
  // Project v onto the subspace orthogonal to the all-ones vector.
  // (For graph Laplacians, constants span the null space.)
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i];
  const mean = s / v.length;
  for (let i = 0; i < v.length; i++) v[i] -= mean;
}

function norm(v) {
  return Math.sqrt(dot(v, v));
}

function normalize(v) {
  const n = norm(v);
  if (n < 1e-15) return 0;
  for (let i = 0; i < v.length; i++) v[i] /= n;
  return n;
}

// Small seeded RNG (Mulberry32) for reproducible Lanczos seed vectors
function mulberry32(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6D2B79F5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------------------------------------------------------------------
// Jacobi eigendecomposition on a dense symmetric m×m matrix.
// Simple, robust, O(m³·sweeps) — fine for the small tridiagonal we get
// from Lanczos (m ≤ 50).
// ---------------------------------------------------------------------

function jacobiEig(A) {
  const m = A.length;
  // V starts as identity; columns become eigenvectors.
  const V = Array.from({ length: m }, (_, i) => {
    const row = new Float64Array(m);
    row[i] = 1;
    return row;
  });

  const MAX_SWEEPS = 60;
  for (let sweep = 0; sweep < MAX_SWEEPS; sweep++) {
    let offSq = 0;
    for (let i = 0; i < m; i++) {
      for (let j = i + 1; j < m; j++) offSq += A[i][j] * A[i][j];
    }
    if (offSq < 1e-24) break;

    for (let p = 0; p < m - 1; p++) {
      for (let q = p + 1; q < m; q++) {
        const apq = A[p][q];
        if (Math.abs(apq) < 1e-14) continue;

        // Rotation: choose θ so the (p,q) element becomes 0.
        const diff = A[q][q] - A[p][p];
        let t;
        if (Math.abs(apq) < 1e-20 * Math.abs(diff)) {
          t = apq / diff;
        } else {
          const theta = diff / (2 * apq);
          const sign = theta >= 0 ? 1 : -1;
          t = sign / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        }
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;

        // Update diagonal
        A[p][p] -= t * apq;
        A[q][q] += t * apq;
        A[p][q] = 0;
        A[q][p] = 0;

        // Update other rows/columns of A
        for (let r = 0; r < m; r++) {
          if (r !== p && r !== q) {
            const arp = A[r][p], arq = A[r][q];
            A[r][p] = c * arp - s * arq;
            A[p][r] = A[r][p];
            A[r][q] = s * arp + c * arq;
            A[q][r] = A[r][q];
          }
        }
        // Accumulate eigenvectors
        for (let r = 0; r < m; r++) {
          const vrp = V[r][p], vrq = V[r][q];
          V[r][p] = c * vrp - s * vrq;
          V[r][q] = s * vrp + c * vrq;
        }
      }
    }
  }

  // Extract and sort eigenvalues ascending
  const evals = new Float64Array(m);
  for (let i = 0; i < m; i++) evals[i] = A[i][i];
  const order = Array.from({ length: m }, (_, i) => i)
                     .sort((a, b) => evals[a] - evals[b]);
  const sortedEvals = new Float64Array(m);
  // evecs[k] is the k-th eigenvector as a length-m array
  const sortedEvecs = [];
  for (let k = 0; k < m; k++) {
    sortedEvals[k] = evals[order[k]];
    const col = new Float64Array(m);
    for (let i = 0; i < m; i++) col[i] = V[i][order[k]];
    sortedEvecs.push(col);
  }
  return { evals: sortedEvals, evecs: sortedEvecs };
}

// Find the smallest eigenvalue / eigenvector of a symmetric tridiagonal
// matrix by building the dense matrix and calling jacobiEig.
function smallestEigTridiag(alpha, beta) {
  const m = alpha.length;
  if (m === 1) return { eigenvalue: alpha[0], eigenvector: new Float64Array([1]) };
  const T = Array.from({ length: m }, () => new Float64Array(m));
  for (let i = 0; i < m; i++) {
    T[i][i] = alpha[i];
    if (i > 0) T[i][i - 1] = beta[i - 1];
    if (i < m - 1) T[i][i + 1] = beta[i];
  }
  const { evals, evecs } = jacobiEig(T);
  return { eigenvalue: evals[0], eigenvector: evecs[0] };
}

// ---------------------------------------------------------------------
// Lanczos with full reorthogonalization + null-space deflation
// ---------------------------------------------------------------------

export function computeFiedler({ applyL, N, m = 24, seed = 1 }) {
  m = Math.min(m, N - 1);
  const rng = mulberry32(seed);

  // Krylov basis V[0..j]
  const V = [];
  const alpha = new Float64Array(m);
  const beta  = new Float64Array(Math.max(1, m));  // beta[j] = ||w|| after (j-1)-th step

  // Starting vector: a centered ordinal index plus small noise. The
  // ordinal has strong projection onto low-frequency graph modes (the
  // Fiedler eigenvector of most graphs is smooth), which helps avoid
  // Lanczos pathologies on small graphs. The noise breaks any exact
  // symmetries with the eigenbasis and makes the algorithm seedable.
  const v0 = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    v0[i] = (i - (N - 1) / 2) + (rng() - 0.5) * 0.1;
  }
  deflateConst(v0);
  if (normalize(v0) < 1e-15) throw new Error('computeFiedler: degenerate start');
  V.push(v0);

  let actualM = 0;
  for (let j = 0; j < m; j++) {
    // w = L * V[j]
    const w = new Float64Array(N);
    applyL(V[j], w);

    // alpha_j = <V[j], w>
    alpha[j] = dot(V[j], w);

    // w -= alpha_j * V[j]  and  w -= beta_j * V[j-1]
    for (let i = 0; i < N; i++) w[i] -= alpha[j] * V[j][i];
    if (j > 0) {
      const bj = beta[j];
      for (let i = 0; i < N; i++) w[i] -= bj * V[j - 1][i];
    }

    // Full reorthogonalization against all previous basis vectors.
    // Lanczos loses orthogonality numerically without this; full
    // reorth is O(mN) per step which is fine for m ≲ 50.
    for (let k = 0; k <= j; k++) {
      const d = dot(w, V[k]);
      if (d !== 0) for (let i = 0; i < N; i++) w[i] -= d * V[k][i];
    }
    // Also deflate constants explicitly for paranoia — a perfect
    // graph Laplacian applyL won't introduce constant contamination,
    // but floating-point accumulation can.
    deflateConst(w);

    const bNext = norm(w);
    beta[j + 1 < beta.length ? j + 1 : beta.length - 1] = bNext;

    actualM = j + 1;

    // Invariant breakdown — Krylov space has saturated, can't extend.
    if (bNext < 1e-12) break;

    if (j < m - 1) {
      for (let i = 0; i < N; i++) w[i] /= bNext;
      V.push(w);
    }
  }

  // Solve the small tridiagonal eigenvalue problem for smallest eigenvalue.
  const aSlice = alpha.slice(0, actualM);
  const bSlice = beta.slice(1, actualM);       // subdiagonal entries
  const small = smallestEigTridiag(aSlice, bSlice);
  const lambda_2 = small.eigenvalue;
  const c = small.eigenvector;                 // length actualM

  // Reconstruct Fiedler vector in the original N-dimensional space:
  //   v_2 = Σ_j c[j] V[j]
  const v_2 = new Float64Array(N);
  for (let j = 0; j < actualM; j++) {
    const cj = c[j];
    const Vj = V[j];
    for (let i = 0; i < N; i++) v_2[i] += cj * Vj[i];
  }
  deflateConst(v_2);
  normalize(v_2);

  return { lambda_2, v_2 };
}

// ---------------------------------------------------------------------
// Convenience: 2D periodic grid Laplacian apply function
// ---------------------------------------------------------------------

export function gridLaplacian({ L, Kh, Kv }) {
  const N = L * L;
  const idx = (x, y) => y * L + x;
  return {
    N,
    applyL(v, out) {
      for (let y = 0; y < L; y++) {
        for (let x = 0; x < L; x++) {
          const i = idx(x, y);
          const xp = (x + 1) % L, xm = (x - 1 + L) % L;
          const yp = (y + 1) % L, ym = (y - 1 + L) % L;
          const kE = Kh[idx(x, y)];
          const kW = Kh[idx(xm, y)];
          const kN = Kv[idx(x, ym)];
          const kS = Kv[idx(x, y)];
          out[i] = (kE + kW + kN + kS) * v[i]
                 - kE * v[idx(xp, y)]
                 - kW * v[idx(xm, y)]
                 - kN * v[idx(x, ym)]
                 - kS * v[idx(x, yp)];
        }
      }
    },
  };
}

// ---------------------------------------------------------------------
// Convenience: edge sensitivity (v_i - v_j)^2 over the graph bonds,
// with the graph mean subtracted. This is the Fiedler sensitivity
// S_ij that drives the structural channel in the CLR.
// ---------------------------------------------------------------------

export function gridFiedlerSensitivity({ L, v }) {
  const N = L * L;
  const idx = (x, y) => y * L + x;
  const Sh = new Float64Array(N);
  const Sv = new Float64Array(N);
  let sum = 0;
  for (let y = 0; y < L; y++) {
    for (let x = 0; x < L; x++) {
      const i = idx(x, y);
      const dxE = v[i] - v[idx((x + 1) % L, y)];
      const dxS = v[i] - v[idx(x, (y + 1) % L)];
      Sh[i] = dxE * dxE;
      Sv[i] = dxS * dxS;
      sum += Sh[i] + Sv[i];
    }
  }
  const mean = sum / (2 * N);
  for (let i = 0; i < N; i++) { Sh[i] -= mean; Sv[i] -= mean; }
  return { Sh, Sv, mean };
}
