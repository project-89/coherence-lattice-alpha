# Curriculum Review — The Living Lattice — 2026-08-13

*Goal set by Michael: make the site approachable assuming people know nothing;
use visuals that make it easy to get started; use noise and sound; add
supplemental modules for the math (gauge theory, rotations, whatever else is
needed). "I am not good at math so this is as much for me as others."*

*Method: full prerequisite audit of every live chapter (agent-assisted read of
all prose, verified by grep), plus this session's build of the first remedial
slice. The audit's per-chapter detail is summarized here; the build is live in
the working copy.*

---

## 1. The headline findings

1. **The site teaches its physics well and its mathematics almost not at all.**
   The five foundations chapters genuinely build phase → coupling → capital →
   arrow → memory from zero, and mostly keep their "every term is built before
   it is used" promise. What nobody teaches is the *mathematical language the
   later chapters switch into*: manifold, topology, rotation groups,
   eigenvectors, singular values, coarse-graining. The reader who follows
   foundations perfectly still falls off at paradigm §08.

2. **The ten most load-bearing never-taught concepts, ranked by how many
   chapters silently assume them:**
   1. *Manifold* (12 chapters) — only ever pointed at ("a thin surface")
   2. *Attention head / residual stream / LayerNorm* (6 chapters) — the
      chapter meant to define these (transformers §1) is still "planned",
      so the transformers track's actual entry point assumes its own
      missing prerequisite
   3. *Topology as a subject* (7 chapters) — demonstrated, never defined
   4. *PAS ≈ 0.766* — acronym never expanded anywhere
   5. *Fiedler eigenvector / graph Laplacian* — foundations §03 promises
      "Chapter 7 will unpack" — **no chapter anywhere fulfills this**
   6. *Winding number / π₁ / H₁* — three names, never reconciled as one idea
   7. *Coarse-graining* — the entire thesis of §13 rests on it, undefined
   8. *Free energy* — formula given, concept never grounded
   9. *Singular values / participation ratio*
   10. *Legality's four sub-conditions* — named as a list, never unpacked

3. **Least approachable chapters (fix-first list):** paradigm §13 (scale
   ladder — its central sentence "graphs with the same coarsened structure
   produce isometric slow manifolds" is the densest load-bearing claim on the
   site), paradigm §08 (shape — five hard terms in two sentences: d_eff
   formula, singular values, attention heads, GPT-2/layer-11, logit cosine),
   transformers §03 (entry point with missing prerequisites). Foundations §03
   is a near-miss (Fiedler/Laplacian/legality wall).

4. **Models to imitate — the site already knows how to do this:**
   *alpha §01-prelude* (one concept per figure, builds from a single
   oscillator; gentler than anything in foundations — the gold standard),
   *foundations §01* (click-to-expand equations + "In words" recaps),
   *paradigm §11's* rock/thermostat/transformer/mind table, *paradigm §17
   coda* (a reader who reads only it gets the whole gist), and the paradigm
   lander's "whole idea in plain words" box.

5. **Small correctness items found by the audit, fixed this session:**
   foundations §02 credited I_phase to "Chapter 2" (it is Chapter 1) in two
   places, and pointed PLMs at "Chapter 6" (foundations has five chapters;
   PLMs are Chapter 5).

---

## 2. What was built this session (live in the working copy)

**Sound infrastructure** (`js/common.js`): a shared AudioContext behind
per-figure sound toggles (off by default, quiet master gain, per-figure
sub-gains, gesture-gated as browsers require) and a `makeVoices` helper —
sine voices whose pitch/level can track sim state every frame. House rule
established in the code comments: *sonification should be literal where
possible* (voices = oscillators, pitch = actual instantaneous frequency, so
beats and lock are real physics in the mix); where a mapping is used instead
(loudness = cos agreement), **the caption must say so**.

**Primers track** (`primers/`) — a new essay in the library, "The math you
need, by ear and eye. Assume nothing." Linked from a new lander card plus a
"New here, or allergic to math?" line in the hero. Two chapters live:

- **Primer 00 — Phase, beats, and the sound of sync.** Circle→wave→pitch;
  real two-tone beating with the envelope drawn at the detune rate; two
  coupled oscillators with a K slider where you *hear* beats slow near the
  threshold (critical slowing!) and stop at lock. Verified: locked readout at
  K=3, slipping at K=0.5, zero console errors.
- **Primer 01 — Rotations: same shape, another angle.** Rigid rotation with
  live invariant-distance readouts; dot-product-as-shadow with the same
  loudness mapping (one audio language site-wide); a find-the-hidden-rotation
  game between two "star maps" with a match meter, honesty-flagged as a
  cartoon of §15's real 99.3% embedder alignment. Verified: sweep finds 100%
  at the hidden angle.

**Sound in the essays**: foundations §01 fig A (agreement→loudness, labeled
mapping) and fig B (8 of the ring's 24 oscillators as voices at their real
instantaneous frequencies — the sync transition is audible as a detuned
cluster pulling into one pitch). Cross-links added: foundations §01 → primer
00; paradigm §15 and §16 → primer 01.

---

## 3. The roadmap (priority order)

### Tier 1 — close the entry wounds
1. **Write transformers §1 "A transformer is coupled oscillators"** (and its
   §2), or at minimum the primer "Inside a transformer, for people who have
   never opened one" (planned card exists). This single gap breaks six
   chapters. One diagram per term: attention head, residual stream,
   LayerNorm.
2. **Primer: Shapes of spaces — topology and manifolds** (planned card
   exists). Stretching-not-tearing; manifold = locally flat; winding/π₁/H₁
   reconciled as one counting idea. Unlocks §08, §09, §11, §13.
3. **Rewrite paradigm §08's opening** to route through the primers and
   introduce its five hard terms one at a time (it has the interactives to
   do it — they arrived from the transformers essay with their jargon on).

### Tier 2 — the promised-but-missing math
4. **Primer: Graphs, coupling, and spectra** (planned card exists) — the
   gentlest possible eigenvector, the Fiedler idea as "where the bottlenecks
   are". Then fix foundations §03's dangling "Chapter 7 will unpack" promise
   to point at it.
5. **Primer: Gauge and holonomy — the arrow you carry around a loop**
   (planned card exists; Michael asked for gauge theory by name). The
   globe-walk: carry an arrow around a triangle, come back rotated by the
   enclosed curvature. §10 already teaches parallel transport in plain words
   without naming it — the primer names it, and §10 gets one sentence
   connecting the two. Sound idea: a drone whose pitch shifts by the
   accumulated holonomy angle when the loop closes.
6. **Expand PAS at first use** (§06), gloss coarse-graining in §13 (or grow
   it into the topology primer), gloss free energy in §05, unpack legality's
   four conditions in one aside in foundations §03.

### Tier 3 — more sound, more welcome
7. **Sonify the existing hero figures** where the physics earns it:
   foundations §04 (legality-off = white-noise flood drowning the chorus —
   the arrow reversal audible), §06 keystone (the critical band as the
   consonant zone between detuned chaos and dead unison), primer-00-style
   sound for alpha §01-prelude.
8. **A "if you read only three" skeptic's path** on the paradigm lander
   (§06 → §10 → §11) — carried over from the standing to-do list.
9. **Per-chapter "plain words" recap boxes** (imitate paradigm lander's
   on-ramp box; §17-coda shows the voice), starting with the fix-first list.

### Deliberately not doing
- Dumbing down the α essay: the audit confirms it serves a different,
  expert audience (its §11 is journal-supplement dense — that is its job).
  The library lander should just *say* who each essay is for.
- Auto-playing audio, or sound as decoration. Sound only where it carries
  the actual physics (beats, lock, noise floods) or a consistent labeled
  mapping (cos → loudness).

---

## 3b. Addendum (same day): the parallel build + extended backlog

Michael's direction: the primers should become **the central resource for
everything needed to understand the whole body of work** — written for him
as much as for anyone. Accordingly, five modules were commissioned in
parallel (one agent each, non-overlapping files):

- 02 topology & manifolds · 03 inside a transformer · 04 gauge & holonomy ·
  05 graphs & spectra (**eigenvalues** — explicitly requested) ·
  06 **the von Mises circle / where R₀ ≈ 0.303 comes from** (explicitly
  requested; new addition to the roadmap — the birth certificate of the
  site's central constant).

**Extended backlog** (in rough dependency order, for future sessions —
driven by what the corpus actually leans on):

- *Free energy and entropy, gently* — extends foundations §04's ink-in-water
  entropy to F = E − TS; unlocks §05-the-wall's vortex argument properly.
- *Coarse-graining and renormalization, in plain words* — "squint and the
  law survives"; unlocks §13 (its central unexplained mechanism) and the
  BKT/RG language of the α essay.
- *Singular values / SVD* — "every linear map is rotate–stretch–rotate";
  unlocks §08's d_eff formula and transformers §03 honestly.
- *The BKT transition itself, as a primer* — vortex pairs, binding/
  unbinding, why 2/π; the bridge from primer 06 to the α essay.
- *Complex numbers as arrows that rotate* — quietly needed by anything
  touching Fourier/phasors; cheap to write, wide unlock.
- *Legality's four conditions, unpacked* — less a primer than an aside box
  in foundations §03, but tracked here so it isn't lost.

## 3c. Second addendum: the TIER architecture (Michael's direction)

The track is now **"The curriculum"** — three tiers, each a tagged card on
the library lander (beginner green / intermediate blue / advanced purple),
each a labeled section with anchors on `primers/index.html`:

- **Beginner — "Hear it first"**: 00 phase & sound · the **sound workshop**
  (the multi-section pilot module: harmonograph — two phases locking as a
  drawing holding still — and Chladni plates; `primers/sound-workshop/`
  with its own lander + sections/) · 01 rotations.
- **Intermediate — "The language of structure"**: 02 topology & manifolds ·
  03 inside a transformer · 04 gauge & holonomy · 05 eigenvalues & spectra ·
  06 von Mises · 07 **rotation groups** (2D commutes → 3D doesn't → SO(D)
  as independent turntables).
- **Advanced — "The frontier's mathematics"**: 08 **Lohe dynamics on the
  hypersphere** (unit vectors, effective coupling, tokens as paths — the
  curriculum's summit, per Michael's ask) · planned: effective coupling &
  living K · the BKT transition itself · coarse-graining & RG.

Structural rule established: a module is one page unless its topic needs
more; multi-section modules are folders under `primers/` with their own
mini-TOC (the sound workshop is the pilot). Depth question answered by
tiers, not by inflating the ten-minute pages.

## 4. Craft rules for primers (so the track stays coherent)

- One idea per primer, ~10 minutes, zero assumed math; notation is the
  *last* step and always follows the felt version.
- Ears where the idea permits (phase, beats, lock, noise, holonomy-drone);
  hands everywhere (every figure draggable); the honest-flag discipline
  (purple callout) applies to primers exactly as to essays — every cartoon
  names the real measurement it is a cartoon of, with a link.
- Essays link *down* to primers at the exact word that needs them ("when a
  word stops you"), primers link *up* to the essay section that pays the
  idea off. Never require the primer track as homework.
- One audio language site-wide: pitch = frequency, loudness = agreement,
  noise = illegality/decoherence.
