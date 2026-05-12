# ACKNOWLEDGMENTS

This file documents external algorithm references used in the
inverse-telecine (`--vpp-ivtc`) and motion-adaptive deinterlace
(`--vpp-bwdif`) filter implementations.

QSVEnc is licensed under the MIT License. The references documented
below are GPL-2.0-or-later or LGPL-3.0 projects — these projects
informed the **algorithms and published mathematical constants** used
in the QSVEnc filter implementations, but **no source code was
copied** from any of them. Every kernel and host-side filter routine
listed below is from-scratch OpenCL (kernels) and C++ (host) code that
independently re-implements published, widely-shared algorithms.

The distinction observed throughout:

* **Published mathematical constants** (numeric facts) — used directly.
  Numeric data is not copyrightable, and the same constants appear in
  every independent implementation of the same published algorithm.
* **Published algorithm descriptions** (e.g. YADIF's three-tap
  temporal motion metric, BBC PH-2071's w3fdif filter formula,
  Decomb's 5-frame cadence-prediction state machine, DGDecode's RFF
  FrameList state machine) — implemented from scratch in OpenCL / C++,
  using QSVEnc's own data structures and control flow.
* **Source code from those projects** — NOT copied. The QSVEnc
  implementation is a from-scratch OpenCL implementation of the same
  published algorithms.

A full per-block code-comparison audit is preserved in
`analysis/license_audit/AUDIT-REPORT.md`.

---

## Referenced projects

### BBC Research Department Paper PH-2071

**"A Motion Adaptive De-interlacing Method"**, J. Weston, BBC Research
Department Report, 1989.

* **License of paper:** Crown copyright; the algorithm and its
  published numeric coefficients are widely treated as public-domain
  signal-processing technique.
* **What was referenced:** The Weston 3-Field Deinterlacer ("w3fdif")
  spatial-cubic, low-frequency, and high-frequency reconstruction
  filters, and their published 13-bit fixed-point coefficients.
* **Numeric constants used (published, not copyrightable):**

  | Symbol | Value | Role |
  |--------|-------|------|
  | `W3F_SHIFT` | 13 | Fixed-point shift (1<<13 = 8192) |
  | `W3F_SP0`   | 5077 | Spatial-only cubic, ±1 row |
  | `W3F_SP1`   |  981 | Spatial-only cubic, ±3 rows |
  | `W3F_LF0`   | 4309 | Low-frequency, ±1 row |
  | `W3F_LF1`   |  213 | Low-frequency, ±3 rows |
  | `W3F_HF0`   | 5570 | High-frequency, central |
  | `W3F_HF1`   | 3801 | High-frequency, ±2 |
  | `W3F_HF2`   | 1016 | High-frequency, ±4 |

  These are the published BBC PH-2071 coefficients and are shared
  across all independent w3fdif implementations.
* **Where referenced in our code:**
  * `QSVPipeline/rgy_filter_bwdif.cl` — `kernel_bwdif_frame`
  * `QSVPipeline/rgy_filter_ivtc.cl` — `kernel_ivtc_synthesize`,
    `kernel_ivtc_bwdif_deint`
* **Statement:** No source code was copied from any BBC publication.
  The QSVEnc implementation is a from-scratch OpenCL implementation
  that independently applies the published filter formulas with the
  published coefficients.

### YADIF (Yet Another Deinterlacing Filter)

* **Author:** Michael Niedermayer, 2006.
* **Original location:** `libavfilter/vf_yadif.c` in FFmpeg.
* **License:** GPL-2.0-or-later (FFmpeg).
* **What was referenced:** The published 3-tap temporal motion metric
  used to gate spatial-vs-temporal blending in motion-adaptive
  deinterlacers, and the spatial bound-tightening idea (refining the
  motion corridor using same-parity ±2 references). These techniques
  are widely adopted across deinterlacers and are documented in
  conference papers and online tutorials.
* **Where referenced in our code:**
  * `QSVPipeline/rgy_filter_bwdif.cl` — `kernel_bwdif_frame` (motion
    metric and corridor refinement, lines ~125-189)
  * `QSVPipeline/rgy_filter_ivtc.cl` — `kernel_ivtc_bwdif_deint`
    (motion metric and corridor refinement, lines ~571-635)
* **Statement:** No source code was copied from FFmpeg / YADIF. The
  QSVEnc OpenCL kernels independently re-implement the same published
  algorithm with different code structure (sequential if-update
  reductions and ternary expressions instead of YADIF's nested
  `FFMAX3`/`FFMIN3` macros), and use semantically named intermediate
  variables (`crossTimeFull`, `prevPairDeltaSum`, `predDriftAboveCur`,
  `upperBound`, `lowerBound`, `spreadMargin`) chosen for our codebase.

### BWDIF (Bob Weaver Deinterlacing Filter)

* **Author:** Westley Martinez and contributors. C++ Avisynth+ port at
  `https://github.com/Asd-g/AviSynth-BWDIF` (license: LGPL-3.0).
  Original ffmpeg implementation in `libavfilter/vf_bwdif.c`
  (license: GPL-2.0-or-later, ffmpeg).
* **License:** LGPL-3.0 for the Avisynth+ port; GPL-2.0+ for the
  ffmpeg implementation.
* **What was referenced:** The combination of YADIF's motion metric
  with PH-2071's w3fdif spatial reconstruction (the BWDIF concept of
  motion-adaptive w3fdif), the per-row-context dispatch (full-context
  vs spatial-bounds-only vs flat-edge), and the same-parity temporal
  reference convention.
* **Where referenced in our code:**
  * `QSVPipeline/rgy_filter_bwdif.h` (header comment),
    `QSVPipeline/rgy_filter_bwdif.cpp` (host-side filter pipeline),
    `QSVPipeline/rgy_filter_bwdif.cl` (kernel)
  * `QSVPipeline/rgy_filter_ivtc.cl` — `kernel_ivtc_bwdif_deint`
    (full BWDIF deinterlacer for D2V-flagged interlaced frames)
* **Statement:** No source code was copied from any BWDIF
  implementation. The QSVEnc kernels are from-scratch OpenCL with a
  GPU-native execution model (per-pixel work-items, no SIMD intrinsics,
  no per-row C++ templates). Variable names, expression form, and
  control flow were chosen for the OpenCL idiom and not for similarity
  to upstream C++.

### Decomb / Telecide

* **Author:** Donald A. Graft ("tritical"), 2003-present.
* **Original location:** AviSynth Decomb plugin, `Telecide.{cpp,h}`.
* **License:** GPL-2.0-or-later.
* **What was referenced:** The 5-frame cadence-prediction state
  machine for 3:2 pulldown detection (`PredictHardYUY2`); the
  `gthresh` percent-error gate for cadence-override validation; the
  `highest_sumc` block-MAX combing-metric concept (single hot block
  drives the frame-level signal); the `vmetric` post-assembly veto
  concept.
* **Where referenced in our code:**
  * `QSVPipeline/rgy_filter_ivtc.cpp::updateCadence` (around lines
    1013-1145) — independent re-implementation using a 2D pattern
    table and per-phase argmax, distinct from Decomb's hex-packed
    switch-case approach.
  * `QSVPipeline/rgy_filter_ivtc.cpp` cadence override block (around
    lines 2746-2830) — independent re-implementation using integer
    arithmetic instead of float, with QSVEnc-specific cadence-tag
    diagnostic encoding.
  * `QSVPipeline/rgy_filter_ivtc.cpp` block-MAX scoring helpers and
    `kernel_ivtc_score_candidates` block-combed flag (around lines
    1170-1273 and the OpenCL kernel) — independent host-side reduction
    plus a per-WG block flag scheme; no overlap with Decomb's YV12
    scalar-plus-SIMD C code.
* **Statement:** No source code was copied from Decomb. Comments in
  the QSVEnc source cite specific Decomb file:line locations purely
  to document where the algorithm being implemented can be cross-read
  in the original. The QSVEnc implementation uses different data
  structures (vector ring vs cache array), different control flow
  (argmax pattern scoring vs hex-switch dispatch), different numeric
  representation (uint64 integer vs double-precision float), and adds
  features unique to QSVEnc (cadence-tag TSV logging, alt-parity
  diagnostic, decoder-driven routing, D2V ground-truth integration).

### TIVTC (TFM + TDecimate)

* **Author:** Kevin Stone, 2004-2008; additional work by pinterf, 2020.
* **Original location:** AviSynth TIVTC plugin.
* **License:** GPL-2.0-or-later.
* **What was referenced:** Conceptual reference for the field-matching
  + decimation pipeline structure, and prior-art for the
  cadence-prediction patterns. Note that TIVTC was originally derived
  from Decomb, so the algorithms in `Telecide.{cpp,h}` and TIVTC's
  `TFM` overlap historically. Where our code refers to "Telecide.cpp"
  or "Telecide.h" file:line locations, the citation is to **Decomb's**
  Telecide files, not TIVTC's TFM.
* **Where referenced in our code:** General field-matcher concepts in
  `QSVPipeline/rgy_filter_ivtc.cpp`. After the 2026-05-09 attribution
  cleanup, in-source references that cite `Telecide.{cpp,h}` line
  numbers are now correctly attributed to "Decomb's Telecide" rather
  than "TFM".
* **Statement:** No source code was copied from TIVTC.

### DGDecode (DVD2AVI / MPEG2Dec3 / DGDecode lineage)

* **Authors:** Chia-chen Kuo (DVD2AVI, 2001), Mathias Born (C++ port,
  2001), MarcFD (YV12 / MPEG2Dec3 modifications), and DGDecode
  maintainers.
* **Original location:** `vfapidec.cpp` in the DGDecode source.
* **License:** GPL-2.0-or-later.
* **What was referenced:** The RFF (repeat-first-field) FrameList
  state machine that converts coded-order MPEG-2 frames with
  per-picture RFF/TFF flags into a display-order schedule. This is the
  standard reference implementation of the published MPEG-2 RFF
  expansion algorithm and is what AviSynth users have been working
  with for two decades.
* **Where referenced in our code:**
  * `QSVPipeline/rgy_filter_ivtc.cpp::buildScheduleFromScan` (around
    lines 1683-1780) — independent re-implementation using a
    look-back lazy-completion pattern (next iteration completes
    pending half-open slot) instead of DGDecode's look-ahead
    eager-open pattern (current iteration pre-opens slot[n+1]). Uses
    `std::vector<IvtcDisplayFrame>` push-back instead of an indexed
    `FrameList[]`.
  * `QSVPipeline/rgy_filter_ivtc.cpp::overlayField` and
    `QSVPipeline/rgy_filter_ivtc.cl::kernel_ivtc_field_overlay`
    (around lines 1409-1432 in the .cpp; lines 39-54 in the .cl) —
    independent GPU-native re-implementation. DGDecode's CPU
    `BitBlt` with stride-doubled pitch is replaced by an OpenCL
    kernel that tests `(iy & 1) == targetParity` per pixel.
* **Statement:** No source code was copied from DGDecode. The
  state-machine semantics are deliberately matched (because they
  define the published RFF expansion algorithm), but the C++ code is
  rewritten with different data structures, different control flow,
  and a different field-overlay execution model.

---

## Filter coefficients

The numeric constants `{4309, 213, 5570, 3801, 1016, 5077, 981}` and
shift `13` that appear in `kernel_bwdif_frame` and
`kernel_ivtc_bwdif_deint` are the published BBC PH-2071 fixed-point
coefficients. They are numeric data — facts of the published algorithm
— and are not copyrightable. The same values appear in every
independent BWDIF / w3fdif implementation.

## Summary

| Reference | License | Code copied? | Algorithm referenced? |
|-----------|---------|:------------:|:---------------------:|
| BBC PH-2071 (Weston, 1989) | Crown copyright; published math | No | Yes (filter formula, coefficients) |
| YADIF (Niedermayer, 2006) | GPL-2.0+ (in FFmpeg) | No | Yes (motion metric, spatial bound tightening) |
| BWDIF (Avisynth+ port) | LGPL-3.0 | No | Yes (motion-adaptive w3fdif concept) |
| BWDIF (FFmpeg) | GPL-2.0+ | No | Yes (motion-adaptive w3fdif concept) |
| Decomb / Telecide | GPL-2.0+ | No | Yes (cadence prediction, gthresh gate, highest_sumc, vmetric) |
| TIVTC (Stone / pinterf) | GPL-2.0+ | No | Conceptual prior art only |
| DGDecode | GPL-2.0+ | No | Yes (RFF FrameList state machine) |

QSVEnc remains MIT-licensed. None of the above references contributed
copyrightable source code expression to QSVEnc.

A full per-block license audit, including the rewritten BWDIF/YADIF
kernel code that replaced earlier paraphrase-class similarities, is
preserved in `analysis/license_audit/AUDIT-REPORT.md`.
