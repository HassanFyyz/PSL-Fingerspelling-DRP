/**
 * Causal SOFI Algorithm 2 — JavaScript port of sofi_streaming.py.
 *
 * Runs in the browser on MediaPipe keypoints so no video ever leaves the
 * client. Verified frame-for-frame against the Python implementation on real
 * PSL-CFRT clips (see web/test_parity.mjs).
 *
 * Parameters are DURATIONS, not frame counts. The offline algorithm was tuned
 * on 50 fps dataset video; a webcam is usually 30 fps. Reusing the tuned frame
 * counts directly degrades segment-count MAE from ~1.2 to ~2.6 at 30 fps and
 * ~4.6 at 15 fps. Converting durations at runtime holds MAE at ~1.2 across
 * 50/30/25 fps. This is measured in streaming_segmentation.ipynb.
 */

export const TUNED_FPS = 50.0;
export const SMOOTH_SEC = 13 / TUNED_FPS;   // 0.26 s
export const RUN_SEC = 7 / TUNED_FPS;       // 0.14 s
export const MIN_SEP_SEC = 15 / TUNED_FPS;  // 0.30 s
export const SEG_LEN_SEC = 11 / TUNED_FPS;  // 0.22 s

// Below this the duration->frame conversion degenerates: at 5 fps it yields
// smooth=1, run=1, so nearly every frame is a "local minimum". A real session
// at 5 fps produced 47 spurious holds.
export const MIN_VIABLE_FPS = 14.0;

// Hard floors so a misconfigured instance can never become degenerate.
const MIN_SMOOTH = 3, MIN_RUN = 2, MIN_SEP = 3;

// Dropouts shorter than this are MediaPipe blinking while the hand is still
// there; longer means the hand genuinely left frame.
export const GAP_TOLERANCE_SEC = 0.20;

// Peak OFI in the window over OFI at the minimum. A hold is a dip *between*
// transitions, so the surrounding motion must rise above it. Live-only:
// dataset clips are pre-trimmed to actual signing, a live hand can sit idle.
export const MIN_CONTRAST = 1.10;

// Live-only hold gates, in physical units: palm lengths (wrist to middle-finger
// knuckle) and seconds. OFI is displacement *per frame* in screen units, so its
// jitter floor does not scale with fps or with distance to the camera while real
// motion does -- at 60 fps a still hand's jitter dips pass MIN_CONTRAST every
// few hundred ms. These gates are measured in units that do not move with fps.
// Off by default (null) so the class still matches sofi_streaming.py exactly;
// the live page turns them on.
//
// STILL_SPEED: mean hand speed at the hold (+/- SPEED_WINDOW_SEC/2) must be below
//   this, so a slow-down while still forming the handshape is not taken as a hold.
// MOVE_DIST:   since the last still pose, the hand must have departed from it by
//   at least this much, sustained for SPEED_WINDOW_SEC, so one sustained pose
//   gives one hold. Departure is the mean of the 3 landmarks that moved most:
//   letters differ in a few fingers, and a mean over all 21 dilutes that (the
//   wrist and knuckles barely move). Translation counts, so a re-articulated
//   doubled letter still gives two holds.
// Starting values, not tuned on data: the page shows the live speed and why
// each candidate was dropped, and accepts ?still= / ?move= to adjust them.
// MIN_STILL_SEC: a stretch this long below STILL_SPEED is itself a hold
//   candidate. Needed because a truly still hand gives a FLAT, noisy OFI, and
//   whether Algorithm 2 finds a local minimum inside it is down to the noise:
//   several (the duplicates) or none (a missed letter). Algorithm 2 minima still
//   catch letters signed without a full stop; MOVE_DIST dedups the two sources.
export const STILL_SPEED = 0.6;       // palm lengths per second
export const MOVE_DIST = 0.35;        // palm lengths
export const MIN_STILL_SEC = 0.25;
export const SPEED_WINDOW_SEC = 0.10; // speed is measured across this interval

function framesFor(seconds, fps, odd = false, minimum = 1) {
  let n = Math.max(minimum, Math.round(seconds * fps));
  if (odd && n % 2 === 0) n += 1;
  return n;
}

function mean(arr, lo, hi) {
  let s = 0;
  for (let i = lo; i < hi; i++) s += arr[i];
  return s / (hi - lo);
}

/** Mean per-landmark distance between two interleaved [x0,y0,x1,y1,...] poses. */
function poseDist(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i += 2) s += Math.hypot(a[i] - b[i], a[i + 1] - b[i + 1]);
  return s / (a.length / 2);
}

/** Mean distance of the 3 landmarks that differ most between two poses. */
function poseChange(a, b) {
  let d1 = 0, d2 = 0, d3 = 0;
  for (let i = 0; i < a.length; i += 2) {
    const d = Math.hypot(a[i] - b[i], a[i + 1] - b[i + 1]);
    if (d > d1) { d3 = d2; d2 = d1; d1 = d; }
    else if (d > d2) { d3 = d2; d2 = d; }
    else if (d > d3) d3 = d;
  }
  return (d1 + d2 + d3) / 3;
}

// Pose history older than this is dropped; only the stretch since the last
// still pose is ever looked at.
const POSE_HISTORY_SEC = 30;

export class StreamingSegmenter {
  /**
   * @param {object} opts
   * @param {number} opts.fps        camera framerate
   * @param {number} [opts.smoothSec] centred smoothing window, seconds
   * @param {number} [opts.runSec]    minimum-window half-width, seconds
   * @param {number} [opts.minSepSec] minimum separation between holds, seconds
   * @param {number} [opts.segLenSec] emitted hold span, seconds
   * @param {number} [opts.aspect]    frame width / height, so x and y share units
   * @param {?number} [opts.stillSpeed] palm lengths/s; null disables the gate
   * @param {?number} [opts.moveDist]   palm lengths; null disables the gate
   */
  constructor({ fps = TUNED_FPS, smoothSec = SMOOTH_SEC, runSec = RUN_SEC,
                minSepSec = MIN_SEP_SEC, segLenSec = SEG_LEN_SEC,
                requireHand = true, gapToleranceSec = GAP_TOLERANCE_SEC,
                minContrast = MIN_CONTRAST, aspect = 1, stillSpeed = null,
                moveDist = null, speedWindowSec = SPEED_WINDOW_SEC,
                minStillSec = MIN_STILL_SEC } = {}) {
    this.fps = fps;
    this.smoothWin = Math.max(MIN_SMOOTH, framesFor(smoothSec, fps, true));
    if (this.smoothWin % 2 === 0) this.smoothWin += 1;
    this.run = Math.max(MIN_RUN, framesFor(runSec, fps));
    this.minSep = Math.max(MIN_SEP, framesFor(minSepSec, fps));
    this.segLen = framesFor(segLenSec, fps, true);
    this.half = (this.smoothWin - 1) >> 1;
    this.delay = Math.max(this.half, this.run, this.minSep);

    this.requireHand = requireHand;
    this.gapTolerance = Math.max(1, framesFor(gapToleranceSec, fps));
    this.minContrast = minContrast;
    /** False when the camera is too slow for the method to behave sensibly. */
    this.viable = fps >= MIN_VIABLE_FPS;

    this.raw = [];
    this.smooth = [];
    this.present = [];
    this.prev = null;
    this.n = 0;
    this.pending = [];
    this.lastEmitted = -1e9;
    this.scannedTo = -1;

    this.aspect = aspect;
    this.stillSpeed = stillSpeed;
    this.moveDist = moveDist;
    this.speedHalf = Math.max(1, framesFor(speedWindowSec / 2, fps));
    this.historyFrames = framesFor(POSE_HISTORY_SEC, fps);
    this.pose = [];         // per frame: Float64Array [x0,y0,...] or null if no hand
    this.palm = [];         // per frame: wrist -> middle knuckle length, same units
    this.speed = [];        // centred hand speed, palm lengths/s; NaN if unknown
    this.anchor = null;     // {frame, pose, palm} of the last still pose seen
    this.dropped = { moving: 0, noMove: 0 };
    // never shorter than the emitted span, so the whole span is inside the stillness
    this.minStill = Math.max(this.segLen, framesFor(minStillSec, fps));
    this.stillRun = 0;      // consecutive frames below stillSpeed
    this.stillFired = false;
    this.stillOut = [];     // holds found by stillness, returned by the next push()
  }

  /** Inherent latency of the method, in milliseconds. */
  get latencyMs() { return 1000 * this.delay / this.fps; }

  /** Most recent hand speed in palm lengths per second, or NaN. */
  get speedNow() {
    for (let i = this.speed.length - 1; i >= 0 && i >= this.speed.length - 5; i--) {
      if (Number.isFinite(this.speed[i])) return this.speed[i];
    }
    return NaN;
  }

  /**
   * Feed one frame.
   * @param {Float64Array|number[]|null} kx 21 x-coords, or null if no hand
   * @param {Float64Array|number[]|null} ky 21 y-coords, or null if no hand
   * @returns {Array<{frame:number, ofi:number, start:number, end:number}>}
   */
  push(kx, ky) {
    this._pushOfi(kx, ky);
    this._updateSmooth();
    this._scanMinima();
    return this._collect(this._emitReady(false));
  }

  /** Merge stillness holds with Algorithm 2 holds, in temporal order. */
  _collect(out) {
    if (!this.stillOut.length) return out;
    return this.stillOut.splice(0).concat(out).sort((a, b) => a.frame - b.frame);
  }

  /** End of stream: flush anything still pending. */
  finish() {
    this._updateSmooth();
    while (this.smooth.length < this.n) {
      const i = this.smooth.length;
      this.smooth.push(mean(this.raw, Math.max(0, i - this.half), this.n));
    }
    this._scanMinima();
    return this._collect(this._emitReady(true));
  }

  _pushOfi(kx, ky) {
    this.present.push(kx != null && ky != null);
    if (kx == null || ky == null) {
      // MediaPipe lost the hand. We cannot interpolate forward in a live
      // stream, so carry the last pose. Marked absent so _blocked() can
      // reject minima that sit inside a sustained gap.
      this.raw.push(0.0);
    } else if (this.prev === null) {
      this.raw.push(0.0);
      this.prev = [Float64Array.from(kx), Float64Array.from(ky)];
    } else {
      const [px, py] = this.prev;
      let s = 0;
      for (let i = 0; i < kx.length; i++) {
        const dx = kx[i] - px[i], dy = ky[i] - py[i];
        s += Math.sqrt(dx * dx + dy * dy);
      }
      this.raw.push(s / kx.length);
      this.prev = [Float64Array.from(kx), Float64Array.from(ky)];
    }
    this.n += 1;
    if (this.stillSpeed != null || this.moveDist != null) this._pushPose(kx, ky);
  }

  _pushPose(kx, ky) {
    if (kx == null || ky == null) {
      this.pose.push(null);
      this.palm.push(0);
    } else {
      const p = new Float64Array(kx.length * 2);
      for (let i = 0; i < kx.length; i++) { p[2 * i] = kx[i] * this.aspect; p[2 * i + 1] = ky[i]; }
      this.pose.push(p);
      this.palm.push(Math.hypot(p[18] - p[0], p[19] - p[1]));   // landmark 9 - landmark 0
    }
    const old = this.pose.length - 1 - this.historyFrames;
    if (old >= 0) this.pose[old] = null;

    const h = this.speedHalf;
    while (this.speed.length <= this.pose.length - 1 - h) {
      const c = this.speed.length;
      const a = c - h >= 0 ? this.pose[c - h] : null, b = this.pose[c + h];
      const palm = (this.palm[c - h] + this.palm[c + h]) / 2;
      const v = a && b && palm > 1e-6 ? poseDist(a, b) / palm / (2 * h / this.fps) : NaN;
      this.speed.push(v);
      if (this.stillSpeed != null) this._trackStill(c, v);
    }
  }

  /** One candidate per stretch of stillness, centred on its first emitted span. */
  _trackStill(c, v) {
    if (!(v <= this.stillSpeed)) { this.stillRun = 0; this.stillFired = false; return; }
    this.stillRun++;
    if (this.stillFired) {
      // This stillness already had its candidate. Keep the anchor on the
      // current pose so slow arm sag through a long hold never adds up to a
      // "movement" -- only motion since the hand last stopped counts.
      if (this.anchor && this.moveDist != null && this.stillRun % this.minStill === 0) {
        const here = this._meanPose(c - this.minStill + 1, c);
        if (here) this.anchor = { frame: c, pose: here.pose, palm: here.palm };
      }
      return;
    }
    if (this.stillRun < this.minStill) return;
    this.stillFired = true;
    const f = c - this.stillRun + 1 + (this.segLen >> 1);
    this._tryEmit(f, NaN, this.stillOut);
  }

  /** minSep, then the live gates; pushes the hold onto `out` if it survives. */
  _tryEmit(frame, ofi, out) {
    if (frame - this.lastEmitted <= this.minSep) return;
    const info = {};
    const why = (this.stillSpeed != null || this.moveDist != null)
      ? this._gate(frame, info) : null;
    if (why) { this.dropped[why]++; return; }
    this.lastEmitted = frame;
    const h = this.segLen >> 1;
    out.push({ frame, ofi, start: Math.max(0, frame - h), end: frame + h, ...info });
  }

  /** Mean pose and palm length over frames [lo, hi], skipping absent frames. */
  _meanPose(lo, hi) {
    let acc = null, palm = 0, k = 0;
    for (let j = Math.max(0, lo); j <= hi && j < this.pose.length; j++) {
      const p = this.pose[j];
      if (!p) continue;
      if (!acc) acc = new Float64Array(p.length);
      for (let i = 0; i < p.length; i++) acc[i] += p[i];
      palm += this.palm[j]; k++;
    }
    if (!k) return null;
    for (let i = 0; i < acc.length; i++) acc[i] /= k;
    return { pose: acc, palm: palm / k };
  }

  /**
   * Live-only gates on a confirmed minimum. Returns null if it may be emitted,
   * else the reason it was dropped ("moving" or "noMove").
   */
  _gate(f, info) {
    const h = this.segLen >> 1;
    if (this.stillSpeed != null) {
      let s = 0, k = 0;
      for (let j = f - this.speedHalf; j <= f + this.speedHalf; j++) {
        const v = this.speed[j];
        if (Number.isFinite(v)) { s += v; k++; }
      }
      info.speed = k ? s / k : NaN;
      if (k && info.speed > this.stillSpeed) return "moving";
    }
    if (this.moveDist == null) return null;

    const here = this._meanPose(f - h, f + h);
    if (!here) return null;
    let moved = Infinity;
    if (this.anchor) {
      // furthest the hand got from the last still pose, held for 2*speedHalf+1
      // frames, so a one-frame tracking glitch cannot count as movement
      moved = 0;
      const a = this.anchor, need = 2 * this.speedHalf + 1, win = [];
      for (let j = a.frame + 1; j <= f + h && j < this.pose.length; j++) {
        win.push(this.pose[j] ? poseChange(this.pose[j], a.pose) / a.palm : 0);
        if (win.length > need) win.shift();
        if (win.length === need) moved = Math.max(moved, Math.min(...win));
      }
    }
    info.moved = moved;
    // Re-anchor on every still pose, emitted or not, so slow drift through a
    // long hold never accumulates into a fake "movement".
    this.anchor = { frame: f + h, pose: here.pose, palm: here.palm };
    return moved < this.moveDist ? "noMove" : null;
  }

  _updateSmooth() {
    const target = this.n - 1 - this.half;
    while (this.smooth.length <= target) {
      const i = this.smooth.length;
      this.smooth.push(mean(this.raw, Math.max(0, i - this.half),
                            Math.min(this.n, i + this.half + 1)));
    }
  }

  /** True if the +/-run window around i touches a sustained absence. */
  _blocked(i) {
    const lo = Math.max(0, i - this.run);
    const hi = Math.min(this.present.length, i + this.run + 1);
    let j = lo;
    while (j < hi) {
      if (this.present[j]) { j++; continue; }
      let start = j;
      while (start > 0 && !this.present[start - 1]) start--;
      let end = j;
      while (end + 1 < this.present.length && !this.present[end + 1]) end++;
      if (end - start + 1 >= this.gapTolerance) return true;
      j = end + 1;
    }
    return false;
  }

  _scanMinima() {
    const s = this.smooth;
    const limit = s.length - this.run - 1;
    let i = Math.max(this.scannedTo + 1, this.run);
    for (; i <= limit; i++) {
      const c = s[i];
      let isMin = true, peak = -Infinity;
      for (let j = i - this.run; j <= i + this.run; j++) {
        if (s[j] < c - 1e-12) { isMin = false; break; }
        if (s[j] > peak) peak = s[j];
      }
      if (isMin && s[i - this.run] > c && s[i + this.run] > c &&
          peak >= this.minContrast * (c + 1e-12) &&
          (!this.requireHand || !this._blocked(i))) {
        this.pending.push([i, c]);
      }
      this.scannedTo = i;
    }
  }

  /**
   * Mirrors collapse_minima: chain candidates within minSep, keep the deepest.
   * A chain is only safe to resolve once no later candidate could still join.
   */
  _emitReady(flush) {
    const out = [];
    while (this.pending.length) {
      const [frame] = this.pending[0];
      if (!flush && this.scannedTo < frame + this.minSep) break;

      const chain = [this.pending.shift()];
      let stalled = false;
      while (this.pending.length &&
             this.pending[0][0] - chain[chain.length - 1][0] <= this.minSep) {
        if (!flush && this.scannedTo < this.pending[0][0] + this.minSep) {
          // chain may still grow — put everything back and wait
          while (chain.length) this.pending.unshift(chain.pop());
          stalled = true;
          break;
        }
        chain.push(this.pending.shift());
      }
      if (stalled) return out;

      let best = chain[0];
      for (const c of chain) if (c[1] < best[1]) best = c;
      this._tryEmit(best[0], best[1], out);
    }
    return out;
  }
}
