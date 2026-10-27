/**
 * Direct port of hand_geometry.py. Must match it exactly -- see test_svm.mjs.
 *
 * Features are invariant to uniform scale and translation of the input
 * coordinates (angles are scale-free; distances are normalised by palm span),
 * so x/y/z only need to be ISOTROPIC relative to each other, not any
 * particular unit. index.html passes [x*videoWidth, y*videoHeight, z*videoWidth].
 */

const FINGERS = [
  [0, 1, 2, 3, 4],
  [0, 5, 6, 7, 8],
  [0, 9, 10, 11, 12],
  [0, 13, 14, 15, 16],
  [0, 17, 18, 19, 20],
];
const TIPS = [4, 8, 12, 16, 20];
const MCPS = [1, 5, 9, 13, 17];
const DIST_POINTS = [0, 4, 8, 12, 16, 20, 2, 5, 9, 13, 17];

function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function norm(v) { return Math.hypot(v[0], v[1], v[2]); }
function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

function angleAt(a, b, c) {
  const v1 = sub(a, b), v2 = sub(c, b);
  const n1 = norm(v1), n2 = norm(v2);
  const cos = dot(v1, v2) / Math.max(n1 * n2, 1e-8);
  return Math.acos(Math.max(-1, Math.min(1, cos)));
}

/** pts: array of 21 [x,y,z]. Returns 19 angles: 15 flexion + 4 spread. */
function jointAngles(pts) {
  const out = [];
  for (const f of FINGERS) {
    for (let i = 0; i < f.length - 2; i++) {
      out.push(angleAt(pts[f[i]], pts[f[i + 1]], pts[f[i + 2]]));
    }
  }
  for (let i = 0; i < 4; i++) {
    const v1 = sub(pts[TIPS[i]], pts[MCPS[i]]);
    const v2 = sub(pts[TIPS[i + 1]], pts[MCPS[i + 1]]);
    const n1 = norm(v1), n2 = norm(v2);
    const cos = dot(v1, v2) / Math.max(n1 * n2, 1e-8);
    out.push(Math.acos(Math.max(-1, Math.min(1, cos))));
  }
  return out;
}

/** pts: array of 21 [x,y,z]. Returns 55 palm-normalised pairwise distances. */
function normalisedDistances(pts) {
  const scale = Math.max(norm(sub(pts[9], pts[0])), 1e-8);
  const out = [];
  for (let i = 0; i < DIST_POINTS.length; i++) {
    for (let j = i + 1; j < DIST_POINTS.length; j++) {
      out.push(norm(sub(pts[DIST_POINTS[i]], pts[DIST_POINTS[j]])) / scale);
    }
  }
  return out;
}

/** pts: array of 21 [x,y,z] isotropic landmarks for one frame. Returns 74 features. */
export function handFeatures(pts) {
  return jointAngles(pts).concat(normalisedDistances(pts));
}
