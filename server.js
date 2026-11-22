'use strict';
/**
 * Pencil sketch server. Upload a photo -> hand-drawn pencil-on-paper version.
 * One file, two dependencies (express, sharp). Light enough for Render's free plan.
 *
 * Look: graphite linework (dodge blend at two scales), wobbly cross-hatching that gets denser in shadows,
 * soft smudged shading, paper grain + fibers + vignette. Optional colored-pencil mode.
 * Env (optional): PORT, MAX_DIM (default 1200)
 */
const express = require('express');
const sharp = require('sharp');

const PORT = process.env.PORT || 3000;
const MAX_DIM = Number(process.env.MAX_DIM) || 1200;

const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
const smooth = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
function rng(seed) { // small seeded PRNG so the same photo always gives the same paper/strokes
  let a = seed >>> 0;
  return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

async function blur1(buf, w, h, sigma) {
  return sharp(buf, { raw: { width: w, height: h, channels: 1 } }).blur(Math.max(0.3, sigma)).raw().toBuffer();
}
// smooth low-frequency noise (used to make strokes wobble like a human hand)
async function lowNoise(w, h, cell, rand) {
  const sw = Math.max(2, Math.ceil(w / cell)), sh = Math.max(2, Math.ceil(h / cell));
  const b = Buffer.alloc(sw * sh);
  for (let i = 0; i < b.length; i++) b[i] = (rand() * 255) | 0;
  return sharp(b, { raw: { width: sw, height: sh, channels: 1 } }).resize(w, h, { fit: 'fill', kernel: 'cubic' }).raw().toBuffer();
}

async function pencil(buffer, colored) {
  const { data: rgb, info } = await sharp(buffer, { failOn: 'none' }).rotate()
    .resize({ width: MAX_DIM, height: MAX_DIM, fit: 'inside', withoutEnlargement: true })
    .flatten({ background: '#ffffff' }).toColourspace('srgb').removeAlpha().raw().toBuffer({ resolveWithObject: true });
  const w = info.width, h = info.height, n = w * h, side = Math.max(w, h);
  const rand = rng(0x5eed + n);

  // 1. grayscale with local contrast (CLAHE) so faces in shadow keep their features
  const gi = await sharp(rgb, { raw: { width: w, height: h, channels: 3 } }).greyscale()
    .clahe({ width: Math.max(16, Math.round(w / 6)), height: Math.max(16, Math.round(h / 6)), maxSlope: 3 })
    .normalise({ lower: 1, upper: 99 }).raw().toBuffer({ resolveWithObject: true });
  const g = gi.info.channels === 1 ? gi.data : Buffer.from(Array.from({ length: n }, (_, i) => gi.data[i * gi.info.channels]));

  // 2. line art: color-dodge of gray against blurred inverse (fine lines + softer broad lines)
  const inv = Buffer.alloc(n);
  for (let i = 0; i < n; i++) inv[i] = 255 - g[i];
  const [bFine, bSoft] = await Promise.all([blur1(inv, w, h, side / 150), blur1(inv, w, h, side / 45)]);

  // 3. noise fields
  const [j1, j2, j3, mottle] = await Promise.all([
    lowNoise(w, h, side / 14, rand), lowNoise(w, h, side / 14, rand), lowNoise(w, h, side / 14, rand), lowNoise(w, h, side / 5, rand),
  ]);
  const gr = Buffer.alloc(n);
  for (let i = 0; i < n; i++) gr[i] = (rand() * 255) | 0;
  const grain = await blur1(gr, w, h, 0.75);

  const period = Math.max(3.5, side / 150);
  const A = [Math.cos(0.80), Math.sin(0.80)], B = [Math.cos(-0.62), Math.sin(-0.62)], C = [Math.cos(1.25), Math.sin(1.25)];
  const out = Buffer.alloc(n * 3);

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x, o = i * 3;
      const gv = g[i] / 255, tone = 1 - gv;
      const gn = (grain[i] - 128) / 45; // ~[-1.2, 1.2]

      // linework
      const skF = clamp(gv / (1 - bFine[i] / 255 + 1e-3), 0, 1);
      const skS = clamp(gv / (1 - bSoft[i] / 255 + 1e-3), 0, 1);
      const line = clamp(Math.pow(1 - skF, 0.85) * 1.1 + (1 - skS) * 0.25, 0, 1) * (1 - 0.6 * smooth(0.7, 0.95, tone)); // deep shadows: hatching, not noisy lines

      // hatching: density follows tone; lines wobble with smooth noise
      const t2 = Math.pow(tone, 1.2);
      const wA = smooth(0.25, 0.7, t2) * 0.72, wB = smooth(0.55, 0.92, t2) * 0.72;
      const uA = (x * A[0] + y * A[1]) / period + ((j1[i] - 128) / 128) * 0.9;
      const uB = (x * B[0] + y * B[1]) / period + ((j2[i] - 128) / 128) * 0.9;
      const dA = Math.abs(uA - Math.floor(uA) - 0.5) * 2, dB = Math.abs(uB - Math.floor(uB) - 0.5) * 2;
      const strokeGrain = clamp(0.88 + 0.2 * gn, 0.55, 1);
      const hA = clamp((wA - dA) / 0.3, 0, 1) * strokeGrain, hB = clamp((wB - dB) / 0.3, 0, 1) * strokeGrain;

      // soft smudged graphite
      const smudge = Math.pow(bSoft[i] / 255, 1.6) * 0.2;

      let ink = 1 - (1 - line * 0.95) * (1 - hA * 0.5) * (1 - hB * 0.55) * (1 - smudge);
      ink = clamp(ink * clamp(0.9 + 0.18 * gn, 0.6, 1.05), 0, 1);

      // paper
      const dx = x / w - 0.5, dy = y / h - 0.5;
      const vig = 1 - 0.1 * smooth(0.35, 0.8, Math.hypot(dx, dy) * 1.4);
      const pm = (1 + 0.022 * gn + 0.02 * ((mottle[i] - 128) / 128)) * vig;
      let pr = 248 * pm, pg = 244 * pm, pb = 236 * pm;

      if (colored) { // light colored-pencil layer under the graphite, hatched at a third angle
        const uC = (x * C[0] + y * C[1]) / (period * 1.1) + ((j3[i] - 128) / 128) * 0.9;
        const dC = Math.abs(uC - Math.floor(uC) - 0.5) * 2;
        const cov = clamp((0.92 - dC) / 0.3, 0, 1) * clamp(0.75 + 0.3 * gn, 0.4, 1);
        const a = clamp(0.08 + 0.34 * cov, 0, 0.45) * (0.6 + 0.4 * Math.min(1, tone * 1.6 + 0.2));
        const cr = Math.min(255, rgb[o] * 1.06), cg = Math.min(255, rgb[o + 1] * 1.06), cb = Math.min(255, rgb[o + 2] * 1.06);
        pr = pr * (1 - a) + cr * a * pm; pg = pg * (1 - a) + cg * a * pm; pb = pb * (1 - a) + cb * a * pm;
      }

      out[o] = clamp(Math.round(pr * (1 - ink) + 46 * ink), 0, 255);
      out[o + 1] = clamp(Math.round(pg * (1 - ink) + 46 * ink), 0, 255);
      out[o + 2] = clamp(Math.round(pb * (1 - ink) + 52 * ink), 0, 255);
    }
  }
  return sharp(out, { raw: { width: w, height: h, channels: 3 } }).jpeg({ quality: 90, mozjpeg: true }).toBuffer();
}

// ---------------- server ----------------
const app = express();
app.get('/health', (_q, res) => res.json({ ok: true }));
app.post('/pencil', express.raw({ type: '*/*', limit: '25mb' }), async (req, res) => {
  if (!Buffer.isBuffer(req.body) || !req.body.length) return res.status(400).send('Send an image as the request body');
  try {
    const t = Date.now();
    const jpg = await pencil(req.body, req.query.color === '1');
    res.set('X-Ms', String(Date.now() - t)).type('jpeg').send(jpg);
  } catch (e) {
    console.error(e);
    res.status(422).send('Could not process that image: ' + e.message);
  }
});
app.get('/', (_q, res) => res.type('html').send(PAGE));
app.listen(PORT, () => console.log('Pencil sketch on :' + PORT));

const PAGE = `<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Pencil Sketch</title>
<style>
:root{color-scheme:light dark;font-family:system-ui,sans-serif}
body{max-width:960px;margin:0 auto;padding:20px 16px}
input,button{font:inherit;padding:10px;border-radius:8px;border:1px solid #8885}
button{background:#444;color:#fff;border:0;font-weight:600;cursor:pointer}button:disabled{opacity:.5}
label{margin:0 10px;white-space:nowrap}
.row{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:14px}
@media(max-width:640px){.row{grid-template-columns:1fr}}
img{width:100%;border-radius:6px}figcaption{font-size:.85rem;opacity:.7}figure{margin:0}
</style></head><body>
<h1>Pencil Sketch</h1>
<input id="file" type="file" accept="image/*"> <label><input id="color" type="checkbox"> Colored pencil</label>
<button id="go">Draw it</button><p id="st"></p>
<div class="row"><figure><img id="a" hidden><figcaption>Original</figcaption></figure>
<figure><img id="b" hidden><figcaption>Sketch</figcaption><a id="dl" hidden download="sketch.jpg">Download</a></figure></div>
<script>
var $=function(i){return document.getElementById(i)};
$('file').onchange=function(){var f=this.files[0];if(f){$('a').src=URL.createObjectURL(f);$('a').hidden=false}};
$('go').onclick=function(){
  var f=$('file').files[0]; if(!f) return $('st').textContent='Choose a photo first';
  $('go').disabled=true; $('st').textContent='Drawing...';
  fetch('/pencil'+($('color').checked?'?color=1':''),{method:'POST',body:f}).then(function(r){
    if(!r.ok) return r.text().then(function(t){throw new Error(t)});
    return r.blob().then(function(b){var u=URL.createObjectURL(b);$('b').src=u;$('b').hidden=false;$('dl').href=u;$('dl').hidden=false;$('st').textContent=''});
  }).catch(function(e){$('st').textContent='Error: '+e.message}).finally(function(){$('go').disabled=false});
};
</script></body></html>`;
