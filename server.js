'use strict';
/**
 * Toonify (single-file server). Upload a photo -> toon version.
 *
 * HYBRID: if TOON_ENGINE_URL points at a running toon_engine.py (GPU machine), the photo is
 * redrawn as an anime character by AI. If that is not set, offline, or fails, it automatically
 * falls back to a built-in comic filter that runs right here (no GPU, no APIs).
 *
 * Env (all optional): PORT, MAX_DIM (default 1024), TOON_ENGINE_URL
 */
const express = require('express');
const sharp = require('sharp');

const PORT = process.env.PORT || 3000;
const MAX_DIM = Number(process.env.MAX_DIM) || 1024;
const ENGINE = (process.env.TOON_ENGINE_URL || '').replace(/\/$/, '');

// ---------------- built-in comic filter (fallback) ----------------
const P = { bands: 5, sat: 1.25, lines: 0.55, hatch: 0.8, halftone: 0.6 };
const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
const smooth = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };

async function inkLines(gray, w, h, side) {
  const g = await sharp(gray, { raw: { width: w, height: h, channels: 1 } }).blur(Math.max(0.8, side / 900)).raw().toBuffer();
  const mag = new Float32Array(w * h);
  let max = 0;
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    const i = y * w + x;
    const gx = g[i - w + 1] + 2 * g[i + 1] + g[i + w + 1] - g[i - w - 1] - 2 * g[i - 1] - g[i + w - 1];
    const gy = g[i + w - 1] + 2 * g[i + w] + g[i + w + 1] - g[i - w - 1] - 2 * g[i - w] - g[i - w + 1];
    const m = Math.sqrt(gx * gx + gy * gy);
    mag[i] = m; if (m > max) max = m;
  }
  if (!max) return new Uint8Array(w * h);
  const bins = 2048, hist = new Uint32Array(bins), sc = (bins - 1) / max;
  for (let i = 0; i < mag.length; i++) hist[(mag[i] * sc) | 0]++;
  const target = (0.97 - 0.12 * P.lines) * mag.length;
  let acc = 0, t = max;
  for (let b = 0; b < bins; b++) { acc += hist[b]; if (acc >= target) { t = b / sc; break; } }
  t = Math.max(t, 30);
  const mask = Buffer.alloc(w * h);
  for (let i = 0; i < mag.length; i++) mask[i] = Math.round(255 * smooth(t * 0.8, t * 1.2, mag[i]));
  const soft = await sharp(mask, { raw: { width: w, height: h, channels: 1 } }).blur(Math.max(0.6, side / 1500)).raw().toBuffer();
  const out = new Uint8Array(w * h);
  for (let i = 0; i < out.length; i++) out[i] = clamp(Math.round((soft[i] / 255 - 0.2) * 2.2 * 255), 0, 255);
  return out;
}

function sat2d(src, w, h) {
  const W = w + 1, t = new Float64Array(W * (h + 1));
  for (let y = 0; y < h; y++) { let row = 0; for (let x = 0; x < w; x++) { row += src[y * w + x]; t[(y + 1) * W + x + 1] = t[y * W + x + 1] + row; } }
  return t;
}
const rect = (t, W, x0, y0, x1, y1) => t[(y1 + 1) * W + x1 + 1] - t[y0 * W + x1 + 1] - t[(y1 + 1) * W + x0] + t[y0 * W + x0];

function kuwahara(rgb, w, h, r) {
  const n = w * h, W = w + 1;
  const C = [0, 1, 2].map(() => new Float32Array(n)), L = new Float32Array(n), L2 = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 3; c++) C[c][i] = rgb[i * 3 + c];
    const l = 0.299 * rgb[i * 3] + 0.587 * rgb[i * 3 + 1] + 0.114 * rgb[i * 3 + 2];
    L[i] = l; L2[i] = l * l;
  }
  const sC = C.map((a) => sat2d(a, w, h)), sL = sat2d(L, w, h), sL2 = sat2d(L2, w, h);
  const out = new Uint8Array(n * 3);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let best = Infinity, m = [0, 0, 0];
    for (let q = 0; q < 4; q++) {
      const x0 = Math.max(0, q & 1 ? x : x - r), x1 = Math.min(w - 1, q & 1 ? x + r : x);
      const y0 = Math.max(0, q & 2 ? y : y - r), y1 = Math.min(h - 1, q & 2 ? y + r : y);
      const a = (x1 - x0 + 1) * (y1 - y0 + 1), mL = rect(sL, W, x0, y0, x1, y1) / a;
      const v = rect(sL2, W, x0, y0, x1, y1) / a - mL * mL;
      if (v < best) { best = v; m = sC.map((s) => rect(s, W, x0, y0, x1, y1) / a); }
    }
    const o = (y * w + x) * 3; out[o] = m[0]; out[o + 1] = m[1]; out[o + 2] = m[2];
  }
  return out;
}

async function filterToon(buffer) {
  const { data: orig, info } = await sharp(buffer, { failOn: 'none' }).rotate()
    .resize({ width: MAX_DIM, height: MAX_DIM, fit: 'inside', withoutEnlargement: true })
    .flatten({ background: '#fff' }).toColourspace('srgb').removeAlpha().raw().toBuffer({ resolveWithObject: true });
  const w = info.width, h = info.height, side = Math.max(w, h), n = w * h;

  const gray = new Uint8Array(n);
  for (let i = 0; i < n; i++) gray[i] = 0.299 * orig[i * 3] + 0.587 * orig[i * 3 + 1] + 0.114 * orig[i * 3 + 2];
  const ink = await inkLines(gray, w, h, side);

  let img = kuwahara(orig, w, h, Math.max(2, Math.round(side / 380)));
  img = kuwahara(img, w, h, Math.max(2, Math.round(side / 380) - 1));

  const N = P.bands, cell = Math.max(5, Math.round(side / 130)), hp = Math.max(4, Math.round(side / 260));
  const out = Buffer.alloc(n * 3), c45 = Math.SQRT1_2;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x, o = i * 3;
    let r = img[o] / 255, g = img[o + 1] / 255, b = img[o + 2] / 255;
    const L = 0.299 * r + 0.587 * g + 0.114 * b, Lc = clamp((L - 0.5) * 1.05 + 0.5, 0, 1);
    const band = Math.min(N - 1, Math.floor(Lc * N)), Lq = band === N - 1 ? 1 : (band + 0.5) / N;
    const f = (Lq + 0.06) / (L + 0.06); r *= f; g *= f; b *= f;
    const gr = 0.299 * r + 0.587 * g + 0.114 * b;
    r = gr + (r - gr) * P.sat; g = gr + (g - gr) * P.sat; b = gr + (b - gr) * P.sat;
    let k = 1;
    const u = (x + y) * c45 / cell, v = (x - y) * c45 / cell;
    if (Math.hypot(u - Math.round(u), v - Math.round(v)) < 0.55 * Math.sqrt(clamp(1 - Lc, 0, 1))) {
      k *= 1 - 0.2 * P.halftone * smooth(0.3, 0.5, Lc) * (1 - smooth(0.7, 0.92, Lc));
    }
    if (Lq < 0.35) { const d = (((x + y) % hp) + hp) % hp; k *= 1 - 0.5 * P.hatch * clamp(1.2 - Math.min(d, hp - d), 0, 1); }
    if (Lq < 0.2) { const d = (((x - y) % hp) + hp) % hp; k *= 1 - 0.5 * P.hatch * clamp(1.2 - Math.min(d, hp - d), 0, 1); }
    const m = ink[i] / 255;
    out[o] = clamp(Math.round(r * k * 255 * (1 - m) + 26 * m), 0, 255);
    out[o + 1] = clamp(Math.round(g * k * 255 * (1 - m) + 20 * m), 0, 255);
    out[o + 2] = clamp(Math.round(b * k * 255 * (1 - m) + 22 * m), 0, 255);
  }
  return sharp(out, { raw: { width: w, height: h, channels: 3 } }).png().toBuffer();
}

// ---------------- hybrid: AI first, filter as fallback ----------------
async function aiToon(buffer) {
  if (!ENGINE) throw new Error('no engine configured');
  const r = await fetch(ENGINE + '/toon', {
    method: 'POST', headers: { 'Content-Type': 'application/octet-stream' },
    body: buffer, signal: AbortSignal.timeout(10 * 60 * 1000),
  });
  if (!r.ok) throw new Error('engine ' + r.status);
  return Buffer.from(await r.arrayBuffer());
}

const app = express();
app.get('/health', (_q, res) => res.json({ ok: true, engine: ENGINE || null }));
app.post('/toon', express.raw({ type: '*/*', limit: '25mb' }), async (req, res) => {
  if (!Buffer.isBuffer(req.body) || !req.body.length) return res.status(400).send('Send an image as the request body');
  try {
    let png, used = 'ai';
    try { png = await aiToon(req.body); }
    catch (e) { console.log('[hybrid] AI unavailable (' + e.message + '), using filter'); used = 'filter'; png = await filterToon(req.body); }
    res.set('X-Engine', used).type('png').send(png);
  } catch (e) {
    console.error(e);
    res.status(500).send(e.message);
  }
});
app.get('/', (_q, res) => res.type('html').send(PAGE));
app.listen(PORT, () => console.log('Toonify on :' + PORT + (ENGINE ? ' (AI engine: ' + ENGINE + ')' : ' (filter only)')));

const PAGE = `<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Toonify</title>
<style>
:root{color-scheme:light dark;font-family:system-ui,sans-serif}
body{max-width:900px;margin:0 auto;padding:20px 16px}
input,button{font:inherit;padding:10px;border-radius:8px;border:1px solid #8885}
button{background:#d33;color:#fff;border:0;font-weight:600;cursor:pointer}button:disabled{opacity:.5}
.row{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:14px}
@media(max-width:640px){.row{grid-template-columns:1fr}}
img{width:100%;border-radius:10px}figcaption{font-size:.85rem;opacity:.7}figure{margin:0}
</style></head><body>
<h1>Toonify</h1>
<input id="file" type="file" accept="image/*"> <button id="go">Toonify</button>
<p id="st"></p>
<div class="row"><figure><img id="a" hidden><figcaption>Original</figcaption></figure>
<figure><img id="b" hidden><figcaption id="cap"></figcaption><a id="dl" hidden download="toon.png">Download</a></figure></div>
<script>
var $=function(i){return document.getElementById(i)};
$('file').onchange=function(){var f=this.files[0];if(f){$('a').src=URL.createObjectURL(f);$('a').hidden=false}};
$('go').onclick=function(){
  var f=$('file').files[0]; if(!f) return $('st').textContent='Choose a photo first';
  $('go').disabled=true; $('st').textContent='Working... (AI can take up to a minute)';
  fetch('/toon',{method:'POST',body:f}).then(function(r){
    if(!r.ok) return r.text().then(function(t){throw new Error(t)});
    var eng=r.headers.get('X-Engine'); return r.blob().then(function(b){
      var u=URL.createObjectURL(b); $('b').src=u; $('b').hidden=false; $('dl').href=u; $('dl').hidden=false;
      $('cap').textContent='Toon - made by '+(eng==='ai'?'AI engine':'built-in filter (AI engine offline)');
      $('st').textContent='';
    });
  }).catch(function(e){$('st').textContent='Error: '+e.message}).finally(function(){$('go').disabled=false});
};
</script></body></html>`;
