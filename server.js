'use strict';
/**
 * Toonify (self-contained edition)
 * Photo in -> comic-book version out. 100% local: no APIs, no keys, no network.
 *
 * Every output pixel is derived from the original pixels, so composition, pose,
 * colors and any printed text stay exactly where they were.
 *
 * Pipeline (all in this file, using only `sharp` for fast blur/IO):
 *   1. LOAD      fix EXIF rotation, cap size
 *   2. INK       Sobel edges on a lightly blurred copy -> adaptive threshold -> clean ink-line mask
 *   3. SMOOTH    Kuwahara filter (O(1) per pixel via summed-area tables): flat painterly regions, edges kept sharp
 *   4. CEL       quantize brightness into bands, keep the original hue, boost saturation
 *   5. HATCH     diagonal (and cross) hatching in the darkest bands
 *   6. HALFTONE  Ben-Day dots in mid/light tones (sky, skin highlights)
 *   7. COMPOSE   ink lines over the result
 *
 * Heavy work runs in worker threads so the server stays responsive.
 *
 * Usage:
 *   npm start                                   -> web UI + API on :3000
 *   node server.js --cli in.jpg [out.png] [style]  -> one-off conversion, no server
 */
const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');
const sharp = require('sharp');
const fs = require('fs');
const os = require('os');
const path = require('path');

const MAX_DIM = Number(process.env.MAX_DIM) || 1600;

// ---------- style presets ----------
const PRESETS = {
  comic:  { label: 'Comic book (ink + hatching + halftone)', bands: 5, sat: 1.25, lines: 0.55, hatch: 0.8,  halftone: 0.6, dot: 1,   smooth: 2 },
  popart: { label: 'Retro pop-art (big dots, bold color)',   bands: 3, sat: 1.6,  lines: 0.5,  hatch: 0.25, halftone: 1,   dot: 1.8, smooth: 2 },
  noir:   { label: 'Noir (black & white ink)',               bands: 4, sat: 0,    lines: 0.7,  hatch: 1,    halftone: 0.3, dot: 1,   smooth: 2 },
};

const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
const smoothstep = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };

// ---------- 1. load ----------
async function loadRaw(buffer) {
  const { data, info } = await sharp(buffer, { failOn: 'none' })
    .rotate()
    .resize({ width: MAX_DIM, height: MAX_DIM, fit: 'inside', withoutEnlargement: true })
    .flatten({ background: '#ffffff' })
    .toColourspace('srgb')
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  if (info.channels !== 3) throw new Error('Unexpected channel count: ' + info.channels);
  return { data, w: info.width, h: info.height };
}

// ---------- 2. ink lines ----------
async function inkLines(gray, w, h, strength, side) {
  const sigma = Math.max(0.8, side / 900);
  const g = await sharp(gray, { raw: { width: w, height: h, channels: 1 } }).blur(sigma).raw().toBuffer();

  const mag = new Float32Array(w * h);
  let max = 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const tl = g[i - w - 1], t = g[i - w], tr = g[i - w + 1];
      const l = g[i - 1], r = g[i + 1];
      const bl = g[i + w - 1], b = g[i + w], br = g[i + w + 1];
      const gx = tr + 2 * r + br - tl - 2 * l - bl;
      const gy = bl + 2 * b + br - tl - 2 * t - tr;
      const m = Math.sqrt(gx * gx + gy * gy);
      mag[i] = m;
      if (m > max) max = m;
    }
  }
  if (max === 0) return new Uint8Array(w * h);

  // adaptive threshold: keep the strongest X% of edges (X grows with `strength`)
  const bins = 2048;
  const hist = new Uint32Array(bins);
  const scale = (bins - 1) / max;
  for (let i = 0; i < mag.length; i++) hist[(mag[i] * scale) | 0]++;
  const keepPct = 0.97 - 0.12 * clamp(strength, 0, 1);
  const target = keepPct * mag.length;
  let acc = 0, t = max;
  for (let b = 0; b < bins; b++) { acc += hist[b]; if (acc >= target) { t = b / scale; break; } }
  t = Math.max(t, 30); // never draw ink on pure sensor noise

  const mask = Buffer.alloc(w * h);
  for (let i = 0; i < mag.length; i++) mask[i] = Math.round(255 * smoothstep(t * 0.8, t * 1.2, mag[i]));

  // soften + re-sharpen: thickens lines slightly and removes stray specks
  const soft = await sharp(mask, { raw: { width: w, height: h, channels: 1 } })
    .blur(Math.max(0.6, side / 1500)).raw().toBuffer();
  const out = new Uint8Array(w * h);
  for (let i = 0; i < out.length; i++) out[i] = clamp(Math.round((soft[i] / 255 - 0.2) * 2.2 * 255), 0, 255);
  return out;
}

// ---------- 3. Kuwahara (painterly smoothing, edge preserving) ----------
function buildSAT(src, w, h) {
  const W = w + 1;
  const t = new Float64Array(W * (h + 1));
  for (let y = 0; y < h; y++) {
    let row = 0;
    for (let x = 0; x < w; x++) {
      row += src[y * w + x];
      t[(y + 1) * W + (x + 1)] = t[y * W + (x + 1)] + row;
    }
  }
  return t;
}
function rectSum(t, W, x0, y0, x1, y1) {
  return t[(y1 + 1) * W + (x1 + 1)] - t[y0 * W + (x1 + 1)] - t[(y1 + 1) * W + x0] + t[y0 * W + x0];
}

function kuwahara(rgb, w, h, r) {
  const n = w * h, W = w + 1;
  const R = new Float32Array(n), G = new Float32Array(n), B = new Float32Array(n);
  const L = new Float32Array(n), L2 = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const rr = rgb[i * 3], gg = rgb[i * 3 + 1], bb = rgb[i * 3 + 2];
    R[i] = rr; G[i] = gg; B[i] = bb;
    const l = 0.299 * rr + 0.587 * gg + 0.114 * bb;
    L[i] = l; L2[i] = l * l;
  }
  const sR = buildSAT(R, w, h), sG = buildSAT(G, w, h), sB = buildSAT(B, w, h);
  const sL = buildSAT(L, w, h), sL2 = buildSAT(L2, w, h);

  const out = new Uint8Array(n * 3);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let bestVar = Infinity, bR = 0, bG = 0, bB = 0;
      for (let q = 0; q < 4; q++) {
        const x0 = Math.max(0, q & 1 ? x : x - r), x1 = Math.min(w - 1, q & 1 ? x + r : x);
        const y0 = Math.max(0, q & 2 ? y : y - r), y1 = Math.min(h - 1, q & 2 ? y + r : y);
        const area = (x1 - x0 + 1) * (y1 - y0 + 1);
        const mL = rectSum(sL, W, x0, y0, x1, y1) / area;
        const v = rectSum(sL2, W, x0, y0, x1, y1) / area - mL * mL;
        if (v < bestVar) {
          bestVar = v;
          bR = rectSum(sR, W, x0, y0, x1, y1) / area;
          bG = rectSum(sG, W, x0, y0, x1, y1) / area;
          bB = rectSum(sB, W, x0, y0, x1, y1) / area;
        }
      }
      const o = (y * w + x) * 3;
      out[o] = bR; out[o + 1] = bG; out[o + 2] = bB;
    }
  }
  return out;
}

// ---------- 4-7. shade + compose ----------
function shade(img, lineMask, w, h, p, side) {
  const N = clamp(Math.round(p.bands), 2, 8);
  const cell = Math.max(5, Math.round((side / 130) * (p.dot || 1)));
  const hp = Math.max(4, Math.round(side / 260)); // hatch spacing
  const ink = [26, 20, 22];
  const c45 = Math.SQRT1_2;
  const out = Buffer.alloc(w * h * 3);

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x, o = i * 3;
      let r = img[o] / 255, g = img[o + 1] / 255, b = img[o + 2] / 255;
      const L = 0.299 * r + 0.587 * g + 0.114 * b;

      // cel shading: quantize brightness, keep hue
      // band centers (not 0): shadows keep detail and skin tone instead of crushing to black; top band stays white
      const Lc = clamp((L - 0.5) * 1.05 + 0.5, 0, 1);
      const band = Math.min(N - 1, Math.floor(Lc * N));
      const Lq = band === N - 1 ? 1 : (band + 0.5) / N;
      const f = (Lq + 0.06) / (L + 0.06);
      r *= f; g *= f; b *= f;

      // saturation
      const gr = 0.299 * r + 0.587 * g + 0.114 * b;
      r = gr + (r - gr) * p.sat; g = gr + (g - gr) * p.sat; b = gr + (b - gr) * p.sat;

      // multiplicative darkening from halftone + hatching
      let k = 1;
      if (p.halftone > 0) {
        const u = (x + y) * c45 / cell, v = (x - y) * c45 / cell;
        const d = Math.hypot(u - Math.round(u), v - Math.round(v));
        const mid = smoothstep(0.3, 0.5, Lc) * (1 - smoothstep(0.7, 0.92, Lc));
        const rad = 0.55 * Math.sqrt(clamp(1 - Lc, 0, 1));
        if (d < rad) k *= 1 - 0.2 * p.halftone * mid;
      }
      if (p.hatch > 0) {
        if (Lq < 0.35) {
          const d = (((x + y) % hp) + hp) % hp;
          k *= 1 - 0.5 * p.hatch * clamp(1.2 - Math.min(d, hp - d), 0, 1);
        }
        if (Lq < 0.2) {
          const d = (((x - y) % hp) + hp) % hp;
          k *= 1 - 0.5 * p.hatch * clamp(1.2 - Math.min(d, hp - d), 0, 1);
        }
      }
      r *= k; g *= k; b *= k;

      // ink on top
      const m = lineMask[i] / 255;
      out[o]     = clamp(Math.round((r * (1 - m)) * 255 + ink[0] * m), 0, 255);
      out[o + 1] = clamp(Math.round((g * (1 - m)) * 255 + ink[1] * m), 0, 255);
      out[o + 2] = clamp(Math.round((b * (1 - m)) * 255 + ink[2] * m), 0, 255);
    }
  }
  return out;
}

// sanity check: do large-scale light/dark layouts still match? (Pearson on a 48x48 grayscale thumbnail)
async function structureScore(a, b, w, h) {
  const thumb = (buf) =>
    sharp(buf, { raw: { width: w, height: h, channels: 3 } })
      .greyscale().resize(48, 48, { fit: 'fill' }).raw().toBuffer();
  const [ta, tb] = await Promise.all([thumb(a), thumb(b)]);
  const n = ta.length;
  let ma = 0, mb = 0;
  for (let i = 0; i < n; i++) { ma += ta[i]; mb += tb[i]; }
  ma /= n; mb /= n;
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < n; i++) {
    const x = ta[i] - ma, y = tb[i] - mb;
    num += x * y; da += x * x; db += y * y;
  }
  return da && db ? Math.round(clamp(num / Math.sqrt(da * db), 0, 1) * 100) : 0;
}

// ---------- main conversion ----------
async function toonify(buffer, o = {}) {
  const t0 = Date.now();
  const p = { ...(PRESETS[o.style] || PRESETS.comic) };
  if (Number.isFinite(o.lines)) p.lines = clamp(o.lines, 0, 1);
  if (Number.isFinite(o.hatch)) p.hatch = clamp(o.hatch, 0, 1);
  if (Number.isFinite(o.bands)) p.bands = clamp(Math.round(o.bands), 2, 8);

  const { data: orig, w, h } = await loadRaw(buffer);
  const side = Math.max(w, h);

  const gray = new Uint8Array(w * h);
  for (let i = 0; i < gray.length; i++) {
    gray[i] = 0.299 * orig[i * 3] + 0.587 * orig[i * 3 + 1] + 0.114 * orig[i * 3 + 2];
  }

  const lineMask = await inkLines(gray, w, h, p.lines, side);

  let img = orig;
  const r = Math.max(2, Math.round(side / 380));
  for (let i = 0; i < p.smooth; i++) img = kuwahara(img, w, h, i === 0 ? r : Math.max(2, r - 1));

  const out = shade(img, lineMask, w, h, p, side);
  const score = await structureScore(orig, out, w, h);
  const png = await sharp(out, { raw: { width: w, height: h, channels: 3 } })
    .png({ compressionLevel: 6 }).toBuffer();
  return { png, score, ms: Date.now() - t0, width: w, height: h };
}

// ---------- worker plumbing ----------
function runInWorker(buffer, opts) {
  return new Promise((resolve, reject) => {
    const wk = new Worker(__filename, { workerData: { buffer, opts } });
    wk.once('message', (m) => (m.error ? reject(new Error(m.error)) : resolve({ ...m, png: Buffer.from(m.png) })));
    wk.once('error', reject);
    wk.once('exit', (c) => { if (c !== 0) reject(new Error('Worker exited with code ' + c)); });
  });
}

if (!isMainThread) {
  toonify(Buffer.from(workerData.buffer), workerData.opts)
    .then((r) => parentPort.postMessage(r))
    .catch((e) => parentPort.postMessage({ error: e.message }));
} else if (process.argv[2] === '--cli') {
  const [, , , input, output, style] = process.argv;
  if (!input) { console.error('Usage: node server.js --cli input.jpg [output.png] [comic|popart|noir]'); process.exit(1); }
  const outPath = output || path.join(path.dirname(input), path.basename(input, path.extname(input)) + '.toon.png');
  toonify(fs.readFileSync(input), { style })
    .then((r) => { fs.writeFileSync(outPath, r.png); console.log(`Wrote ${outPath} (${r.width}x${r.height}, ${r.ms} ms, structure ${r.score}/100)`); })
    .catch((e) => { console.error(e.message); process.exit(1); });
} else {
  startServer();
}

// ---------- web app ----------
function startServer() {
  const express = require('express');
  const multer = require('multer');

  const PORT = Number(process.env.PORT) || 3000;
  const MAX_UPLOAD_MB = Number(process.env.MAX_UPLOAD_MB) || 25;
  const MAX_JOBS = Number(process.env.MAX_CONCURRENT_JOBS) || Math.max(1, os.cpus().length - 1);
  let running = 0;

  const app = express();
  const upload = multer({
    storage: multer.memoryStorage(),
    limits: { fileSize: MAX_UPLOAD_MB * 1024 * 1024 },
    fileFilter: (_req, file, cb) =>
      /^image\//i.test(file.mimetype) ? cb(null, true) : cb(new Error('Only image files are allowed')),
  });

  app.get('/health', (_req, res) => res.json({ ok: true, running, maxJobs: MAX_JOBS }));
  app.get('/api/styles', (_req, res) =>
    res.json(Object.entries(PRESETS).map(([key, s]) => ({ key, label: s.label }))));

  // POST /api/toonify  multipart: photo, style?, lines?, bands?, hatch?   -> PNG
  app.post('/api/toonify', upload.single('photo'), async (req, res) => {
    if (!req.file) return res.status(400).json({ error: 'Attach an image in the "photo" field' });
    if (running >= MAX_JOBS) return res.status(429).json({ error: 'Busy, try again in a moment' });
    running++;
    try {
      const b = req.body || {};
      const num = (v) => (v === undefined || v === '' ? undefined : Number(v));
      const r = await runInWorker(req.file.buffer, {
        style: PRESETS[b.style] ? b.style : 'comic',
        lines: num(b.lines), bands: num(b.bands), hatch: num(b.hatch),
      });
      res.set('X-Processing-Ms', String(r.ms));
      res.set('X-Structure-Score', String(r.score));
      res.type('png').send(r.png);
    } catch (e) {
      console.error(e);
      res.status(500).json({ error: e.message });
    } finally {
      running--;
    }
  });

  app.get('/', (_req, res) => res.type('html').send(PAGE));

  // eslint-disable-next-line no-unused-vars
  app.use((err, _req, res, _next) => {
    res.status(err.code === 'LIMIT_FILE_SIZE' ? 413 : 400).json({ error: err.message });
  });

  app.listen(PORT, () => console.log(`Toonify (local engine) on http://localhost:${PORT}`));
}

const PAGE = `<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Toonify</title>
<style>
  :root{color-scheme:light dark;font-family:system-ui,sans-serif}
  body{max-width:1000px;margin:0 auto;padding:20px 16px}
  h1{margin:0 0 4px}
  form{display:grid;gap:10px;margin:16px 0}
  select,button,input{font:inherit;padding:10px;border-radius:8px;border:1px solid #8885}
  button{background:#d33;color:#fff;border:0;cursor:pointer;font-weight:600}
  button:disabled{opacity:.5;cursor:wait}
  fieldset{border:1px solid #8885;border-radius:8px;display:grid;gap:6px}
  fieldset:disabled{opacity:.45}
  label{display:grid;grid-template-columns:110px 1fr 40px;align-items:center;gap:8px;font-size:.9rem}
  .row{display:grid;grid-template-columns:1fr 1fr;gap:12px}
  @media(max-width:640px){.row{grid-template-columns:1fr}}
  img{width:100%;border-radius:10px;display:block}
  figure{margin:0}figcaption{font-size:.85rem;opacity:.7;margin-top:4px}
  #status{margin:8px 0;font-size:.95rem}
</style></head><body>
<h1>Toonify</h1>
<p>Upload a photo, get a comic-book version. Runs entirely on this machine.</p>
<form id="f">
  <input type="file" name="photo" accept="image/*" required>
  <select name="style" id="style"></select>
  <label style="grid-template-columns:auto 1fr;justify-content:start"><input type="checkbox" id="tune"> Fine-tune</label>
  <fieldset id="fs" disabled>
    <label>Ink lines <input type="range" name="lines" min="0" max="1" step="0.05" value="0.55"><output>0.55</output></label>
    <label>Color bands <input type="range" name="bands" min="2" max="8" step="1" value="5"><output>5</output></label>
    <label>Hatching <input type="range" name="hatch" min="0" max="1" step="0.05" value="0.8"><output>0.8</output></label>
  </fieldset>
  <button id="go">Toonify</button>
</form>
<div id="status"></div>
<div class="row">
  <figure><img id="orig" hidden><figcaption>Original</figcaption></figure>
  <figure><img id="out" hidden><figcaption id="cap">Toon</figcaption></figure>
</div>
<p><a id="dl" hidden download="toon.png">Download PNG</a></p>
<script>
var f=document.getElementById('f'),st=document.getElementById('status'),go=document.getElementById('go'),
    orig=document.getElementById('orig'),out=document.getElementById('out'),dl=document.getElementById('dl'),
    cap=document.getElementById('cap'),fs=document.getElementById('fs'),tune=document.getElementById('tune');
fetch('/api/styles').then(function(r){return r.json()}).then(function(s){
  document.getElementById('style').innerHTML=s.map(function(x){return '<option value="'+x.key+'">'+x.label+'</option>'}).join('');
});
tune.onchange=function(){fs.disabled=!tune.checked;};
Array.prototype.forEach.call(fs.querySelectorAll('input'),function(i){i.oninput=function(){i.nextElementSibling.textContent=i.value;};});
f.querySelector('input[type=file]').onchange=function(e){
  var file=e.target.files[0]; if(file){orig.src=URL.createObjectURL(file);orig.hidden=false;}
};
f.onsubmit=function(e){
  e.preventDefault(); go.disabled=true; st.textContent='Converting...';
  fetch('/api/toonify',{method:'POST',body:new FormData(f)}).then(function(r){
    if(!r.ok) return r.json().then(function(j){throw new Error(j.error)});
    var ms=r.headers.get('X-Processing-Ms'),sc=r.headers.get('X-Structure-Score');
    return r.blob().then(function(b){
      var u=URL.createObjectURL(b); out.src=u; out.hidden=false; dl.href=u; dl.hidden=false;
      cap.textContent='Toon ('+(ms/1000).toFixed(1)+'s, structure match '+sc+'/100)';
      st.textContent='Done'; go.disabled=false;
    });
  }).catch(function(e){st.textContent='Error: '+e.message;go.disabled=false;});
};
</script></body></html>`;
