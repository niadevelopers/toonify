'use strict';
/**
 * Local hand-drawn image renderer.
 *
 * Install: npm i express sharp
 * Run:    node pencil-server.js
 * POST an image to /pencil, or open / in a browser.
 *
 * This is intentionally non-generative: it runs entirely on the server and
 * preserves the source composition while rendering it as graphite on paper.
 * Optional query parameters:
 *   /pencil?color=1       light colored-pencil underdrawing
 *   /pencil?strength=0.8  0.45..1.35, controls graphite density
 */

const express = require('express');
const sharp = require('sharp');

const PORT = Number(process.env.PORT || 3000);
const MAX_DIM = Number(process.env.MAX_DIM || 1600);
const MAX_BYTES = Number(process.env.MAX_BYTES || 30 * 1024 * 1024);

const clamp = (v, lo = 0, hi = 1) => v < lo ? lo : v > hi ? hi : v;
const lerp = (a, b, t) => a + (b - a) * t;
const smooth = (a, b, x) => { const t = clamp((x - a) / (b - a)); return t * t * (3 - 2 * t); };

function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

async function rawBlur(buf, w, h, sigma) {
  return sharp(buf, { raw: { width: w, height: h, channels: 1 } })
    .blur(Math.max(0.25, sigma)).raw().toBuffer();
}

async function noiseField(w, h, cell, random) {
  const sw = Math.max(2, Math.ceil(w / cell));
  const sh = Math.max(2, Math.ceil(h / cell));
  const small = Buffer.alloc(sw * sh);
  for (let i = 0; i < small.length; i++) small[i] = random() * 255;
  return sharp(small, { raw: { width: sw, height: sh, channels: 1 } })
    .resize(w, h, { fit: 'fill', kernel: 'cubic' }).raw().toBuffer();
}

function sample(buf, x, y, w, h) {
  x = Math.max(0, Math.min(w - 1, x));
  y = Math.max(0, Math.min(h - 1, y));
  return buf[y * w + x];
}

async function render(input, colored, requestedStrength) {
  const strength = clamp(Number.isFinite(requestedStrength) ? requestedStrength : 0.92, 0.45, 1.35);
  const decoded = await sharp(input, { failOn: 'none' }).rotate()
    .resize({ width: MAX_DIM, height: MAX_DIM, fit: 'inside', withoutEnlargement: true })
    .flatten({ background: '#ffffff' }).toColourspace('srgb').removeAlpha()
    .raw().toBuffer({ resolveWithObject: true });

  const { data: rgb, info } = decoded;
  const w = info.width, h = info.height, n = w * h;
  const side = Math.max(w, h);
  const random = rng((w * 73856093) ^ (h * 19349663) ^ n);

  // Work from a mildly softened luminance image. It keeps faces and small forms
  // intact instead of turning the whole photograph into a foggy silhouette.
  const gray = Buffer.alloc(n);
  for (let i = 0; i < n; i++) {
    const p = i * 3;
    gray[i] = clamp(Math.round(0.2126 * rgb[p] + 0.7152 * rgb[p + 1] + 0.0722 * rgb[p + 2]), 0, 255);
  }
  const base = await rawBlur(gray, w, h, Math.max(0.35, side / 900));
  const broad = await rawBlur(gray, w, h, Math.max(1.2, side / 70));
  const paperNoise = await noiseField(w, h, Math.max(18, side / 18), random);
  const fiberNoise = await noiseField(w, h, Math.max(4, side / 75), random);

  // Human pencil linework: local gradient + a high-frequency contour pass.
  const edge = Buffer.alloc(n);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const gx = sample(base, x + 1, y, w, h) - sample(base, x - 1, y, w, h);
      const gy = sample(base, x, y + 1, w, h) - sample(base, x, y - 1, w, h);
      const gradient = Math.sqrt(gx * gx + gy * gy) / 255;
      const high = Math.abs(gray[i] - base[i]) / 255;
      // Soft threshold avoids noisy outlines in flat photographic regions.
      edge[i] = clamp(Math.pow(smooth(0.025, 0.30, gradient) * 0.82 + high * 0.24, 0, 1) * 255);
    }
  }

  // Three families of irregular strokes, at different scales and angles.
  const jA = await noiseField(w, h, Math.max(7, side / 25), random);
  const jB = await noiseField(w, h, Math.max(11, side / 19), random);
  const jC = await noiseField(w, h, Math.max(5, side / 34), random);
  const period = Math.max(6, side / 125);
  const angles = [0.78, -0.58, 1.26];
  const dirs = angles.map(a => [Math.cos(a), Math.sin(a)]);
  const out = Buffer.alloc(n * 3);

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x, o = i * 3;
      const value = base[i] / 255;
      const shadow = clamp(1 - value);
      const localShadow = clamp(1 - broad[i] / 255);
      const paperJitter = (paperNoise[i] - 128) / 128;
      const fiber = (fiberNoise[i] - 128) / 128;

      // Edges are the darkest marks; shadows build gradually underneath them.
      const contour = clamp(edge[i] / 255 * (0.92 + 0.10 * paperJitter));
      const modeledTone = Math.pow(shadow, 1.12) * 0.54 * strength;
      const softTone = Math.pow(localShadow, 1.45) * 0.17 * strength;

      // Hatching appears mainly in modeled shadows, not over the entire canvas.
      const hatchZone = smooth(0.15, 0.58, shadow) * (1 - smooth(0.86, 1.0, value));
      const line = (x, y, dir, field, width, phase) => {
        const u = (x * dir[0] + y * dir[1]) / period + (field[i] - 128) / 190 + phase;
        const d = Math.abs((u - Math.floor(u)) - 0.5) * 2;
        return clamp((width - d) / 0.34) * (0.88 + 0.15 * paperJitter);
      };
      const hatchA = line(x, y, dirs[0], jA, 0.73, 0.11) * hatchZone * 0.30 * strength;
      const hatchB = line(x, y, dirs[1], jB, 0.63, 0.37) * smooth(0.36, 0.82, shadow) * 0.24 * strength;
      const hatchC = line(x, y, dirs[2], jC, 0.52, 0.71) * smooth(0.61, 0.94, shadow) * 0.14 * strength;
      const graphite = clamp(contour * 0.78 + modeledTone + softTone + hatchA + hatchB + hatchC);

      // Warm, slightly uneven paper with a subtle edge falloff.
      const dx = x / Math.max(1, w - 1) - 0.5, dy = y / Math.max(1, h - 1) - 0.5;
      const vignette = 1 - 0.055 * smooth(0.45, 0.78, Math.hypot(dx, dy) * 1.55);
      const paper = vignette * (1 + paperJitter * 0.018 + fiber * 0.008);
      let pr = 249 * paper, pg = 246 * paper, pb = 239 * paper;

      if (colored) {
        // Restrained colored-pencil wash remains subordinate to graphite.
        const colorAmount = clamp((0.08 + 0.25 * shadow) * (1 - contour * 0.25), 0, 0.32);
        pr = lerp(pr, rgb[o] * 1.04, colorAmount);
        pg = lerp(pg, rgb[o + 1] * 1.04, colorAmount);
        pb = lerp(pb, rgb[o + 2] * 1.04, colorAmount);
      }

      // A faint paper gap prevents the result from becoming a solid black/white filter.
      const dark = clamp(graphite, 0, 0.93);
      out[o] = clamp(Math.round(lerp(pr, 38, dark)));
      out[o + 1] = clamp(Math.round(lerp(pg, 35, dark)));
      out[o + 2] = clamp(Math.round(lerp(pb, 31, dark)));
    }
  }

  return sharp(out, { raw: { width: w, height: h, channels: 3 } })
    .jpeg({ quality: 94, mozjpeg: true }).toBuffer();
}

const PAGE = `<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Graphite Studio</title>
<style>body{max-width:1100px;margin:auto;padding:22px 16px;font:16px system-ui;background:#f4f0e8;color:#26221d}h1{font-family:Georgia,serif}.bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap}input,button{font:inherit;padding:10px;border:1px solid #9b9184;border-radius:8px}button{background:#302a24;color:white;border:0;font-weight:700;cursor:pointer}button:disabled{opacity:.5}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:18px}@media(max-width:700px){.grid{grid-template-columns:1fr}}figure{margin:0;background:#fffdf8;padding:10px;border-radius:10px;box-shadow:0 2px 12px #46351b18}img{width:100%;display:block;border-radius:5px}figcaption{padding-top:8px;color:#655a4c;font-size:.9rem}a{display:inline-block;margin-top:8px;color:#684a28}</style></head><body>
<h1>Graphite Studio</h1><p>Local rendering: contour drawing, layered graphite, cross-hatching and warm paper texture.</p>
<div class="bar"><input id="file" type="file" accept="image/*"><label><input id="color" type="checkbox"> colored pencil wash</label><label>strength <input id="strength" type="range" min="45" max="135" value="92"></label><button id="go">Draw image</button></div><p id="status"></p>
<div class="grid"><figure><img id="original" hidden><figcaption>Original</figcaption></figure><figure><img id="result" hidden><figcaption>Hand-drawn render</figcaption><a id="download" hidden download="hand-drawn.jpg">Download JPEG</a></figure></div>
<script>const $=id=>document.getElementById(id);$('file').onchange=()=>{const f=$('file').files[0];if(f){$('original').src=URL.createObjectURL(f);$('original').hidden=false}};$('go').onclick=async()=>{const f=$('file').files[0];if(!f)return $('status').textContent='Choose an image first';$('go').disabled=true;$('status').textContent='Rendering graphite...';try{const q=new URLSearchParams({strength:(+$('strength').value/100).toFixed(2)});if($('color').checked)q.set('color','1');const r=await fetch('/pencil?'+q,{method:'POST',body:f});if(!r.ok)throw Error(await r.text());const u=URL.createObjectURL(await r.blob());$('result').src=u;$('result').hidden=false;$('download').href=u;$('download').hidden=false;$('status').textContent='Done'}catch(e){$('status').textContent='Error: '+e.message}finally{$('go').disabled=false}};</script></body></html>`;

const app = express();
app.get('/health', (_req, res) => res.json({ ok: true }));
app.get('/', (_req, res) => res.type('html').send(PAGE));
app.post('/pencil', express.raw({ type: '*/*', limit: MAX_BYTES }), async (req, res) => {
  if (!Buffer.isBuffer(req.body) || !req.body.length) return res.status(400).send('Send an image as the request body');
  try {
    const started = Date.now();
    const jpg = await render(req.body, req.query.color === '1', Number(req.query.strength));
    res.set('X-Render-Ms', String(Date.now() - started)).type('jpeg').send(jpg);
  } catch (err) {
    console.error(err);
    res.status(422).send('Could not process that image: ' + err.message);
  }
});
app.listen(PORT, () => console.log('Graphite Studio listening on :' + PORT));
