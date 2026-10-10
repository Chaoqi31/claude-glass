// The museum's moving parts: walls that take on each room's colour as a visitor walks in; and one <dialog>
// that shows every work (a plate zooms and pans, a long painting unrolls from its left end) and turns it over,
// to the program on its back.
const root = document.documentElement;
root.classList.add('js');
const links = [...document.querySelectorAll('main a.plate')];
const viewer = document.querySelector('.viewer');
const stage = viewer.querySelector('.stage');
const back = viewer.querySelector('.back');
const calm = matchMedia('(prefers-reduced-motion: reduce)');
const pts = new Map(), sources = new Map();
let i = 0, img, nw, nh, s = 1, fit = 1, x = 0, y = 0, g = 1, from, opener;

const turned = () => viewer.classList.contains('turned');
const unrolled = () => viewer.classList.contains('scroll') && !turned();
const one = () => 1 / devicePixelRatio;  // one plate pixel to one screen pixel

// The walls: a room's colours come up as it reaches the middle of the screen, and its name in the marker.
const marker = document.querySelector('.marker'), here = marker.querySelector('summary');
const theme = document.querySelector('meta[name="theme-color"]');
const walk = new IntersectionObserver(seen => {
  for (const q of seen) {
    if (!q.isIntersecting) continue;
    const d = q.target.dataset;
    for (const k of ['wall', 'ink', 'dim']) root.style.setProperty(`--${k}`, d[k]);
    theme.content = d.wall;
    here.textContent = d.room || '';
    marker.classList.toggle('on', !!d.room);
    if (!d.room) marker.open = false;
  }
}, {rootMargin: '-50% 0px -50% 0px'});
document.querySelectorAll('[data-wall]').forEach(r => walk.observe(r));
marker.addEventListener('click', e => { if (e.target.closest('a')) marker.open = false; });

const rise = new IntersectionObserver(seen => {
  for (const q of seen) if (q.isIntersecting) { q.target.classList.add('seen'); rise.unobserve(q.target); }
}, {rootMargin: '0px 0px -8% 0px'});
document.querySelectorAll('.work').forEach(w => rise.observe(w));

// The film shows its poster and one play mark; once it plays, its own controls.
const film = document.getElementById('film');
if (film) {
  const play = document.querySelector('.play');
  play.addEventListener('click', () => film.play());
  film.addEventListener('play', () => { film.controls = true; play.hidden = true; });
  document.querySelector('a[href="#film"]').addEventListener('click', () => film.play());  // "Watch the film" at the door
}

// The viewer.
function open(n, by) {
  from = n;
  opener = by;
  viewer.showModal();
  show(n);
}

function show(n) {
  i = (n + links.length) % links.length;
  const a = links[i], fig = a.closest('figure'), thumb = a.querySelector('img');
  nw = +a.dataset.w;
  nh = +a.dataset.h;
  viewer.classList.toggle('scroll', fig.classList.contains('long'));
  viewer.setAttribute('aria-label', thumb.alt);
  viewer.querySelector('.where').textContent = fig.closest('.room').dataset.room;
  viewer.querySelector('.label').innerHTML = fig.querySelector('.label').innerHTML;
  hint();
  img = new Image(nw, nh);
  img.alt = thumb.alt;
  img.draggable = false;
  img.style.backgroundImage = `url("${thumb.src}")`;  // the thumbnail stands in while the plate loads
  img.src = a.href;
  stage.replaceChildren(img);
  viewer.scrollLeft = 0;
  if (turned()) code();
  else if (!unrolled()) reset();
}

function hint() {
  viewer.querySelector('.turn').textContent = turned() ? 'Turn back' : 'Turn over';
  viewer.querySelector('.hint').textContent = turned()
    ? 'The back of the canvas: the program that painted it. T turns it back; ← → for the next work.'
    : unrolled()
      ? 'A long painting, read from left to right: drag or scroll along it. T turns it over; ← → for the next work.'
      : 'Wheel or pinch to zoom, drag to move, double-click for 1:1. T turns it over; ← → for the next work.';
}

// The back of the canvas carries the work's own source, fetched once, and the way to it on GitHub.
function code() {
  const a = links[i], src = `works/${a.dataset.slug}.py`, out = back.querySelector('code'), file = back.querySelector('.file');
  const there = Object.assign(document.createElement('a'), {href: `${viewer.dataset.repo}/blob/main/${src}`, textContent: src});
  file.replaceChildren(there, ` · ${a.dataset.lines} lines of Python`);
  out.textContent = '';
  if (!sources.has(src)) sources.set(src, fetch(src).then(r => r.ok ? r.text() : Promise.reject(r.status)));
  sources.get(src).then(
    t => { if (links[i] === a) { out.textContent = t; back.firstElementChild.scrollTop = 0; } },
    () => { if (links[i] === a) out.textContent = `Read ${src} in the repository.`; });
}

function turn() {
  const flip = () => {
    viewer.classList.toggle('turned');
    hint();
    if (turned()) code();
    else if (!unrolled()) reset();
  };
  if (calm.matches) return flip();
  const p = 'perspective(2400px) rotateY';
  viewer.animate([{transform: `${p}(0deg)`}, {transform: `${p}(90deg)`}], {duration: 220, easing: 'ease-in'}).finished.then(() => {
    flip();
    viewer.animate([{transform: `${p}(-90deg)`}, {transform: `${p}(0deg)`}], {duration: 320, easing: 'ease-out'});
  });
}

function reset() {
  s = fit = Math.min(stage.clientWidth / nw, stage.clientHeight / nh);
  draw();
}

function draw(glide) {
  const W = stage.clientWidth, H = stage.clientHeight, w = nw * s, h = nh * s;
  x = w < W ? (W - w) / 2 : Math.min(0, Math.max(W - w, x));
  y = h < H ? (H - h) / 2 : Math.min(0, Math.max(H - h, y));
  img.style.transition = glide ? 'transform .3s ease' : 'none';
  img.style.transform = `translate(${x}px, ${y}px) scale(${s})`;
}

function zoom(cx, cy, to, glide) {
  const r = stage.getBoundingClientRect(), px = cx - r.left, py = cy - r.top;
  to = Math.min(4, Math.max(Math.min(fit, one()), to));
  x = px - (px - x) * to / s;
  y = py - (py - y) * to / s;
  s = to;
  draw(glide);
}

document.addEventListener('click', e => {
  const a = e.target.closest('main a.plate, .picture');  // a work on the walls, or the one at the door
  if (!a || e.button || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
  e.preventDefault();
  open(links.findIndex(l => l.href === a.href), a);
});
viewer.querySelector('.turn').onclick = turn;
viewer.querySelector('.prev').onclick = () => show(i - 1);
viewer.querySelector('.next').onclick = () => show(i + 1);
viewer.querySelector('.close').onclick = () => viewer.close();
viewer.addEventListener('close', () => {
  viewer.classList.remove('turned');
  stage.replaceChildren();
  (i === from ? opener : links[i]).focus({preventScroll: i === from});
});
new ResizeObserver(() => viewer.open && !unrolled() && !turned() && reset()).observe(stage);

document.addEventListener('keydown', e => {
  if (!viewer.open || e.metaKey || e.ctrlKey || e.altKey) return;
  const key = e.key, r = stage.getBoundingClientRect();
  if (key === 'ArrowRight' || key === 'ArrowLeft') show(i + (key === 'ArrowRight' ? 1 : -1));
  else if (key === 't' || key === 'T') turn();
  else if (unrolled() && (key === 'ArrowDown' || key === 'ArrowUp'))
    viewer.scrollBy({left: (key === 'ArrowDown' ? 0.6 : -0.6) * innerWidth, behavior: calm.matches ? 'auto' : 'smooth'});
  else if (!unrolled() && !turned() && ['+', '=', '-'].includes(key))
    zoom(r.left + r.width / 2, r.top + r.height / 2, s * (key === '-' ? 2 / 3 : 1.5), true);
  else return;
  e.preventDefault();
});

stage.addEventListener('pointerdown', e => {
  if (e.button || (unrolled() && e.pointerType !== 'mouse')) return;  // fingers scroll a long painting natively
  stage.setPointerCapture(e.pointerId);
  pts.set(e.pointerId, {x: e.clientX, y: e.clientY});
  stage.classList.add('held');
});
stage.addEventListener('pointermove', e => {
  const p = pts.get(e.pointerId);
  if (!p) return;
  const dx = e.clientX - p.x, dy = e.clientY - p.y, o = [...pts.values()].find(q => q !== p);
  if (unrolled()) viewer.scrollLeft -= dx;
  else if (!o) { x += dx; y += dy; draw(); }
  else {  // pinch: the midpoint moves half as far as this finger, and the spread sets the scale
    x += dx / 2;
    y += dy / 2;
    zoom((e.clientX + o.x) / 2, (e.clientY + o.y) / 2,
      s * Math.hypot(e.clientX - o.x, e.clientY - o.y) / (Math.hypot(p.x - o.x, p.y - o.y) || 1));
  }
  p.x = e.clientX;
  p.y = e.clientY;
});
for (const type of ['pointerup', 'pointercancel']) stage.addEventListener(type, e => {
  pts.delete(e.pointerId);
  stage.classList.toggle('held', pts.size > 0);
});

viewer.addEventListener('wheel', e => {
  const d = e.deltaY * (e.deltaMode ? 32 : 1);  // some browsers count lines, not pixels
  if (unrolled()) {
    if (Math.abs(e.deltaY) < Math.abs(e.deltaX)) return;  // a sideways swipe scrolls natively
    viewer.scrollLeft += d;  // wheel down reads on, to the right
  } else if (stage.contains(e.target)) {
    zoom(e.clientX, e.clientY, s * Math.exp(-d * (e.ctrlKey ? 0.01 : 0.002)));
  } else return;
  e.preventDefault();
}, {passive: false});
stage.addEventListener('dblclick', e => {
  if (!unrolled()) zoom(e.clientX, e.clientY, Math.abs(s - one()) < 1e-3 ? fit : one(), true);
});
// Safari's trackpad pinch arrives as gesture events, not as ctrl+wheel
stage.addEventListener('gesturestart', e => { e.preventDefault(); g = s; });
stage.addEventListener('gesturechange', e => {
  e.preventDefault();
  if (!unrolled() && pts.size < 2) zoom(e.clientX, e.clientY, g * e.scale);
});
