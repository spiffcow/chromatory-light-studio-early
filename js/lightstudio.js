// Chromatory Light Studio (Architecture v2 §26b).
// Client-side WebGL renderer for STL light-placement reference — especially NMM painting.
// Driven imperatively from Blazor via JS interop; the service never renders anything.
import * as THREE from 'three';
import { STLLoader } from '../lib/three/STLLoader.js';
import { OrbitControls } from '../lib/three/OrbitControls.js';
import { RoomEnvironment } from '../lib/three/RoomEnvironment.js';
import { TransformControls } from '../lib/three/TransformControls.js';

let renderer, scene, camera, controls, canvasEl, hemi, ground;
let axisControl = null;      // translate gizmo (X/Y/Z arrows) for the selected light
let selectedLightId = null;
let mesh = null;
let modelScale = 1;          // units/mm applied to the primary model; reused so imported STLs keep relative size

// Imported STLs (scene props) — extra models brought in to light a composed scene.
let propControl = null;      // move/rotate gizmo for the selected prop
const props = new Map();     // id → { mesh, name }
let selectedPropId = null;
let propSeq = 0;

// Occlusion: whether the model blocks light. On by default; a toggle exists because cube
// shadow maps over millions of triangles can strain weak GPUs (the standalone runs anywhere).
let shadowsEnabled = true;

/// Re-renders the shadow maps on the next frame (they are frozen between scene changes).
function requestShadowUpdate() {
    if (renderer && shadowsEnabled) renderer.shadowMap.needsUpdate = true;
}

export function setShadows(on) {
    captureUndo('shadows', false);
    shadowsEnabled = !!on;
    if (!renderer) return;
    renderer.shadowMap.enabled = shadowsEnabled;
    renderer.shadowMap.needsUpdate = true;
    // Toggling the shadow system requires shader recompilation on lit materials.
    if (mesh) mesh.material.needsUpdate = true;
    if (ground) ground.material.needsUpdate = true;
}

export function getShadows() {
    return shadowsEnabled;
}

/// Marks a primary light as an occluded caster (bounce/mirror lights stay unoccluded fakes).
/// Point lights get 512px cube faces: a point shadow is SIX full-scene renders, and cube maps
/// at 1024 stack into WebGL context loss once a rig holds several casters on an 8 GB card.
/// `mapSizeOverride` lets secondary casters (cluster members, far tints) take smaller maps so a
/// composite emitter's total VRAM matches a single plain caster. `nearOverride` lets sized
/// emitters skip occluders inside their own source volume (an orb held IN a hand must light the
/// scene from its surface, not be swallowed by the palm around its centre).
function configureShadowCaster(light, mapSizeOverride = null, nearOverride = null) {
    light.castShadow = true;
    const mapSize = mapSizeOverride ?? (light.isPointLight ? 512 : 1024);
    light.shadow.mapSize.set(mapSize, mapSize);
    light.shadow.bias = -0.0015;
    light.shadow.normalBias = 1.0; // dense print meshes shadow-acne badly without this
    if (light.isDirectionalLight) {
        const s = 160;
        light.shadow.camera.left = -s;
        light.shadow.camera.right = s;
        light.shadow.camera.top = s;
        light.shadow.camera.bottom = -s;
        light.shadow.camera.near = 1;
        light.shadow.camera.far = 800;
    }
    else {
        light.shadow.camera.near = nearOverride ?? 2;
        light.shadow.camera.far = 800;
    }
    // Assigning frustum fields does NOT rebuild the camera's projection matrix — without this the
    // shadow pass kept rendering through the DEFAULT frustum (directional: a ±5-unit ortho box on
    // an 80-unit model), so casters existed but occlusion never landed anywhere visible.
    light.shadow.camera.updateProjectionMatrix();
}

// Point-shadow BUDGET. Every cube shadow map is one fragment texture unit, and the lit material
// itself uses several — a rig of sized/gradient lights whose sub-lights ALL cast blew past
// MAX_TEXTURE_IMAGE_UNITS(16): the program failed validation and the model rendered black.
// Candidates carry userData.shadowRank (0 = an emitter's primary caster, 1 = far tints,
// 3+ = extra cluster members); after any light change the budget keeps the best six and demotes
// the rest, so quality degrades gracefully instead of the shader dying.
const MAX_POINT_CASTERS = 6;

function enforceShadowBudget() {
    const candidates = [];
    for (const entry of lights.values())
        for (const light of entry.lights ?? [])
            if (light.isPointLight && light.userData.shadowRank !== undefined) candidates.push(light);
    candidates.sort((a, b) => a.userData.shadowRank - b.userData.shadowRank);
    candidates.forEach((light, i) => {
        const cast = i < MAX_POINT_CASTERS;
        if (light.castShadow !== cast) {
            light.castShadow = cast;
            if (!cast && light.shadow?.map) { light.shadow.map.dispose(); light.shadow.map = null; }
        }
    });
}

// The table the mini stands on: its color paints the visible ground plane AND tints every
// light's bounce (snow, lava, grass, and wood all throw different light up into the model).
// Bounce tint uses the floor's hue/saturation at a lifted luminance, so a realistically dark
// table still produces a usable under-light — strength stays the intensity control.
let floorColor = '#4a4136';

function floorBounceTint() {
    const hsl = {};
    new THREE.Color(floorColor).getHSL(hsl);
    return new THREE.Color().setHSL(hsl.h, hsl.s, Math.max(hsl.l, 0.62));
}

/// Sets the floor color; the ground plane repaints and every bounce light re-tints.
export function setFloor(options) {
    captureUndo('floor', true);
    if (options.color) floorColor = options.color;
    if (ground) ground.material.color.set(floorColor);
    for (const entry of lights.values()) {
        if (entry.bounceStrength > 0) { unmountLights(entry); mountLights(entry); }
    }
}

export function getFloorColor() {
    return floorColor;
}
let modelRadius = 50;
const lights = new Map();   // id -> { light, gizmo, type }
let nextLightId = 1;

// Handles never sink below the table: the ground plane is opaque, so a handle dragged under it
// simply vanishes (the classic "my directional light's dot disappeared"). Lighting from below
// is the bounce system's job anyway.
const HANDLE_MIN_Y = 1.5;
function clampHandle(position) {
    if (position.y < HANDLE_MIN_Y) position.y = HANDLE_MIN_Y;
    return position;
}

// ---------------------------------------------------------------------------------------------
// Undo (snapshot-based): every mutating action first pushes the CURRENT rig state (getSetup
// minus camera — undo restores the rig, not the viewpoint), and Ctrl+Z pops one. Snapshots are
// tiny (a few KB of JSON), so whole-state restore beats per-action inverse bookkeeping.
// ---------------------------------------------------------------------------------------------

const undoStack = [];
const UNDO_LIMIT = 50;
let restoringUndo = false;   // an undo restore must not capture itself
let undoSuppressDepth = 0;   // compound ops (preset, rig load) capture once, not per sub-op
let changeCallback = null;   // notifies the hosting UI of module-initiated changes (keys, undo)

/// cb is either a plain function (standalone) or a Blazor DotNetObjectReference — anything
/// with invokeMethodAsync — whose 'OnStudioChangedFromCanvas' method is invoked.
export function setChangeCallback(cb) {
    changeCallback = cb;
}

function notifyChanged() {
    if (!changeCallback) return;
    if (typeof changeCallback === 'function') changeCallback();
    else changeCallback.invokeMethodAsync('OnStudioChangedFromCanvas');
}

/// coalesce=true collapses bursts of the same tag (slider drags) into one undo step:
/// the first event of the burst captured the pre-state, which is the one worth going back to.
function captureUndo(tag, coalesce) {
    if (restoringUndo || undoSuppressDepth > 0 || !renderer) return;
    const now = performance.now();
    const top = undoStack[undoStack.length - 1];
    if (coalesce && top && top.tag === tag && now - top.time < 1200) { top.time = now; return; }
    const snapshot = JSON.parse(getSetup());
    delete snapshot.camera;
    undoStack.push({ tag, time: now, json: JSON.stringify(snapshot) });
    if (undoStack.length > UNDO_LIMIT) undoStack.shift();
}

function withUndoSuppressed(fn) {
    undoSuppressDepth++;
    try { return fn(); } finally { undoSuppressDepth--; }
}

/// Restores the rig to before the last action. Returns true when something was undone.
export function undo() {
    const top = undoStack.pop();
    if (!top) return false;
    restoringUndo = true;
    try {
        if (top.paintDiff) {
            for (const [i, prev] of top.paintDiff) regionIndex[i] = prev;
            repaintColors();
            refreshRegionLights();
            regionsDirty = true; // undone paint still differs from what's saved
            if (isolated >= 0 && highlightAttr) {
                for (const [i] of top.paintDiff) highlightAttr.setX(i, regionIndex[i] === isolated ? 1 : 0);
                highlightAttr.needsUpdate = true;
            }
            notifyChanged(); // the panel re-reads the dirty state
        } else {
            applySetup(top.json);
        }
    } finally { restoringUndo = false; }
    return true;
}

/// Keyboard: Delete removes the selected light, Ctrl/Cmd+Z undoes. Lives on window so it works
/// wherever focus sits, but stays out of the way while the user types in a field.
function onKeyDown(event) {
    const target = event.target;
    if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' ||
                   target.tagName === 'SELECT' || target.isContentEditable)) return;

    if ((event.ctrlKey || event.metaKey) && !event.shiftKey && event.key.toLowerCase() === 'z') {
        event.preventDefault();
        if (undo()) notifyChanged();
    }
    else if (event.key === 'Delete' || event.key === 'Backspace') {
        if (selectedPropId !== null) {
            event.preventDefault();
            deleteProp(selectedPropId);
            notifyChanged();
        }
        else if (selectedLightId !== null) {
            event.preventDefault();
            removeLight(selectedLightId);
            notifyChanged();
        }
    }
}

// The curated look presets — the SINGLE source of truth consumed by both the in-app studio
// page (via getLooksJson) and the standalone Light Studio. Values tuned on a real 3M-triangle
// print (2026-07). "color" omitted = keep the user's current color when switching looks.
export const LOOKS = {
    // The NMM looks use plain PBR metal (the user's preference after trying the painted-ramp
    // shader). The painterly 'nmm' shader stays fully functional — the materials editor creates
    // painted-NMM materials on demand, and rigs saved with one keep rendering it.
    silvernmm: { label: 'Silver NMM', fixedReference: false,
        appearance: { shader: 'pbr', color: '#c8ccd2', metalness: 1.0, roughness: 0.3, clearcoat: 0, envIntensity: 0.75 } },
    goldnmm: { label: 'Gold NMM', fixedReference: false,
        appearance: { shader: 'pbr', color: '#d9b36c', metalness: 1.0, roughness: 0.3, clearcoat: 0, envIntensity: 0.8 } },
    realmetal: { label: 'Real metal', fixedReference: false,
        appearance: { shader: 'pbr', color: '#d8d8d8', metalness: 1.0, roughness: 0.45, clearcoat: 0, envIntensity: 0.7 } },
    gloss: { label: 'Gloss paint', fixedReference: false,
        appearance: { shader: 'pbr', metalness: 0.0, roughness: 0.18, clearcoat: 0.7, envIntensity: 0.5 } },
    satin: { label: 'Satin paint', fixedReference: false,
        appearance: { shader: 'pbr', metalness: 0.0, roughness: 0.38, clearcoat: 0, envIntensity: 0.25 } },
    matte: { label: 'Matte paint', fixedReference: false,
        appearance: { shader: 'pbr', metalness: 0.0, roughness: 0.85, clearcoat: 0, envIntensity: 0.15 } },
    bands: { label: 'Bands (practice)', fixedReference: false,
        appearance: { shader: 'toon', toonBands: 4 } },
    chrome: { label: 'Chrome', fixedReference: true,
        appearance: { shader: 'matcap', matcap: 'chrome', color: '#ffffff' } },
    gold: { label: 'Gold', fixedReference: true,
        appearance: { shader: 'matcap', matcap: 'gold', color: '#ffffff' } },
    contrast: { label: 'High contrast', fixedReference: true,
        appearance: { shader: 'matcap', matcap: 'highcontrast', color: '#ffffff' } }
};

/// The look presets for non-JS consumers (the Blazor page deserializes this).
export function getLooksJson() {
    return JSON.stringify(LOOKS);
}

// The model's two materials are long-lived and SWAPPED, never rebuilt per paint-mode toggle:
// disposing a material releases its compiled program, so rebuild-per-toggle meant recompiling the
// physical mega-shader (a visible stall on big prints), churned GPU memory, and re-ran the fragile
// fresh-program-first-draw path. lookMaterial rebuilds only when the look itself changes (or the
// region patch must attach); paintMaterial is created once per model.
let lookMaterial = null;
let paintMaterial = null;

// Current appearance state (kept so shader switches preserve color etc., and for getSetup()).
const appearance = {
    shader: 'pbr',          // 'pbr' | 'phong' | 'matcap' | 'toon' | 'nmm'
    matcap: 'chrome',
    nmmMetal: 'steel',      // nmm: which authored ramp ('steel' | 'gold')
    toonBands: 4,
    color: '#c8ccd2',       // Silver NMM default (matches LightStudio.razor's default look)
    specular: '#ffffff',    // phong: hotspot color
    shininess: 90,          // phong: hotspot tightness
    metalness: 1.0,
    roughness: 0.3,         // mid roughness: hotspots read as SHAPES on dense print meshes
    clearcoat: 0.0,
    envIntensity: 0.75      // studio env = the sky/ground body gradient NMM painters copy
};

// ---------------------------------------------------------------------------------------------
// Setup / teardown
// ---------------------------------------------------------------------------------------------

export function init(canvas) {
    canvasEl = canvas;
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
    // Cap the backing store: dpr beyond 2 quadruples fill cost on a 3M-triangle scene for no
    // visible gain, and VRAM headroom matters — the same GPU often hosts the local AI models.
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

    // A lost WebGL context (driver reset, VRAM pressure from local AI) otherwise leaves the
    // canvas permanently blank. preventDefault opts into restoration; on restore, rebuild the
    // pieces that live in GPU-only render targets (the PMREM environment) and refresh materials.
    canvas.addEventListener('webglcontextlost', event => {
        event.preventDefault();
        console.warn('lightstudio: WebGL context lost — waiting for restore');
    });
    canvas.addEventListener('webglcontextrestored', () => {
        console.warn('lightstudio: WebGL context restored — rebuilding environment');
        buildEnvironment();
        matcapCache.clear();
        paintMaterial?.dispose();
        paintMaterial = null;
        if (mesh) {
            rebuildLookMaterial();
            if (paintMode) mesh.material = ensurePaintMaterial();
        }
        if (ground) ground.material.needsUpdate = true;
        requestShadowUpdate();
    });

    // ACES filmic: lifts midtones and rolls off hot speculars, so exaggerated metals read as
    // bright shapes instead of blown streaks. Matcap looks opt out (toneMapped=false) — they
    // are pre-authored reference images. Tuned on a real 3M-triangle print (2026-07).
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;

    // Shadows: the model blocks light (a lantern behind the body must not light the face).
    // Maps re-render only when the scene actually changes (requestShadowUpdate) — shadowing a
    // 3M-triangle print every frame would not be interactive on typical GPUs.
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.shadowMap.autoUpdate = false;

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x101315);

    camera = new THREE.PerspectiveCamera(45, 1, 0.1, 5000);
    camera.position.set(0, 60, 160);

    controls = new OrbitControls(camera, canvas);
    controls.enableDamping = true;
    // Zoom bounds: past minDistance the camera enters the sculpt and the near plane clips
    // everything away — which reads as the render "going blank".
    controls.minDistance = 15;
    controls.maxDistance = 1200;

    // Precise placement: clicking a light handle shows X/Y/Z arrows; dragging an arrow moves
    // the light along that axis only. Free drag on the ball itself still works.
    axisControl = new TransformControls(camera, canvas);
    axisControl.setMode('translate');
    axisControl.setSize(0.8);
    axisControl.addEventListener('dragging-changed', event => {
        controls.enabled = !event.value;
        if (event.value) captureUndo('arrow', false); // each arrow grab is one undo step
    });
    axisControl.addEventListener('objectChange', () => {
        const entry = selectedLightId !== null ? lights.get(selectedLightId) : null;
        if (entry) {
            clampHandle(entry.gizmo.position);
            syncLightPositions(entry);
        }
    });
    scene.add(axisControl);

    // A second gizmo for imported STLs — move (translate) or turn (rotate) the selected prop.
    propControl = new TransformControls(camera, canvas);
    propControl.setSize(0.9);
    propControl.addEventListener('dragging-changed', event => {
        controls.enabled = !event.value;
        if (!event.value) notifyChanged(); // on release, the panel debounce-saves the scene
    });
    propControl.addEventListener('objectChange', () => requestShadowUpdate());
    scene.add(propControl);

    // A soft sky/ground fill so unlit faces read as shape, never pure black. Hemisphere rather
    // than flat ambient: the subtle top-vs-bottom difference keeps form readable on the dark
    // NMM body without competing with the user's placed lights.
    hemi = new THREE.HemisphereLight(0x8fa3b4, 0x4a4238, 0.18);
    scene.add(hemi);

    // Neutral studio environment: PBR metals are mostly reflection — without SOMETHING to
    // reflect they render near-black no matter where the lights sit. Kept subdued so the
    // user's own lights stay the story; matcap/toon looks ignore it entirely.
    buildEnvironment();

    ground = new THREE.Mesh(
        new THREE.CircleGeometry(500, 48),
        applyStudioShaderPatches(
            new THREE.MeshStandardMaterial({ color: new THREE.Color(floorColor), roughness: 0.95 })));
    ground.rotation.x = -Math.PI / 2;
    ground.name = 'ground';
    ground.receiveShadow = true; // the model's cast shadow on the table reads light direction
    scene.add(ground);           // (never castShadow: bounce lights live beneath it)

    resize();
    window.addEventListener('resize', resize);
    window.addEventListener('keydown', onKeyDown);
    canvas.addEventListener('pointerdown', onPointerDown);
    canvas.addEventListener('pointermove', onPointerMove);
    canvas.addEventListener('pointerup', onPointerUp);
    canvas.addEventListener('pointerleave', () => {
        if (brushCursor) brushCursor.visible = false;
        hideFillPreview();
    });
    // Right-drag erases in paint mode — the browser menu would swallow the gesture.
    canvas.addEventListener('contextmenu', event => event.preventDefault());

    renderer.setAnimationLoop(() => {
        processPendingPaint();
        controls.update();
        syncStudioUniforms();
        renderer.render(scene, camera);
    });
}

/// (Re)builds the PMREM studio environment — also called after a WebGL context restore,
/// because prefiltered environments live in render targets that do not survive the loss.
function buildEnvironment() {
    scene.environment?.dispose(); // free the previous prefiltered env before replacing it
    const pmrem = new THREE.PMREMGenerator(renderer);
    scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    pmrem.dispose();
}

export function dispose() {
    renderer?.setAnimationLoop(null);
    window.removeEventListener('resize', resize);
    window.removeEventListener('keydown', onKeyDown);

    // Free every GPU resource the scene holds (renderer.dispose alone does NOT) — the model geometry
    // can be hundreds of MB, so leaking it across open/close cycles is how the studio runs out of memory.
    if (mesh) { mesh.geometry.dispose(); mesh = null; }
    lookMaterial?.dispose();
    lookMaterial = null;
    paintMaterial?.dispose();
    paintMaterial = null;
    resetRegions(); // disposes the region-highlight overlay
    if (brushCursor) { brushCursor.geometry.dispose(); brushCursor.material.dispose(); brushCursor = null; }
    for (const entry of lights.values()) {
        unmountLights(entry); // disposes each light's shadow map
        entry.gizmo.geometry.dispose();
        entry.gizmo.material.dispose();
    }
    lights.clear();
    for (const entry of props.values()) { entry.mesh.geometry.dispose(); entry.mesh.material.dispose(); }
    props.clear();
    scene?.environment?.dispose();
    renderer?.dispose();
    selectedPropId = null;
    undoStack.length = 0;
    changeCallback = null;
}

function resize() {
    if (!canvasEl || !renderer) return;
    const w = canvasEl.clientWidth || 800;
    const h = canvasEl.clientHeight || 600;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();

    // OrbitControls rotation is normalized by canvas HEIGHT (a full-height drag = one turn),
    // which makes big canvases feel sluggish and small ones twitchy. Scale rotateSpeed with
    // height so the feel is constant: ~65° per 100px dragged on any screen.
    if (controls) controls.rotateSpeed = Math.min(2.5, Math.max(0.7, h / 550));
}

// ---------------------------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------------------------

// STLs are a triangle soup — no shared vertices — so computeVertexNormals can only produce FLAT
// facet normals. Dense prints hide that, but the studio-LOD decimation leaves a small feature
// (a gemstone, a rivet) only a handful of large triangles, and the high-gradient looks (the gem
// window, sharp speculars) make the facets glaring. Weld normals across bit-identical positions,
// area-weighted, keeping edges sharper than the crease angle hard. Everything happens IN PLACE:
// vertex count and order are untouched, so saved region masks (keyed to position.count) and the
// paint/fill topology stay valid.
function smoothSoupNormals(geometry, creaseDeg = 60) {
    const pos = geometry.attributes.position;
    const n = pos.count;
    // The cap is a MEMORY/load-time guard only, never a visual judgement — zoomed in, facets show
    // at any density (a 1.4M-triangle studio mesh still facets on a gemstone filling the screen).
    // The typed-array weld below handles studio-size meshes in well under a second.
    if (!n || n % 3 !== 0 || n > 15_000_000) { geometry.computeVertexNormals(); return; }

    const a = pos.array;
    const triCount = n / 3;
    const fn = new Float32Array(triCount * 3);   // area-weighted face normal (unnormalized cross)
    const fnu = new Float32Array(triCount * 3);  // unit face normal, for the crease test
    for (let t = 0; t < triCount; t++) {
        const i = t * 9;
        const ux = a[i + 3] - a[i], uy = a[i + 4] - a[i + 1], uz = a[i + 5] - a[i + 2];
        const vx = a[i + 6] - a[i], vy = a[i + 7] - a[i + 1], vz = a[i + 8] - a[i + 2];
        const cx = uy * vz - uz * vy, cy = uz * vx - ux * vz, cz = ux * vy - uy * vx;
        fn[t * 3] = cx; fn[t * 3 + 1] = cy; fn[t * 3 + 2] = cz;
        const l = Math.sqrt(cx * cx + cy * cy + cz * cz) || 1;
        fnu[t * 3] = cx / l; fnu[t * 3 + 1] = cy / l; fnu[t * 3 + 2] = cz / l;
    }

    // Weld corners sharing bit-identical positions (STL duplicates are exact copies). All typed
    // arrays — a string-keyed Map dies of GC at millions of vertices. Open hash over the position
    // bits; only group LEADERS sit in the collision chains, members hang off their leader.
    const bits = new Int32Array(a.buffer, a.byteOffset, n * 3);
    let cap = 1; while (cap < n * 2) cap <<= 1;
    const hmask = cap - 1;
    const head = new Int32Array(cap).fill(-1);  // hash slot → first leader in the chain
    const hnext = new Int32Array(n);            // leader → next leader in the same slot
    const mfirst = new Int32Array(n);           // leader → first member of its group
    const mnext = new Int32Array(n);            // member → next member (−1 ends)
    for (let i = 0; i < n; i++) {
        const x = bits[i * 3], y = bits[i * 3 + 1], z = bits[i * 3 + 2];
        const h = (Math.imul(x, 73856093) ^ Math.imul(y, 19349663) ^ Math.imul(z, 83492791)) & hmask;
        let leader = -1;
        for (let j = head[h]; j !== -1; j = hnext[j]) {
            if (bits[j * 3] === x && bits[j * 3 + 1] === y && bits[j * 3 + 2] === z) { leader = j; break; }
        }
        if (leader === -1) {
            hnext[i] = head[h]; head[h] = i;    // new leader
            mfirst[i] = i; mnext[i] = -1;
        } else {
            mnext[i] = mfirst[leader]; mfirst[leader] = i;  // prepend to the group
            mfirst[i] = -2;                      // mark as non-leader
        }
    }

    const cosCrease = Math.cos(creaseDeg * Math.PI / 180);
    const out = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
        if (mfirst[i] === -2) continue;          // members are written when their leader is visited
        for (let c = mfirst[i]; c !== -1; c = mnext[c]) {
            const t0 = ((c / 3) | 0) * 3;
            let x = 0, y = 0, z = 0;
            for (let c2 = mfirst[i]; c2 !== -1; c2 = mnext[c2]) {
                const t2 = ((c2 / 3) | 0) * 3;
                // Average only faces on this corner's smoothing side of the crease.
                if (fnu[t0] * fnu[t2] + fnu[t0 + 1] * fnu[t2 + 1] + fnu[t0 + 2] * fnu[t2 + 2] > cosCrease) {
                    x += fn[t2]; y += fn[t2 + 1]; z += fn[t2 + 2];
                }
            }
            const l = Math.sqrt(x * x + y * y + z * z) || 1;
            out[c * 3] = x / l; out[c * 3 + 1] = y / l; out[c * 3 + 2] = z / l;
        }
    }
    geometry.setAttribute('normal', new THREE.BufferAttribute(out, 3));
}

export function loadStl(bytes) {
    if (mesh) { scene.remove(mesh); mesh.geometry.dispose(); }
    resetRegions(); // a new geometry invalidates any painted mask (before the material rebuild reads attr state)
    lookMaterial?.dispose();
    lookMaterial = null;
    paintMaterial?.dispose();
    paintMaterial = null;

    const geometry = new STLLoader().parse(bytes.buffer ?? bytes);
    smoothSoupNormals(geometry); // welded, crease-aware normals — see above
    geometry.center();

    // STLs come in arbitrary units/orientation; many are Z-up. Normalise: Y-up, ~80 units tall,
    // sitting on the ground plane.
    geometry.rotateX(-Math.PI / 2);
    geometry.computeBoundingBox();
    const size = new THREE.Vector3();
    geometry.boundingBox.getSize(size);
    const scale = 80 / Math.max(size.x, size.y, size.z);
    modelScale = scale; // imported props scale by the same factor so their size stays relative
    geometry.scale(scale, scale, scale);
    geometry.computeBoundingBox();
    geometry.translate(0, -geometry.boundingBox.min.y, 0);
    geometry.computeBoundingSphere();
    modelRadius = geometry.boundingSphere.radius;

    paintMode = false; // a fresh model starts on the lit look
    mesh = new THREE.Mesh(geometry, rebuildLookMaterial());
    mesh.name = 'model';
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    scene.add(mesh);
    orientation.identity();
    requestShadowUpdate();

    controls.target.set(0, modelRadius * 0.6, 0);
    camera.position.set(0, modelRadius * 1.1, modelRadius * 2.6);

    if (lights.size === 0) applyPreset('keyrim');
    return geometry.attributes.position.count / 3; // triangle count
}

// ---------------------------------------------------------------------------------------------
// Orientation — many print STLs arrive sideways (posed for supports, or a different up-axis
// than the loader's assumption). Quarter-turn steps around world axes, always re-seated on the
// ground; the cumulative orientation persists in rigs so a saved study reloads as it looked.
// ---------------------------------------------------------------------------------------------

const orientation = new THREE.Quaternion();

/// Rotates the model a quarter turn around a world axis: 'x' tips forward/back, 'y' spins,
/// 'z' rolls. quarterTurns is +1 or -1.
export function reorient(axis, quarterTurns) {
    if (!mesh) return;
    captureUndo('orient', false);
    const axes = { x: new THREE.Vector3(1, 0, 0), y: new THREE.Vector3(0, 1, 0), z: new THREE.Vector3(0, 0, 1) };
    const step = new THREE.Quaternion().setFromAxisAngle(axes[axis], quarterTurns * Math.PI / 2);
    applyOrientationDelta(step);
}

/// Back to the orientation the model loaded with.
export function resetOrientation() {
    if (!mesh) return;
    captureUndo('orient', false);
    applyOrientationDelta(orientation.clone().invert());
}

function applyOrientationDelta(delta) {
    const geometry = mesh.geometry;

    // Rotate about the model's current center so it pivots in place…
    geometry.computeBoundingBox();
    const center = new THREE.Vector3();
    geometry.boundingBox.getCenter(center);
    geometry.translate(-center.x, -center.y, -center.z);
    geometry.applyQuaternion(delta);

    // …then re-seat it on the ground, centered.
    geometry.computeBoundingBox();
    geometry.translate(
        -(geometry.boundingBox.min.x + geometry.boundingBox.max.x) / 2,
        -geometry.boundingBox.min.y,
        -(geometry.boundingBox.min.z + geometry.boundingBox.max.z) / 2);
    geometry.computeBoundingSphere(); // normals rotate with applyQuaternion; no recompute needed
    modelRadius = geometry.boundingSphere.radius;
    paintGrid = null; // vertex positions moved — the brush grid must be rebuilt

    orientation.premultiply(delta);
    requestShadowUpdate();
}

// ---------------------------------------------------------------------------------------------
// Materials & shaders
// ---------------------------------------------------------------------------------------------

const matcapCache = new Map();

// Per-metal matcap recipes, authored in DISK space (see matcapTexture): a bright sky band up
// top, a dark ground band below, a sharp horizon between them, and a punchy specular hotspot —
// the bold, high-contrast reflection NMM painters actually copy. horizon in [-1,1] sets where
// sky meets ground (negative = more sky visible); horizonSoft its sharpness.
const MATCAPS = {
    chrome: {
        skyTop: [0.97, 0.99, 1.0], skyHorizon: [0.42, 0.52, 0.66],
        groundHorizon: [0.02, 0.03, 0.05], groundBottom: [0.18, 0.21, 0.27],
        horizon: -0.05, horizonSoft: 0.035, spec: 1.0, specPower: 70, bounce: 0.35, edge: 0.4
    },
    steel: {
        skyTop: [0.84, 0.87, 0.9], skyHorizon: [0.40, 0.44, 0.5],
        groundHorizon: [0.05, 0.06, 0.08], groundBottom: [0.20, 0.22, 0.26],
        horizon: -0.02, horizonSoft: 0.08, spec: 0.85, specPower: 34, bounce: 0.25, edge: 0.35
    },
    gold: {
        skyTop: [1.0, 0.94, 0.68], skyHorizon: [0.68, 0.46, 0.12],
        groundHorizon: [0.10, 0.05, 0.01], groundBottom: [0.42, 0.27, 0.06],
        horizon: -0.05, horizonSoft: 0.04, spec: 1.0, specPower: 60, bounce: 0.4,
        specColor: [1.0, 0.97, 0.82], edge: 0.4
    },
    bronze: {
        skyTop: [0.94, 0.74, 0.5], skyHorizon: [0.5, 0.31, 0.15],
        groundHorizon: [0.08, 0.04, 0.02], groundBottom: [0.3, 0.18, 0.08],
        horizon: -0.03, horizonSoft: 0.06, spec: 0.85, specPower: 40, bounce: 0.3,
        specColor: [1.0, 0.9, 0.72], edge: 0.38
    },
    highcontrast: {
        skyTop: [1.0, 1.0, 1.0], skyHorizon: [0.5, 0.5, 0.5],
        groundHorizon: [0.0, 0.0, 0.0], groundBottom: [0.12, 0.12, 0.12],
        horizon: 0.0, horizonSoft: 0.02, spec: 1.0, specPower: 90, bounce: 0.2, edge: 0.45
    }
};

// Sphere-shaded matcap in DISK space: the surface normal's screen y maps up-facing pixels to
// the top of the disk (sky) and down-facing to the bottom (ground). This authored parameter-
// isation reads far bolder on dense sculpts than a physically-reflected one, which collapses
// most normals into a muddy mid-tone. Key specular lobe upper-left, small fill lower-right.
function matcapTexture(name) {
    if (matcapCache.has(name)) return matcapCache.get(name);
    const texture = generateMatcap(MATCAPS[name] ?? MATCAPS.chrome);
    matcapCache.set(name, texture);
    return texture;
}

/// A custom matcap recipe (hex colours) → texture, cached by content. Matcap canvases hold raw
/// sRGB bytes, so hex converts by simple byte split — no linear round-trip.
function matcapTextureFromDef(def) {
    const key = JSON.stringify(def);
    if (matcapCache.has(key)) return matcapCache.get(key);
    const toRaw = (hex, fallback) => {
        if (Array.isArray(hex)) return hex;
        if (typeof hex !== 'string') return fallback;
        const n = parseInt(hex.replace('#', ''), 16);
        return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
    };
    const base = MATCAPS.chrome;
    const p = {
        skyTop: toRaw(def.skyTop, base.skyTop), skyHorizon: toRaw(def.skyHorizon, base.skyHorizon),
        groundHorizon: toRaw(def.groundHorizon, base.groundHorizon), groundBottom: toRaw(def.groundBottom, base.groundBottom),
        horizon: def.horizon ?? base.horizon, horizonSoft: def.horizonSoft ?? base.horizonSoft,
        spec: def.spec ?? base.spec, specPower: def.specPower ?? base.specPower,
        bounce: def.bounce ?? base.bounce, edge: def.edge ?? base.edge,
        specColor: def.specColor !== undefined ? toRaw(def.specColor, [1, 1, 1]) : undefined
    };
    const texture = generateMatcap(p);
    matcapCache.set(key, texture);
    return texture;
}

function generateMatcap(p) {
    const size = 256;
    const c = document.createElement('canvas');
    c.width = c.height = size;
    const g = c.getContext('2d');
    const image = g.createImageData(size, size);
    const data = image.data;

    const key = normalize([-0.4, 0.5, 0.75]);
    const fill = normalize([0.45, -0.3, 0.84]);
    const specColor = p.specColor ?? [1, 1, 1];
    const horizonSoft = p.horizonSoft ?? 0.05;
    const edgeAmt = p.edge ?? 0.35;
    const smooth = (a, b, x) => {
        const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
        return t * t * (3 - 2 * t);
    };

    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            const nx = (x + 0.5) / (size / 2) - 1;
            const ny = 1 - (y + 0.5) / (size / 2);
            const r2 = nx * nx + ny * ny;
            const i = (y * size + x) * 4;
            if (r2 > 1) { data[i + 3] = 255; continue; }
            const nz = Math.sqrt(1 - r2);

            // Disk-space vertical: ny in [-1,1], up = sky, down = ground, split at the horizon.
            const sky = smooth(p.horizon - horizonSoft, p.horizon + horizonSoft, ny);
            const skyPos = smooth(p.horizon, 1, ny);
            const groundPos = smooth(p.horizon, -1, ny);
            const base = [0, 0, 0];
            for (let ch = 0; ch < 3; ch++) {
                const skyCol = p.skyHorizon[ch] + (p.skyTop[ch] - p.skyHorizon[ch]) * skyPos;
                const groundCol = p.groundHorizon[ch] + (p.groundBottom[ch] - p.groundHorizon[ch]) * groundPos;
                base[ch] = groundCol + (skyCol - groundCol) * sky;
            }

            const dKey = Math.max(0, nx * key[0] + ny * key[1] + nz * key[2]);
            const dFill = Math.max(0, nx * fill[0] + ny * fill[1] + nz * fill[2]);
            const specKey = Math.pow(dKey, p.specPower) * p.spec;
            const specFill = Math.pow(dFill, p.specPower * 1.5) * p.bounce;
            const edge = Math.pow(1 - nz, 3) * edgeAmt; // darken the silhouette

            for (let ch = 0; ch < 3; ch++) {
                let v = base[ch] * (1 - edge) + (specKey + specFill) * specColor[ch];
                data[i + ch] = Math.round(255 * Math.min(1, Math.max(0, v)));
            }
            data[i + 3] = 255;
        }
    }

    g.putImageData(image, 0, 0);
    const texture = new THREE.CanvasTexture(c);
    texture.colorSpace = THREE.SRGBColorSpace;
    return texture; // callers cache (by preset name or by def content)
}

function normalize(v) {
    const l = Math.hypot(v[0], v[1], v[2]);
    return [v[0] / l, v[1] / l, v[2] / l];
}

// ---------------------------------------------------------------------------------------------
// Painterly NMM ramps — what "really well done NMM" actually is: the DIFFUSE response is an
// authored, hue-shifting colour ramp (gold shades through chocolate → rust → orange → yellow →
// ivory; steel through blue-black → slate → steel → white), never one hue scaled by light. The
// ramp is driven by the accumulated light so it still FOLLOWS the user's placed lights (the
// studio's whole point), with a re-thresholded crisp hotspot and a painter's silhouette lining.
// Tuned against 'Eavy Metal-grade reference photos (2026-07).
// ---------------------------------------------------------------------------------------------
// The light→ramp mapping is a Reinhard knee (t = x/(x+knee)) rather than a linear clamp: a
// linear mapping was tuned to ONE scene's light level and collapsed to all-shadow (or all-top)
// under the user's own rigs. The knee keeps mids in the middle across a wide brightness range;
// gain rides as a LIVE uniform behind the Response slider for per-rig taste.
const NMM_RAMPS = {
    steel: {
        stops: [
            [0.00, [0.043, 0.051, 0.086]],  // blue-black lining
            [0.22, [0.118, 0.145, 0.212]],  // dark slate
            [0.50, [0.345, 0.388, 0.463]],  // mid steel, blue-grey
            [0.80, [0.663, 0.729, 0.831]],  // pale steel
            [1.00, [0.910, 0.949, 1.000]]   // sky-white
        ],
        spec: [1.0, 1.0, 1.0], specLo: 0.10, specHi: 0.30, edge: 0.55, gain: 1.0, knee: 0.40, skyDown: 0.12, contrast: 1.45
    },
    gold: {
        stops: [
            [0.00, [0.078, 0.031, 0.016]],  // chocolate-brown lining
            [0.22, [0.355, 0.145, 0.035]],  // sienna brown
            [0.50, [0.718, 0.373, 0.055]],  // orange gold
            [0.80, [0.945, 0.718, 0.235]],  // yellow gold
            [1.00, [1.000, 0.933, 0.690]]   // pale gold
        ],
        spec: [1.0, 0.97, 0.87], specLo: 0.10, specHi: 0.30, edge: 0.50, gain: 1.0, knee: 0.40, skyDown: 0.12, contrast: 1.45
    }
};

// Gem-shader knobs (region gems light through uGemPow/uGemGain uniforms). Editable through the
// materials editor; ride in `appearance` (gemPower/gemGain) so rigs save them.
const gemParams = { power: 1.35, gain: 2.4 };

function syncGemParams() {
    // Global gem defaults are baked into the per-vertex attribute (slots without their own
    // values inherit them), so a change refills the buffer — a few ms even on dense prints.
    repaintColors();
}

/// A custom NMM ramp def (hex colours, editor-friendly) → the linear-space numbers the patch
/// bakes. Colours pass through THREE.Color so the sRGB→linear conversion matches the renderer.
function resolveNmmDef() {
    const custom = appearance.nmmRamp;
    if (!custom || !Array.isArray(custom.stops) || custom.stops.length < 2)
        return { def: NMM_RAMPS[appearance.nmmMetal] ?? NMM_RAMPS.steel, key: `builtin-${appearance.nmmMetal}` };
    const toLin = hex => { const c = new THREE.Color(hex); return [c.r, c.g, c.b]; };
    const def = {
        stops: custom.stops.map(s => [Number(s[0]) || 0, Array.isArray(s[1]) ? s[1] : toLin(s[1])]),
        spec: Array.isArray(custom.spec) ? custom.spec : toLin(custom.spec ?? '#ffffff'),
        specLo: custom.specLo ?? 0.10, specHi: custom.specHi ?? 0.30,
        edge: custom.edge ?? 0.5, gain: custom.gain ?? 1.0, knee: custom.knee ?? 0.40,
        skyDown: custom.skyDown ?? 0.12, contrast: custom.contrast ?? 1.45
    };
    return { def, key: JSON.stringify(custom) };
}

/// Turns a Phong material into the painterly-NMM shader: light intensity → authored ramp,
/// specular → thresholded hotspot, silhouette → dark lining. Chains AFTER the studio patches
/// (lock-specular keeps working: it redirects the view vector the Phong specular term uses).
function applyNmmRampPatch(material, p, cacheKey, hasRegions = false) {
    const f = n => n.toFixed(4);
    const vec3 = c => `vec3(${f(c[0])}, ${f(c[1])}, ${f(c[2])})`;
    const prevCompile = material.onBeforeCompile;
    material.onBeforeCompile = shader => {
        if (prevCompile) prevCompile(shader);
        shader.uniforms.uNmmGain = { value: p.gain * (appearance.nmmGain ?? 1) };
        shader.uniforms.uNmmContrast = { value: p.contrast * (appearance.nmmContrast ?? 1) };
        shader.uniforms.uNmmTint = { value: new THREE.Color(appearance.color ?? '#ffffff') };
        shader.fragmentShader = shader.fragmentShader
            .replace('void main() {', 'uniform float uNmmGain;\nuniform float uNmmContrast;\nuniform vec3 uNmmTint;\nvoid main() {')
            .replace('#include <opaque_fragment>', `
            {
                const vec3 NMM_LUMA = vec3(0.2126, 0.7152, 0.0722);
                // Direct light carries the form; ambient fill is down-weighted so it lifts the
                // shadows a little without flattening the whole ramp toward one value.
                float nmmX = (dot(reflectedLight.directDiffuse, NMM_LUMA)
                            + 0.45 * dot(reflectedLight.indirectDiffuse, NMM_LUMA)) * uNmmGain;
                // Reinhard knee: scene-brightness robust — mids stay mids whether the rig is a
                // candle or a floodlight, instead of the whole model sliding to one ramp end.
                float nmmT = nmmX / (nmmX + ${f(p.knee)});
                // Sky-down: up-facing surfaces brighten, down-facing darken — the vertical logic
                // painters grade EVERY plate with, so flat-lit plates still carry a gradient.
                vec3 nmmWorldN = inverseTransformDirection(normal, viewMatrix);
                nmmT = pow(clamp(nmmT + nmmWorldN.y * ${f(p.skyDown)}, 0.0, 1.0), uNmmContrast);
                vec3 nmmC = mix(${vec3(p.stops[0][1])}, ${vec3(p.stops[1][1])}, smoothstep(${f(p.stops[0][0])}, ${f(p.stops[1][0])}, nmmT));
                nmmC = mix(nmmC, ${vec3(p.stops[2][1])}, smoothstep(${f(p.stops[1][0])}, ${f(p.stops[2][0])}, nmmT));
                nmmC = mix(nmmC, ${vec3(p.stops[3][1])}, smoothstep(${f(p.stops[2][0])}, ${f(p.stops[3][0])}, nmmT));
                nmmC = mix(nmmC, ${vec3(p.stops[4][1])}, smoothstep(${f(p.stops[3][0])}, ${f(p.stops[4][0])}, nmmT));
                float nmmNdV = clamp(dot(normal, normalize(vViewPosition)), 0.0, 1.0);
                // Painter's lining: silhouettes fall to the shadow colour instead of catching light.
                nmmC = mix(nmmC, ${vec3(p.stops[0][1])}, pow(1.0 - nmmNdV, 3.0) * ${f(p.edge)});
                // The colour picker TINTS the ramp (white = the authored metal untouched).
                nmmC *= uNmmTint;
                // The crisp painted hotspot: re-threshold the specular so it reads as a deliberate
                // mark — and mask it away from grazing angles, where the physical Fresnel term
                // otherwise smears it into a satin sheen along silhouettes.
                float nmmSpec = smoothstep(${f(p.specLo)}, ${f(p.specHi)}, dot(reflectedLight.directSpecular, NMM_LUMA))
                              * smoothstep(0.30, 0.55, nmmNdV);
                vec3 nmmOut = nmmC + nmmSpec * ${vec3(p.spec)};
                ${hasRegions
                    // Painted regions keep the standard physical result (their tint, per-region
                    // surface, gem behaviour); the ramp claims only unpainted fragments, with the
                    // soft region boundary blending between the two.
                    ? 'outgoingLight = mix(nmmOut, outgoingLight, smoothstep(0.0, 0.5, vRegionSurf.w));'
                    : 'outgoingLight = nmmOut;'}
            }
            #include <opaque_fragment>`);
        material.userData.nmmShader = shader;
    };
    const prevKey = material.customProgramCacheKey?.();
    material.customProgramCacheKey = () => `${prevKey ?? ''}|nmm-${cacheKey}`;
    return material;
}

const toonCache = new Map();
function toonGradient(bands) {
    // Cached like the matcaps: rebuilt materials would otherwise leak a DataTexture each time (a
    // disposed material does not dispose its gradientMap), and there are only a few band counts.
    if (toonCache.has(bands)) return toonCache.get(bands);
    const data = new Uint8Array(bands);
    for (let i = 0; i < bands; i++) data[i] = Math.round((i / (bands - 1)) * 255);
    const texture = new THREE.DataTexture(data, bands, 1, THREE.RedFormat);
    texture.needsUpdate = true;
    texture.minFilter = texture.magFilter = THREE.NearestFilter;
    toonCache.set(bands, texture);
    return texture;
}

// ---------------------------------------------------------------------------------------------
// Studio shader patches — lock-specular and painterly shadows.
//
// Lock specular: painted highlights don't move — NMM is painted from ONE chosen viewpoint.
// Locking captures the camera position and substitutes it into the specular/reflection view
// direction (diffuse shading has no view term and stays live), so the painter can orbit to
// reach the far side of the mini while the hotspots stay where they will be painted.
//
// Shadow tint: painters shade toward a COLOUR (usually cool blue-violet), almost never black.
// Strength backs the occlusion off; tint is what the light fades to where it is blocked.
// Black at full strength is bit-identical to the stock behavior.
// ---------------------------------------------------------------------------------------------

const specLock = { on: false, camPos: new THREE.Vector3() };
let shadowTintColor = '#000000';
let shadowStrength = 1.0;

const _lockedCamView = new THREE.Vector3();

/// Patches a lit material's lights chunk. Applied to the model and the ground — the only lit
/// materials in the scene (gizmos and arrows are unlit).
function applyStudioShaderPatches(material) {
    material.onBeforeCompile = shader => {
        shader.uniforms.uSpecLock = { value: specLock.on ? 1 : 0 };
        shader.uniforms.uLockedCamPosView = { value: new THREE.Vector3() };
        shader.uniforms.uShadowTint = { value: new THREE.Color(shadowTintColor) };
        shader.uniforms.uShadowStrength = { value: shadowStrength };

        // The targets live inside lights_fragment_begin, which is still an unexpanded
        // #include at this point — expand that one chunk and patch the expansion.
        const chunk = THREE.ShaderChunk.lights_fragment_begin
            .replace(
                'vec3 geometryViewDir = ( isOrthographic ) ? vec3( 0, 0, 1 ) : normalize( vViewPosition );',
                'vec3 geometryViewDir = ( isOrthographic ) ? vec3( 0, 0, 1 ) : ' +
                'normalize( mix( vViewPosition, uLockedCamPosView + vViewPosition, uSpecLock ) );')
            .replace(
                /directLight\.color \*= \( directLight\.visible && receiveShadow \) \? (get(?:Point)?Shadow\( [^;]*?) : 1\.0;/g,
                'directLight.color *= mix( uShadowTint, vec3( 1.0 ), ' +
                'mix( 1.0, ( ( directLight.visible && receiveShadow ) ? $1 : 1.0 ), uShadowStrength ) );');

        shader.fragmentShader = shader.fragmentShader
            .replace('#include <lights_fragment_begin>', chunk)
            .replace('void main() {',
                'uniform float uSpecLock;\nuniform vec3 uLockedCamPosView;\n' +
                'uniform vec3 uShadowTint;\nuniform float uShadowStrength;\nvoid main() {');

        material.userData.studioShader = shader;
    };
    // Without a distinct key three.js reuses the unpatched program compiled for other
    // materials of the same class.
    material.customProgramCacheKey = () => 'studio-patch-v1';
    return material;
}

/// Per-frame: feed the patched programs the lock/tint state. The locked camera position must
/// be re-expressed in the CURRENT view space every frame — that is what keeps the frozen
/// world-space view direction constant while the live camera orbits.
function syncStudioUniforms() {
    _lockedCamView.copy(specLock.camPos).applyMatrix4(camera.matrixWorldInverse);
    for (const material of [mesh?.material, ground?.material]) {
        const shader = material?.userData?.studioShader;
        if (!shader) continue;
        shader.uniforms.uSpecLock.value = specLock.on ? 1 : 0;
        shader.uniforms.uLockedCamPosView.value.copy(_lockedCamView);
        shader.uniforms.uShadowTint.value.set(shadowTintColor);
        shader.uniforms.uShadowStrength.value = shadowStrength;
    }
}

/// Locking captures the highlights as seen RIGHT NOW; unlocking goes back to live reflections.
export function setSpecularLock(on) {
    captureUndo('speclock', false);
    specLock.on = !!on;
    if (specLock.on && camera) specLock.camPos.copy(camera.position);
}

export function getSpecularLock() {
    return specLock.on;
}

export function setShadowTint(options) {
    captureUndo('shadowtint', true);
    if (options.color !== undefined) shadowTintColor = options.color;
    if (options.strength !== undefined) shadowStrength = Math.min(1, Math.max(0, options.strength));
}

/// JSON {color, strength} — a string so the Blazor page can consume it without a DTO.
export function getShadowTint() {
    return JSON.stringify({ color: shadowTintColor, strength: shadowStrength });
}

/// (Re)builds the cached lit-look material from the current appearance; swaps it in when it is the
/// one showing. The old one is disposed here — the ONLY place the look material is ever released.
function rebuildLookMaterial() {
    lookMaterial?.dispose();
    lookMaterial = buildMaterial();
    lookMaterial.userData.regionPatch = !!regionSurfAttr;
    if (mesh && !paintMode) mesh.material = lookMaterial;
    return lookMaterial;
}

/// The flat-lit vertex-colour paint-mode material — created once per model, then reused across
/// every paint-mode toggle (rebuilding meant recompiling its program every single toggle).
function ensurePaintMaterial() {
    if (!paintMaterial) {
        paintMaterial = applyRegionSurfacePatch(
            new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.75, metalness: 0.0 }), false);
    }
    return paintMaterial;
}

function buildMaterial(withRegions = true) {
    const color = new THREE.Color(appearance.color);
    switch (appearance.shader) {
        case 'matcap': {
            // Fixed idealized reflection — deliberately IGNORES the scene lights, and skips tone
            // mapping so the authored reference colors arrive on screen untouched. A custom
            // recipe (materials editor) takes precedence over the preset name.
            const texture = appearance.matcapDef && typeof appearance.matcapDef === 'object'
                ? matcapTextureFromDef(appearance.matcapDef)
                : matcapTexture(appearance.matcap);
            const material = new THREE.MeshMatcapMaterial({ color, matcap: texture });
            material.toneMapped = false;
            return material;
        }
        case 'toon': {
            // Half the color internally: physically-bright lights would otherwise push every
            // band to the top and the band edges — the whole point of this look — vanish.
            const dimmed = color.clone().multiplyScalar(0.5);
            return applyStudioShaderPatches(
                new THREE.MeshToonMaterial({ color: dimmed, gradientMap: toonGradient(appearance.toonBands) }));
        }
        case 'phong':
            // Exaggerated NMM metal that FOLLOWS the user's lights: dark metal body with big
            // punchy painterly hotspots — the classic look NMM tutorials paint from.
            return applyStudioShaderPatches(new THREE.MeshPhongMaterial({
                color,
                specular: new THREE.Color(appearance.specular ?? '#ffffff'),
                shininess: appearance.shininess ?? 90
            }));
        case 'nmm': {
            // Painterly NMM on the PHYSICAL material (metalness 0 = Lambert diffuse), so the
            // region patch attaches exactly as it does for the pbr looks: painted regions keep
            // their tints, per-region surfaces, and gem behaviour, while UNPAINTED fragments get
            // the authored ramp. The base colour is WHITE so the lit result measures pure light
            // response; tone mapping is skipped so ramp colours arrive on screen untouched.
            // Roughness gives a broad specular lobe the patch re-thresholds into a crisp painted
            // hotspot (a tight lobe would miss every camera-facing plane and never show).
            const hasRegions = withRegions && !!regionSurfAttr;
            const material = applyStudioShaderPatches(new THREE.MeshPhysicalMaterial({
                color: '#ffffff',
                metalness: 0,
                roughness: 0.30,
                envMapIntensity: (appearance.envIntensity ?? 0.75) * ambientEnvScale()
            }));
            const { def, key } = resolveNmmDef();
            applyNmmRampPatch(material, def, key + (hasRegions ? '|regions' : ''), hasRegions);
            material.toneMapped = false;
            return hasRegions ? applyRegionSurfacePatch(material, true) : material;
        }
        default: {
            const material = applyStudioShaderPatches(new THREE.MeshPhysicalMaterial({
                color,
                metalness: appearance.metalness,
                roughness: appearance.roughness,
                clearcoat: appearance.clearcoat,
                // Scaled by the ambient "room light" level — metals show the room, not the fill.
                envMapIntensity: (appearance.envIntensity ?? 0.4) * ambientEnvScale()
            }));
            // Painted regions with their own material (metal armour on a matte look…) override the
            // surface per vertex. Model only — props have no regions — and only once the region
            // buffers exist, so an unpainted model compiles no extra attributes.
            return withRegions && regionSurfAttr ? applyRegionSurfacePatch(material, true) : material;
        }
    }
}

// The "room light" level. Hemisphere fill alone only reaches DIFFUSE materials — the metal
// looks (metalness 1) ignore it entirely — so ambient also scales the studio environment's
// reflection intensity, which is what metal bodies actually show. Factor 1.0 at the default
// level, so the tuned looks are unchanged until the user moves the slider.
const DEFAULT_AMBIENT = 0.18;
let ambientLevel = DEFAULT_AMBIENT;

function ambientEnvScale() {
    return Math.min(2.5, ambientLevel / DEFAULT_AMBIENT);
}

/// Tone mapping control (dev/tuning hook). ACES lifts midtones and rolls off blown highlights.
export function setToneMapping(mode, exposure) {
    if (!renderer) return;
    renderer.toneMapping = mode === 'aces' ? THREE.ACESFilmicToneMapping : THREE.NoToneMapping;
    renderer.toneMappingExposure = exposure ?? 1.0;
    if (mesh) mesh.material.needsUpdate = true;
}

/// Adjusts the always-on ambient fill (user-facing; saved in rigs since setup v2). Drives the
/// hemisphere (diffuse looks) AND the environment reflection scale (metal looks) together.
export function setEnvironmentLight(options) {
    if (!hemi) return;
    captureUndo('ambient', true);
    if (options.intensity !== undefined) {
        ambientLevel = Math.max(0, options.intensity);
        hemi.intensity = ambientLevel;
        if (mesh && mesh.material.envMapIntensity !== undefined) {
            mesh.material.envMapIntensity = (appearance.envIntensity ?? 0.4) * ambientEnvScale();
            mesh.material.needsUpdate = true;
        }
    }
    if (options.sky !== undefined) hemi.color.set(options.sky);
    if (options.ground !== undefined) hemi.groundColor.set(options.ground);
}

// Appearance keys that drive LIVE uniforms: changing only these updates values in place —
// no material rebuild, no shader recompile — so their sliders stay smooth to drag.
const LIVE_APPEARANCE_KEYS = new Set(['gemPower', 'gemGain', 'nmmGain', 'nmmContrast']);

export function setAppearance(update) {
    captureUndo('appearance', true);
    Object.assign(appearance, update);
    if (update.gemPower !== undefined) gemParams.power = Math.max(0.2, Number(update.gemPower) || 1.35);
    if (update.gemGain !== undefined) gemParams.gain = Math.max(0, Number(update.gemGain) || 2.4);
    if (update.gemPower !== undefined || update.gemGain !== undefined) syncGemParams();
    if (update.nmmGain !== undefined || update.nmmContrast !== undefined) {
        const { def } = resolveNmmDef();
        const shader = lookMaterial?.userData?.nmmShader;
        if (shader?.uniforms?.uNmmGain) {
            shader.uniforms.uNmmGain.value = def.gain * Math.max(0.05, Number(appearance.nmmGain) || 1);
            shader.uniforms.uNmmContrast.value = def.contrast * Math.max(0.2, Number(appearance.nmmContrast) || 1);
        }
    }
    if (Object.keys(update).every(k => LIVE_APPEARANCE_KEYS.has(k))) return;
    // Rebuild the cached look (a real look change is the one legitimate rebuild); in paint mode the
    // flat region material keeps showing and the new look appears on exit via the cache.
    if (mesh) rebuildLookMaterial();
    // Imported props follow the look too (without the region patch — props aren't paintable).
    for (const entry of props.values()) {
        entry.mesh.material.dispose();
        entry.mesh.material = buildMaterial(false);
    }
}

/// The shipped material recipes (hex colours, editor-friendly) — what the materials editor
/// pre-fills when the user forks a built-in look into a custom material.
export function getMaterialDefaultsJson() {
    const linHex = c => '#' + new THREE.Color().setRGB(c[0], c[1], c[2], THREE.LinearSRGBColorSpace).getHexString();
    const rawHex = c => '#' + new THREE.Color().setRGB(c[0], c[1], c[2], THREE.SRGBColorSpace).getHexString();
    const ramps = {};
    for (const [name, r] of Object.entries(NMM_RAMPS)) {
        ramps[name] = {
            stops: r.stops.map(s => [s[0], linHex(s[1])]), spec: linHex(r.spec),
            specLo: r.specLo, specHi: r.specHi, edge: r.edge, gain: r.gain, knee: r.knee,
            skyDown: r.skyDown, contrast: r.contrast
        };
    }
    const matcaps = {};
    for (const [name, m] of Object.entries(MATCAPS)) {
        matcaps[name] = {
            skyTop: rawHex(m.skyTop), skyHorizon: rawHex(m.skyHorizon),
            groundHorizon: rawHex(m.groundHorizon), groundBottom: rawHex(m.groundBottom),
            horizon: m.horizon, horizonSoft: m.horizonSoft ?? 0.05, spec: m.spec,
            specPower: m.specPower, bounce: m.bounce, edge: m.edge ?? 0.35,
            specColor: rawHex(m.specColor ?? [1, 1, 1])
        };
    }
    return JSON.stringify({ nmmRamps: ramps, matcaps, gem: { power: gemParams.power, gain: gemParams.gain } });
}

// ---------------------------------------------------------------------------------------------
// Lights (with draggable gizmos)
// ---------------------------------------------------------------------------------------------

// three r155+ physical lighting: point-light intensity is candela with distance falloff, so
// raw slider values (0-4) vanish at this scene's ~100-unit light distances. UI and saved rigs
// keep the friendly 0-4 scale; converted here so a point light at the reference distance hits
// the model with exactly its slider value REGARDLESS of its falloff exponent — the falloff
// slider then only changes the curve shape (how fast it dims), not overall brightness.
const POINT_REFERENCE_DISTANCE = 100;
const pointCandela = (ui, decay) => ui * Math.pow(POINT_REFERENCE_DISTANCE, decay);
const clampDecay = value => Math.min(3, Math.max(0.5, value ?? 2));

// Near→far color gradient (OSL: candle flame yellow up close, red glow farther out).
// Emulated with a co-located PAIR of point lights: the near color decays faster, the far color
// slower, so their mix shifts with distance — the same trick painters use in paint. The two are
// balanced so their contributions CROSS at ~a third of the reference distance (i.e. within the
// model, where the gradient is actually visible) while the pair still totals the slider value
// at the reference distance, keeping brightness semantics identical to a plain light.
const GRADIENT_NEAR_EXTRA = 1.0;  // near light decays this much faster
const GRADIENT_FAR_LESS = 0.6;    // far light decays this much slower
// Where the two colors meet, as a fraction of the reference distance — user-adjustable per
// light ("gradient start"): small = the far tint takes over almost immediately (tight inner
// glow), large = the near color carries far before shifting.
const DEFAULT_GRADIENT_START = 0.35;
const clampGradientStart = value => Math.min(1.0, Math.max(0.08, value ?? DEFAULT_GRADIENT_START));

// Source size: a physical light has AREA — bigger sources give broader speculars and soft
// shadow edges (softbox vs candle). Three.js area lights cannot cast shadows, so size > 0
// expands each emitter into a tetrahedral cluster of four sub-lights spread over that radius,
// each at quarter intensity: highlights merge into a broader shape and the four overlapping
// shadow maps form a real penumbra.
const CLUSTER_OFFSETS = [
    new THREE.Vector3(1, 1, 1), new THREE.Vector3(1, -1, -1),
    new THREE.Vector3(-1, 1, -1), new THREE.Vector3(-1, -1, 1)
].map(v => v.normalize());

function makePointEmitter(objects, color, candela, decay, size, castShadows = true, mapSize = 512, rankBase = 0, shadowNear = undefined) {
    if (size <= 0) {
        const light = new THREE.PointLight(color, candela, 0, decay);
        if (castShadows) { configureShadowCaster(light, mapSize, shadowNear); light.userData.shadowRank = rankBase; }
        objects.push(light);
        return;
    }
    // EVERY cluster member casts, at quarter map size — 4×256² cube faces cost the same VRAM as
    // the single 512² caster used before. One caster out of four had let 75% of the emitter
    // shine straight through walls (measured: a size-12 light kept 3% of a plain light's
    // occlusion — "line of sight doesn't cast shadows"). Four overlapping maps also make the
    // penumbra REAL. The global caster budget (enforceShadowBudget) demotes extras when a rig
    // holds several such emitters. The near plane sits just past the source volume: an emitter
    // with physical SIZE lights the scene from its surface, so geometry inside the orb (the hand
    // holding it) must not swallow the light.
    for (const [i, offset] of CLUSTER_OFFSETS.entries()) {
        const light = new THREE.PointLight(color, candela / CLUSTER_OFFSETS.length, 0, decay);
        light.userData.clusterOffset = offset.clone().multiplyScalar(size);
        if (castShadows) {
            configureShadowCaster(light, Math.min(mapSize, 256), Math.max(shadowNear ?? 0, 2, size * 1.2));
            light.shadow.radius = 2 + size * 0.15;
            light.userData.shadowRank = i === 0 ? rankBase : rankBase + 3;
        }
        objects.push(light);
    }
}

function buildLightObjects(entry) {
    const objects = [];
    if (entry.type === 'directional') {
        // Parallel rays from infinity — "source size" has no meaning here, so directional lights
        // are always a single sharp caster (size applies to point lights only).
        const light = new THREE.DirectionalLight(new THREE.Color(entry.color), entry.uiIntensity);
        light.target.position.set(0, modelRadius * 0.6, 0);
        configureShadowCaster(light);
        objects.push(light);
    }
    else {
        // Region-anchored lights mount a CONSTELLATION: one emitter set per anchor, intensity
        // split by each anchor's area share, so a long glowing region (a lava river, a rune
        // strip) lights its surroundings along its whole shape — while the UI still shows one
        // light. A glowing region must also never BLOCK its own light: the shadow near plane
        // starts past the surface push-off, so the emitting surface itself is no occluder
        // (the same trick sized emitters use against orb-in-hand self-swallowing).
        const anchors = (entry.regionSource !== null && entry.regionAnchors?.length > 1)
            ? entry.regionAnchors
            : [{ offset: null, share: 1 }];
        const shadowNear = entry.regionSource !== null ? modelRadius * 0.12 * 1.35 : undefined;

        for (const [a, anchor] of anchors.entries()) {
            const before = objects.length;
            if (entry.farColor) {
                const dNear = Math.min(4, entry.decay + GRADIENT_NEAR_EXTRA);
                const dFar = Math.max(0.3, entry.decay - GRADIENT_FAR_LESS);
                // Equal contribution A at the crossover distance x; the pair totals the slider
                // value at the reference distance: A = ui / (t^dNear + t^dFar), candela = A·x^d.
                const t = clampGradientStart(entry.gradientStart);
                const x = t * POINT_REFERENCE_DISTANCE;
                const A = entry.uiIntensity * anchor.share / (Math.pow(t, dNear) + Math.pow(t, dFar));
                // BOTH halves cast. The far tint used to skip its cube map ("shares the near
                // light's position, a second map buys nothing") — but a non-casting light is
                // UNOCCLUDED, so the far colour poured straight through walls (measured: a
                // gradient light kept 16% of a plain light's occlusion). Its map rides at 256²
                // and yields first under the caster budget.
                makePointEmitter(objects, new THREE.Color(entry.color), A * Math.pow(x, dNear), dNear, entry.size, true, 512, a > 0 ? 2 : 0, shadowNear);
                makePointEmitter(objects, new THREE.Color(entry.farColor), A * Math.pow(x, dFar), dFar, entry.size, true, 256, a > 0 ? 3 : 1, shadowNear);
            }
            else {
                makePointEmitter(objects, new THREE.Color(entry.color),
                    pointCandela(entry.uiIntensity, entry.decay) * anchor.share, entry.decay, entry.size,
                    true, 512, a > 0 ? 2 : 0, shadowNear);
            }
            if (anchor.offset)
                for (let j = before; j < objects.length; j++)
                    objects[j].userData.regionOffset = anchor.offset;
        }
    }

    // Ground bounce (NMM's reflected under-light): light returned by the floor into the
    // model's undersides — the secondary highlight painters place under jaws, arms, and
    // shield rims. Tinted by the FLOOR's color. The scene casts no shadows, so below-ground
    // lights reach the model unobstructed while the ground itself (normals up) ignores them.
    if (entry.bounceStrength > 0) {
        let bounce;
        if (entry.type === 'directional') {
            // Uniform incident light bounces uniformly: straight back up, everywhere.
            bounce = new THREE.DirectionalLight(floorBounceTint(), entry.uiIntensity * entry.bounceStrength);
            bounce.position.set(0, -100, 0);
            bounce.target.position.set(0, 0, 0);
        }
        else {
            // A point light bounces from the bright patch on the table under it.
            bounce = new THREE.PointLight(floorBounceTint(),
                pointCandela(entry.uiIntensity * entry.bounceStrength, entry.decay), 0, entry.decay);
        }
        bounce.userData.groundMirror = true;
        objects.push(bounce);
    }
    return objects;
}

/// Positions an entry's light objects at its gizmo — bounce lights sit SHALLOW below the
/// ground under the light's footprint (where the bright reflected patch on a real table is),
/// which puts them close to the model's undersides for a readable grazing under-light.
function syncLightPositions(entry) {
    const p = entry.gizmo.position;
    for (const light of entry.lights) {
        if (light.userData.groundMirror) {
            // Point bounce tracks the bright patch under its parent; directional bounce is
            // uniform up-light whose position (hence direction) is fixed.
            if (light.isPointLight) light.position.set(p.x, -modelRadius * 0.3, p.z);
        }
        else {
            light.position.copy(p);
            if (light.userData.regionOffset) light.position.add(light.userData.regionOffset);
            if (light.userData.clusterOffset) light.position.add(light.userData.clusterOffset);
        }
    }
    requestShadowUpdate();
}

function mountLights(entry) {
    entry.lights = buildLightObjects(entry);
    for (const light of entry.lights) {
        scene.add(light);
        if (light.target) scene.add(light.target);
    }
    enforceShadowBudget(); // a new emitter may push the rig past the texture-unit budget
    syncLightPositions(entry); // requests the shadow re-render itself
}

function unmountLights(entry) {
    for (const light of entry.lights ?? []) {
        scene.remove(light);
        if (light.target) scene.remove(light.target);
        // A shadow-casting light owns a render-target shadow map (several MB). Removing the light does
        // NOT free it, so cycling light presets would pile them up until the GPU runs out of memory.
        light.shadow?.map?.dispose();
        light.shadow?.dispose?.();
    }
    entry.lights = [];
    enforceShadowBudget(); // freed budget re-promotes the best demoted casters
}

export function addLight(type, options) {
    captureUndo('lights', false);
    const id = nextLightId++;

    // The draggable handle: an unlit glowing ball where the light sits.
    const gizmoMaterial = new THREE.MeshBasicMaterial({ color: new THREE.Color(options?.color ?? '#ffffff') });
    gizmoMaterial.toneMapped = false; // handles keep their exact marker color
    const gizmo = new THREE.Mesh(new THREE.SphereGeometry(3.2, 16, 16), gizmoMaterial);
    clampHandle(gizmo.position.set(options?.x ?? 40, options?.y ?? 90, options?.z ?? 60));
    gizmo.userData.lightId = id;
    gizmo.visible = lightHandlesVisible; // the dots can be hidden while the lights stay on
    scene.add(gizmo);

    const entry = {
        type, gizmo,
        color: options?.color ?? '#ffffff',
        farColor: type === 'point' ? (options?.farColor ?? null) : null,
        uiIntensity: options?.intensity ?? 1.0,
        decay: clampDecay(options?.decay),
        gradientStart: clampGradientStart(options?.gradientStart),
        bounceStrength: Math.min(1, Math.max(0, options?.bounceStrength ?? 0)),
        size: type === 'point' ? Math.min(24, Math.max(0, options?.size ?? 0)) : 0,
        // A glow-region light is anchored to a painted region: its position/colour derive from the
        // region (recomputed when painting changes) and the region's surface renders emissive.
        regionSource: type === 'point' ? (options?.regionSource ?? null) : null,
        enabled: options?.enabled !== false // switched off but kept: the dimmed handle stays put
    };
    entry.gizmo.scale.setScalar(1 + entry.size / 8); // the handle hints at the source size
    if (entry.regionSource !== null && options?.x === undefined) anchorRegionLight(entry);
    lights.set(id, entry); // register FIRST: mountLights budgets casters across the whole map
    if (entry.enabled) mountLights(entry);
    else entry.lights = [];
    applyLightOrbLook(entry);
    syncGlowingSlots();
    return id;
}

// ----- Glow-region lights ---------------------------------------------------------------------
// The light source IS a painted region (a glowing blade, lava, runes): a point light sits just off
// the region's surface (area-weighted centroid pushed out along the average normal) in the region's
// colour, and the region itself renders emissive so it reads as the emitter.

/// Recomputes a glow light's position + colour from its region's current painting. A region with
/// spatial EXTENT (a lava river along the base, a winding rune strip) also earns several ANCHOR
/// points via a small area-weighted k-means over its triangles: the light then mounts as a
/// constellation along the shape (see buildLightObjects), so the surroundings are lit along the
/// whole glowing area instead of from one centroid — while the UI still shows ONE light with one
/// handle, one intensity, one colour.
function anchorRegionLight(entry) {
    const k = entry.regionSource;
    if (k === null || !regionIndex || !mesh) return false;
    const pos = mesh.geometry.attributes.position;

    // Area-weighted triangle samples (strided so multi-million-face regions stay cheap).
    const faceCount = regionIndex.length / 3;
    const stride = Math.max(1, Math.floor(faceCount / 60000));
    const sx = [], sy = [], sz = [], sw = [];
    let cx = 0, cy = 0, cz = 0, nx = 0, ny = 0, nz = 0, totalW = 0;
    let minX = Infinity, minY = Infinity, minZ = Infinity, maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (let t = 0; t < faceCount; t += stride) {
        if (regionIndex[t * 3] !== k) continue; // soup: face label = its first vertex's label
        const i = t * 3;
        const x = (pos.getX(i) + pos.getX(i + 1) + pos.getX(i + 2)) / 3;
        const y = (pos.getY(i) + pos.getY(i + 1) + pos.getY(i + 2)) / 3;
        const z = (pos.getZ(i) + pos.getZ(i + 1) + pos.getZ(i + 2)) / 3;
        const ux = pos.getX(i + 1) - pos.getX(i), uy = pos.getY(i + 1) - pos.getY(i), uz = pos.getZ(i + 1) - pos.getZ(i);
        const vx = pos.getX(i + 2) - pos.getX(i), vy = pos.getY(i + 2) - pos.getY(i), vz = pos.getZ(i + 2) - pos.getZ(i);
        const crx = uy * vz - uz * vy, cry = uz * vx - ux * vz, crz = ux * vy - uy * vx;
        const w = Math.max(1e-9, Math.hypot(crx, cry, crz)); // 2× triangle area
        sx.push(x); sy.push(y); sz.push(z); sw.push(w);
        cx += x * w; cy += y * w; cz += z * w;
        nx += crx; ny += cry; nz += crz;
        totalW += w;
        if (x < minX) minX = x; if (x > maxX) maxX = x;
        if (y < minY) minY = y; if (y > maxY) maxY = y;
        if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
    }
    if (totalW === 0) return false; // region erased — the light stays where it was

    cx /= totalW; cy /= totalW; cz /= totalW;
    const nLen = Math.hypot(nx, ny, nz) || 1;
    // Push off the surface so the light isn't born inside the mesh (self-shadowed to nothing).
    const off = modelRadius * 0.12;
    entry.gizmo.position.set(cx + (nx / nLen) * off, cy + (ny / nLen) * off, cz + (nz / nLen) * off);
    clampHandle(entry.gizmo.position);

    // How many anchors the extent earns: a compact patch keeps one, an elongated region two,
    // a sprawling one three. Offsets are stored RELATIVE to the handle, so dragging the handle
    // moves the whole constellation together.
    const extent = Math.hypot(maxX - minX, maxY - minY, maxZ - minZ);
    const anchorCount = extent < modelRadius * 0.55 ? 1 : extent < modelRadius * 1.1 ? 2 : 3;
    entry.regionAnchors = anchorCount <= 1 ? null
        : clusterRegionAnchors(sx, sy, sz, sw, anchorCount, off, entry.gizmo.position);

    const slot = regionPalette[k - 1];
    if (slot) entry.color = slot.color;
    return true;
}

/// Area-weighted k-means (farthest-point seeded, a few Lloyd rounds) over a region's triangle
/// samples. Returns [{offset: Vector3 relative to the handle, share}] with shares summing to 1;
/// clusters under 3% of the area are dropped (a stray brush dot shouldn't earn its own light).
function clusterRegionAnchors(sx, sy, sz, sw, K, surfaceOffset, origin) {
    const n = sx.length;
    if (n < K * 4) return null;

    // Seeds: farthest point from the mean, then greedily the point farthest from all seeds.
    const seedIdx = [];
    const distToSeeds = new Float32Array(n).fill(Infinity);
    let mx = 0, my = 0, mz = 0, mw = 0;
    for (let i = 0; i < n; i++) { mx += sx[i] * sw[i]; my += sy[i] * sw[i]; mz += sz[i] * sw[i]; mw += sw[i]; }
    mx /= mw; my /= mw; mz /= mw;
    let best = 0, bestD = -1;
    for (let i = 0; i < n; i++) {
        const d = (sx[i] - mx) ** 2 + (sy[i] - my) ** 2 + (sz[i] - mz) ** 2;
        if (d > bestD) { bestD = d; best = i; }
    }
    seedIdx.push(best);
    while (seedIdx.length < K) {
        const s = seedIdx[seedIdx.length - 1];
        best = 0; bestD = -1;
        for (let i = 0; i < n; i++) {
            const d = (sx[i] - sx[s]) ** 2 + (sy[i] - sy[s]) ** 2 + (sz[i] - sz[s]) ** 2;
            if (d < distToSeeds[i]) distToSeeds[i] = d;
            if (distToSeeds[i] > bestD) { bestD = distToSeeds[i]; best = i; }
        }
        seedIdx.push(best);
    }

    const centers = seedIdx.map(i => [sx[i], sy[i], sz[i]]);
    const assign = new Int32Array(n);
    for (let iter = 0; iter < 5; iter++) {
        for (let i = 0; i < n; i++) {
            let bi = 0, bd = Infinity;
            for (let c = 0; c < K; c++) {
                const d = (sx[i] - centers[c][0]) ** 2 + (sy[i] - centers[c][1]) ** 2 + (sz[i] - centers[c][2]) ** 2;
                if (d < bd) { bd = d; bi = c; }
            }
            assign[i] = bi;
        }
        const acc = centers.map(() => [0, 0, 0, 0]);
        for (let i = 0; i < n; i++) {
            const a = acc[assign[i]];
            a[0] += sx[i] * sw[i]; a[1] += sy[i] * sw[i]; a[2] += sz[i] * sw[i]; a[3] += sw[i];
        }
        for (let c = 0; c < K; c++)
            if (acc[c][3] > 0) centers[c] = [acc[c][0] / acc[c][3], acc[c][1] / acc[c][3], acc[c][2] / acc[c][3]];
    }

    // Shares + per-cluster surface push-off along the direction from the region's interior
    // (approximate: from the cluster centroid away from the overall mean — cheap and stable).
    const weights = new Array(K).fill(0);
    for (let i = 0; i < n; i++) weights[assign[i]] += sw[i];
    const total = weights.reduce((a, b) => a + b, 0) || 1;
    const anchors = [];
    for (let c = 0; c < K; c++) {
        const share = weights[c] / total;
        if (share < 0.03) continue;
        const dx = centers[c][0] - mx, dy = centers[c][1] - my, dz = centers[c][2] - mz;
        const dLen = Math.hypot(dx, dy, dz);
        // Push out along the local outward hint (or straight up for the central cluster).
        const px = centers[c][0] + (dLen > 1e-6 ? dx / dLen : 0) * surfaceOffset * 0.4;
        const py = centers[c][1] + (dLen > 1e-6 ? dy / dLen : 1) * surfaceOffset * 0.4 + surfaceOffset * 0.6;
        const pz = centers[c][2] + (dLen > 1e-6 ? dz / dLen : 0) * surfaceOffset * 0.4;
        anchors.push({ offset: new THREE.Vector3(px - origin.x, py - origin.y, pz - origin.z), share });
    }
    if (anchors.length < 2) return null;
    const shareSum = anchors.reduce((a, b) => a + b.share, 0);
    for (const a of anchors) a.share /= shareSum;
    return anchors;
}

/// Re-anchors every glow light and refreshes the emissive flags — called whenever painting,
/// palette colours, or the mask itself change shape.
function refreshRegionLights() {
    let anyLight = false;
    for (const entry of lights.values()) {
        if (entry.regionSource === null) continue;
        anyLight = true;
        anchorRegionLight(entry);
        applyLightOrbLook(entry);
        unmountLights(entry);
        if (entry.enabled) mountLights(entry);
    }
    if (anyLight) requestShadowUpdate();
    syncGlowingSlots();
}

/// The emissive gain a glowing region's SURFACE renders at — the same curve the emitter orbs
/// use (1 + intensity × 1.5, tone-mapped so a hot core rolls off to white), so the painted area
/// reads as the actual light source at its actual level. 0 = the slot powers no enabled light.
function glowGainFor(idx) {
    let gain = 0;
    for (const entry of lights.values())
        if (entry.regionSource === idx && entry.enabled)
            gain = Math.max(gain, 1 + entry.uiIntensity * 1.5);
    return Math.min(6, gain);
}

/// Which region slots currently power an ENABLED glow light — those render emissive.
function syncGlowingSlots() {
    const glowing = new Set();
    for (const entry of lights.values()) {
        if (entry.regionSource !== null && entry.enabled) glowing.add(entry.regionSource);
    }
    const changed = glowing.size !== glowingSlots.size || [...glowing].some(k => !glowingSlots.has(k));
    glowingSlots = glowing;
    if (changed && regionIndex) repaintColors(); // rebake the emissive flags into aRegionSurf
}

/// The handle sphere IS the visible light source: an orb at the light's colour driven past 1.0
/// by its intensity, so ACES renders it as a hot glowing emitter — not a flat UI marker. A
/// switched-off light keeps a dim ghost so it can still be selected and moved.
function applyLightOrbLook(entry) {
    const material = entry.gizmo.material;
    if (entry.enabled) {
        material.color.set(entry.color).multiplyScalar(1 + entry.uiIntensity * 1.5);
        material.toneMapped = true; // the tone mapper rolls the hot core off like a real emitter
        material.transparent = false;
        material.opacity = 1;
    } else {
        material.color.set(entry.color);
        material.toneMapped = false;
        material.transparent = true;
        material.opacity = 0.3;
    }
}

export function updateLight(id, update) {
    const entry = lights.get(id);
    if (!entry) return;
    captureUndo('update:' + id, true);

    if (update.color !== undefined) entry.color = update.color;
    if (update.farColor !== undefined) entry.farColor = entry.type === 'point' ? update.farColor : null;
    if (update.intensity !== undefined) entry.uiIntensity = update.intensity;
    if (update.decay !== undefined) entry.decay = clampDecay(update.decay);
    if (update.gradientStart !== undefined) entry.gradientStart = clampGradientStart(update.gradientStart);
    if (update.bounceStrength !== undefined)
        entry.bounceStrength = Math.min(1, Math.max(0, update.bounceStrength));
    if (update.size !== undefined) {
        entry.size = entry.type === 'point' ? Math.min(24, Math.max(0, update.size)) : 0;
        entry.gizmo.scale.setScalar(1 + entry.size / 8);
    }
    if (update.enabled !== undefined) { entry.enabled = !!update.enabled; syncGlowingSlots(); }
    if (update.x !== undefined) clampHandle(entry.gizmo.position.set(update.x, update.y, update.z));
    applyLightOrbLook(entry); // colour/intensity/enabled all feed the orb's glow

    // A glow region's SURFACE brightness follows its light's intensity — rebake the encoded
    // gain (aRegionSurf.w band) so dragging the slider visibly brightens/dims the glow.
    if (entry.regionSource !== null && update.intensity !== undefined && regionIndex) repaintColors();

    // Rebuild rather than mutate: color/decay/gradient changes can change the NUMBER of
    // underlying lights, and ≤4 rig lights make this trivially cheap.
    unmountLights(entry);
    if (entry.enabled) mountLights(entry);
    else requestShadowUpdate(); // a removed caster leaves its shadow behind otherwise
}

export function removeLight(id) {
    const entry = lights.get(id);
    if (!entry) return;
    captureUndo('lights', false);
    if (selectedLightId === id) selectLight(null);
    unmountLights(entry);
    scene.remove(entry.gizmo);
    entry.gizmo.geometry.dispose();
    entry.gizmo.material.dispose();
    lights.delete(id);
    syncGlowingSlots(); // a removed glow light stops its region's emissive
    requestShadowUpdate();
}

export function clearLights() {
    captureUndo('lights', false);
    withUndoSuppressed(() => {
        for (const id of [...lights.keys()]) removeLight(id);
    });
}

/// Duplicates a light with all its settings, placed beside the original and selected.
export function cloneLight(id) {
    const entry = lights.get(id);
    if (!entry) return null;
    const newId = addLight(entry.type, {
        x: entry.gizmo.position.x + Math.max(10, modelRadius * 0.3),
        y: entry.gizmo.position.y,
        z: entry.gizmo.position.z,
        color: entry.color,
        farColor: entry.farColor,
        intensity: entry.uiIntensity,
        decay: entry.decay,
        gradientStart: entry.gradientStart,
        bounceStrength: entry.bounceStrength,
        size: entry.size
    });
    selectLight(newId);
    return newId;
}

export function applyPreset(name) {
    captureUndo('preset', false);
    withUndoSuppressed(() => {
        clearLights();
        const r = modelRadius;
        switch (name) {
            case 'zenithal':
                addLight('directional', { x: 0, y: r * 3, z: 0, intensity: 2.2 });
                addLight('point', { x: 0, y: r * 0.4, z: r * 2.2, color: '#8899aa', intensity: 0.35 });
                break;
            case 'candle':
                // Fire gradient: warm yellow up close shifting to deep red-orange farther out,
                // with a fast falloff — the classic OSL study. (2.6 pre-ACES blew out the core.)
                addLight('point', { x: r * 0.9, y: r * 0.5, z: r * 1.2, color: '#ffd27a',
                    farColor: '#c23f14', intensity: 1.6, decay: 2.4 });
                break;
            default: // keyrim — the key gets a ground bounce (the classic NMM under-light)
                addLight('point', { x: r * 1.6, y: r * 1.8, z: r * 1.6, intensity: 2.0, bounceStrength: 0.5 });
                addLight('point', { x: -r * 1.8, y: r * 1.2, z: -r * 1.4, color: '#7fb4ff', intensity: 1.1 });
                break;
        }
    });
    return getLightsJson();
}

export function getLightsJson() {
    const list = [];
    for (const [id, entry] of lights) {
        list.push({
            id, type: entry.type,
            x: entry.gizmo.position.x, y: entry.gizmo.position.y, z: entry.gizmo.position.z,
            color: entry.color,
            farColor: entry.farColor,
            intensity: entry.uiIntensity,
            decay: entry.decay,
            gradientStart: entry.gradientStart,
            bounceStrength: entry.bounceStrength,
            size: entry.size,
            regionSource: entry.regionSource,
            enabled: entry.enabled,
            selected: id === selectedLightId
        });
    }
    return JSON.stringify(list);
}

// Drag handling: grab a gizmo, move it on the camera-facing plane through its position.
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const dragPlane = new THREE.Plane();
let dragging = null;

function setPointer(event) {
    const rect = canvasEl.getBoundingClientRect();
    pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
}

function onPointerDown(event) {
    if (axisControl?.dragging || propControl?.dragging) return; // a gizmo grab owns this gesture

    setPointer(event);
    raycaster.setFromCamera(pointer, camera);

    // Region painting owns the gesture when it lands on the model; empty space still orbits.
    // Left paints the selected region; RIGHT erases (region 0) without reselecting.
    if (paintMode && mesh && (event.button === 0 || event.button === 2)) {
        const paintHit = raycaster.intersectObject(mesh)[0];
        if (paintHit) {
            strokeDiff = new Map(); // record pre-stroke region indices so the stroke can be undone
            strokeRegion = event.button === 2 ? 0 : brush.region;
            if (brush.mode === 'fill') {
                // One click = one fill = one undo step; no drag gesture to hold.
                ensureRegionBuffers();
                fillFromFace(paintHit.faceIndex ?? -1);
                pushPaintUndo(strokeDiff);
                strokeDiff = null;
                notifyChanged();
                return;
            }
            painting = true;
            controls.enabled = false;
            canvasEl.setPointerCapture(event.pointerId);
            paintAt(paintHit.point);
            updateBrushCursor(paintHit);
        }
        return;
    }

    // Hidden handles are not clickable — invisible grab targets would be baffling.
    const gizmos = lightHandlesVisible ? [...lights.values()].map(entry => entry.gizmo) : [];
    const hit = raycaster.intersectObjects(gizmos)[0];
    if (!hit) {
        // Not a light handle — try an imported prop, else clear the selection.
        const propMeshes = [...props.values()].map(e => e.mesh);
        const propHit = propMeshes.length ? raycaster.intersectObjects(propMeshes)[0] : null;
        if (propHit) {
            selectProp(propHit.object.userData.propId);
            notifyChanged();
            return;
        }
        if (selectedLightId !== null) { selectLight(null); notifyChanged(); }
        if (selectedPropId !== null) { deselectProp(); notifyChanged(); }
        return;
    }

    if (selectedPropId !== null) deselectProp(); // switching from a prop to a light
    if (selectedLightId !== hit.object.userData.lightId) {
        selectLight(hit.object.userData.lightId);
        notifyChanged(); // the panel highlights the selected light's card
    }
    captureUndo('drag', false); // each handle grab is one undo step
    dragging = hit.object.userData.lightId;
    dragPlane.setFromNormalAndCoplanarPoint(
        camera.getWorldDirection(new THREE.Vector3()).negate(), hit.object.position);
    controls.enabled = false;
    canvasEl.setPointerCapture(event.pointerId);
}

// The glowing balls marking light positions. Hiding them (lights unchanged!) gives a clean view
// for reading the render; captures exclude them regardless.
let lightHandlesVisible = true;

export function setLightHandlesVisible(on) {
    lightHandlesVisible = !!on;
    for (const entry of lights.values()) entry.gizmo.visible = lightHandlesVisible;
    if (!lightHandlesVisible) selectLight(null); // arrows belong to the handles; hide those too
}

export function getLightHandlesVisible() {
    return lightHandlesVisible;
}

/// Shows the X/Y/Z arrows on one light's handle (null hides them).
export function selectLight(id) {
    selectedLightId = id;
    const entry = id !== null ? lights.get(id) : null;
    if (entry) axisControl.attach(entry.gizmo);
    else axisControl.detach();
}

function onPointerMove(event) {
    // Paint-mode moves are COALESCED to one raycast per rendered frame (processPendingPaint):
    // gaming mice fire 250+ moves/sec, and every one used to raycast a multi-million-triangle
    // mesh — several full-mesh raycasts per frame was most of paint mode's sluggishness.
    if (painting || (paintMode && mesh && dragging === null)) {
        pendingPaintMove = event;
        return;
    }
    if (dragging === null) return;
    setPointer(event);
    raycaster.setFromCamera(pointer, camera);
    const point = new THREE.Vector3();
    if (raycaster.ray.intersectPlane(dragPlane, point)) {
        const entry = lights.get(dragging);
        entry.gizmo.position.copy(point);
        clampHandle(entry.gizmo.position);
        syncLightPositions(entry);
    }
}

/// One raycast per frame for the paint brush (stamp while stroking, ring preview otherwise; in
/// fill mode the hover shows a dashed boundary of what a click would fill).
let lastPaintHover = null;
function processPendingPaint() {
    if (!pendingPaintMove) return;
    const event = pendingPaintMove;
    pendingPaintMove = null;
    if (!mesh || !paintMode) return;
    lastPaintHover = event; // re-injected when the fill angle changes, so the preview tracks the slider
    setPointer(event);
    raycaster.setFromCamera(pointer, camera);
    const hit = raycaster.intersectObject(mesh)[0];
    if (brush.mode === 'fill') {
        if (!hit || hit.faceIndex === undefined) { hideFillPreview(); return; }
        const now = performance.now();
        const changed = hit.faceIndex !== fillPreview.face || brush.fillAngle !== fillPreview.angle;
        if (changed && now - fillPreview.time > 90) { // BFS per hover is real work on big fills
            updateFillPreview(hit.faceIndex);
            fillPreview.face = hit.faceIndex;
            fillPreview.angle = brush.fillAngle;
            fillPreview.time = now;
        }
        return;
    }
    if (painting && hit) paintAt(hit.point);
    updateBrushCursor(hit);
}

function onPointerUp() {
    if (painting) {
        painting = false;
        controls.enabled = true;
        pushPaintUndo(strokeDiff); // one undo step per stroke
        strokeDiff = null;
        notifyChanged(); // tells the panel there are unsaved region changes
        return;
    }
    dragging = null;
    controls.enabled = true;
}

/// A paint stroke/clear is undone by restoring the recorded vertices, not by a full setup snapshot.
/// Paint diffs hold a Map entry per touched vertex — on a multi-million-vertex print a broad stroke
/// is tens of MB, so they get their own tight cap (and whole-model diffs aren't kept at all: a
/// 9M-entry Map is hundreds of MB of heap, a real out-of-memory source).
const PAINT_UNDO_LIMIT = 6;
const PAINT_UNDO_MAX_ENTRIES = 1_000_000;
function pushPaintUndo(diff) {
    if (!diff || diff.size === 0) return;
    if (diff.size > PAINT_UNDO_MAX_ENTRIES) return;
    if (undoStack.filter(u => u.paintDiff).length >= PAINT_UNDO_LIMIT) {
        const oldest = undoStack.findIndex(u => u.paintDiff);
        if (oldest !== -1) undoStack.splice(oldest, 1);
    }
    undoStack.push({ tag: 'paint', paintDiff: diff });
    if (undoStack.length > UNDO_LIMIT) undoStack.shift();
}

// ---------------------------------------------------------------------------------------------
// Region painting — mark areas on the STL surface (per-vertex) to highlight by area.
//
// The mask is a per-vertex region index (0 = unpainted); painting is a spherical brush that sets
// every vertex within a world-space radius of the hit point. In paint mode the model switches to
// a flat, unlit vertex-colour material so regions read clearly; the lit look is left untouched and
// rebuilt on exit. The mask + palette persist as one JSON blob per model.
// ---------------------------------------------------------------------------------------------

let paintMode = false;
let painting = false;
let strokeDiff = null;         // during a stroke: vertexIndex → previous region index, for undo
let strokeRegion = 1;          // region this stroke lays down: brush.region, or 0 on a right-drag (erase)
let pendingPaintMove = null;   // latest paint-mode pointermove; raycast once per frame, not per event
let regionsDirty = false;      // unsaved mask/palette edits; the panel offers Save/Revert while set
let regionIndex = null;        // Uint8Array, one per vertex; 0 = unpainted
let regionColorAttr = null;    // THREE 'color' BufferAttribute on the geometry
let regionSurfAttr = null;     // per-vertex (use, metalness, roughness) for regions with their own material
let gemFlagAttr = null;        // per-vertex 0/1: fragment shades as a gem (transmission look). Its own
                               // byte — multiplexing it into aRegionSurf.w would make interpolation
                               // at every region border sweep through the gem level and band-glow.
let isolated = -1;             // region highlighted alone; -1 = show all
let paintGrid = null;          // uniform spatial grid over vertices, so the brush is O(nearby), not O(all)
let highlightAttr = null;      // per-vertex 0/1 mask (aHighlight) driving the on-model glow overlay
let overlayMesh = null;        // additive glow of the isolated region, drawn over the LIT model
const brush = {
    region: 1,
    radius: 0.10,   // as a fraction of the model radius (brush mode)
    mode: 'brush',  // 'brush' drags a sphere; 'fill' flood-fills the clicked surface patch
    fillAngle: 30,  // fill spreads while faces stay within this many degrees of the clicked face
};
let fillTopology = null;       // lazily built face adjacency + normals for the fill tool

let brushCursor = null;               // ring showing the brush footprint on the surface while painting
const Z_AXIS = new THREE.Vector3(0, 0, 1);

function ensureBrushCursor() {
    if (brushCursor) return;
    // A thin ring (unit outer radius) laid flat on the surface. Unlit and depth-test off so it stays
    // visible over both the light clay and any dark/coloured regions, never hidden by the geometry.
    const mat = new THREE.MeshBasicMaterial({
        color: 0xffffff, transparent: true, opacity: 0.85,
        depthTest: false, depthWrite: false, side: THREE.DoubleSide, toneMapped: false,
    });
    brushCursor = new THREE.Mesh(new THREE.RingGeometry(0.86, 1.0, 48), mat);
    brushCursor.renderOrder = 3;      // over the model and the region glow overlay
    brushCursor.frustumCulled = false;
    brushCursor.visible = false;
    scene.add(brushCursor);
}

/// Lays the brush ring on the surface at a raycast hit (sized to the brush), or hides it when there
/// is no hit, we're not in paint mode, or the fill tool is active (a radius ring would mislead).
function updateBrushCursor(hit) {
    if (!paintMode || !hit || !mesh || brush.mode === 'fill') { if (brushCursor) brushCursor.visible = false; return; }
    ensureBrushCursor();
    const nrm = (hit.face
        ? hit.face.normal.clone().transformDirection(mesh.matrixWorld)
        : camera.getWorldDirection(new THREE.Vector3()).negate()).normalize();
    brushCursor.position.copy(hit.point).addScaledVector(nrm, modelRadius * 0.003); // lift off z-fighting
    brushCursor.quaternion.setFromUnitVectors(Z_AXIS, nrm);
    const rad = brush.radius * modelRadius;
    brushCursor.scale.set(rad, rad, 1);
    brushCursor.visible = true;
}

// A light clay grey for unpainted surface: lit in paint mode (see below) it reads as a grey model
// whose shape and surface detail are clearly visible, while the coloured regions stand out on top.
const NEUTRAL = new THREE.Color('#9298a0');

const DEFAULT_PALETTE = [
    { name: 'Armour',  color: '#3f6fd0', material: 'look' },
    { name: 'Cloth',   color: '#c0392b', material: 'look' },
    { name: 'Skin',    color: '#e0a875', material: 'look' },
    { name: 'Metal',   color: '#c9a13b', material: 'metal' },
    { name: 'Leather', color: '#7a4a2e', material: 'look' },
    { name: 'Base',    color: '#5f6a4a', material: 'look' },
];
let regionPalette = DEFAULT_PALETTE.map(r => ({ ...r }));
let regionThreeColors = buildRegionColors();
let glowingSlots = new Set(); // region slots powering an enabled glow light — rendered emissive

// Per-region surface finishes (metalness, roughness) — same values as the matching LOOKS presets,
// so an area set to "metal" gleams exactly like the Silver NMM look does. 'look' (or an unknown /
// missing value on an older saved mask) means the area has no surface of its own and follows the
// global look, exactly as before regions had materials.
const REGION_MATERIALS = {
    matte: { metalness: 0.0, roughness: 0.85 },
    satin: { metalness: 0.0, roughness: 0.38 },
    gloss: { metalness: 0.0, roughness: 0.15 },
    metal: { metalness: 1.0, roughness: 0.30 },
    // Gem: transmission look — the shader inverts the diffuse term for these fragments (see the
    // gem section of applyRegionSurfacePatch), so only the specular settings matter here: low
    // roughness gives the small sharp light-side sparkle painters put on a stone.
    gem:   { metalness: 0.0, roughness: 0.16 },
};

// (materialUse, metalness, roughness, tintUse) in [0,1] for the normalized aRegionSurf attribute —
// the setters convert to bytes themselves (BufferAttribute normalize-on-set). EVERY painted vertex
// tints the lit render with its region colour (tintUse=1); the metalness/roughness override only
// applies when the slot names a material ('Follow look' keeps the look's surface under the tint).
function surfFor(idx) {
    if (idx <= 0) return [0, 0, 0, 0];
    const slot = regionPalette[idx - 1];
    const m = slot ? REGION_MATERIALS[slot.material] : undefined;
    // w: 0 = unpainted, 0.6 = tinted, 0.86–1.0 = tinted + EMISSIVE with the glow light's gain
    // encoded into the band (0.86 → 0, 1.0 → 6): the region surface IS the source, so its
    // brightness must track the light's intensity, not a fixed constant.
    const gain = glowGainFor(idx);
    const w = gain > 0 ? 0.86 + (gain / 6) * 0.14 : 0.6;
    if (!m) return [0, 0, 0, w];
    // Per-slot fine-tune: an explicit metalness/roughness on the slot overrides the preset's
    // values (the presets become starting points). Rides in the palette, so it saves with the mask.
    const metal = Math.min(1, Math.max(0, slot.metalness ?? m.metalness));
    const rough = Math.min(1, Math.max(0, slot.roughness ?? m.roughness));
    return [1, metal, rough, w];
}

// ---------------------------------------------------------------------------------------------
// Region surface patch: where a painted region has its own material, override the material's
// metalness/roughness (and, outside paint mode, its colour) per fragment. Driven by aRegionSurf
// (use, metalness, roughness — normalized bytes, so 4 bytes/vertex on multi-million-vertex prints)
// plus the existing per-vertex 'color' attribute as the tint, so no second colour buffer exists.
// vRegionSurf.x interpolates 1→0 across a region's boundary triangles, giving a soft edge.
// Only PBR materials get the patch — matcap/toon are fixed stylized references by design.
// ---------------------------------------------------------------------------------------------
function applyRegionSurfacePatch(material, useTint) {
    const prevCompile = material.onBeforeCompile;
    material.onBeforeCompile = shader => {
        if (prevCompile) prevCompile(shader);

        shader.vertexShader = shader.vertexShader.replace('void main() {',
            'attribute vec4 aRegionSurf;\nvarying vec4 vRegionSurf;\n' +
            'attribute vec3 aGemFlag;\nvarying vec3 vGemParams;\n' +
            (useTint ? 'attribute vec3 color;\nvarying vec3 vRegionTint;\n' : '') +
            'void main() {\n  vRegionSurf = aRegionSurf;\n  vGemParams = aGemFlag;' +
            (useTint ? '\n  vRegionTint = color;' : ''));

        // Varyings go at the very TOP of the fragment source (not at void main): the gem term
        // below lives inside RE_Direct_Physical, a pars function defined before main, and GLSL
        // resolves names in file order.
        shader.fragmentShader =
            'varying vec4 vRegionSurf;\nvarying vec3 vGemParams;\n' +
            (useTint ? 'varying vec3 vRegionTint;\n' : '') +
            shader.fragmentShader;
        material.userData.regionShader = shader;

        // The factor assignments live inside unexpanded #includes — expand and patch them, as the
        // lock-specular patch does for the lights chunk.
        shader.fragmentShader = shader.fragmentShader
            .replace('#include <metalnessmap_fragment>', THREE.ShaderChunk.metalnessmap_fragment
                .replace('float metalnessFactor = metalness;',
                    'float metalnessFactor = mix( metalness, vRegionSurf.y, vRegionSurf.x );'))
            .replace('#include <roughnessmap_fragment>', THREE.ShaderChunk.roughnessmap_fragment
                .replace('float roughnessFactor = roughness;',
                    'float roughnessFactor = mix( roughness, vRegionSurf.z, vRegionSurf.x );'));

        // ---- Gem transmission (per-fragment, driven by vGemParams) ----------------------------
        // A painted gem lights BACKWARDS: the light enters the stone and exits the far side, so
        // the area facing the light stays dark (bar a small sharp sparkle) and the area facing
        // away glows — the "window". Two patches make that happen:
        //  1. RE_Direct_Physical's diffuse term flips to the far side (pow tightens the window
        //     toward the far pole; specular keeps the true normal, giving the light-side dot).
        shader.fragmentShader = shader.fragmentShader
            .replace('#include <lights_physical_pars_fragment>', THREE.ShaderChunk.lights_physical_pars_fragment
                .replace('reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );',
                    // Wider lobe than Lambert (window exponent) and a focus boost (glow gain): the
                    // transmitted window painters put on a stone is broad and MORE saturated-bright
                    // than a plain lit face. PER-REGION values ride in vGemParams (y: power over
                    // [0.2,4], z: gain over [0,6] — gemEncode is the writing side).
                    'vec3 gemIrradiance = pow( saturate( dot( -geometryNormal, directLight.direction ) ), 0.2 + vGemParams.y * 3.8 ) * ( vGemParams.z * 6.0 ) * directLight.color;\n' +
                    '\treflectedLight.directDiffuse += mix( irradiance, gemIrradiance, vGemParams.x ) * BRDF_Lambert( material.diffuseColor );'))
        //  2. Gem fragments stop RECEIVING shadows: the window sits on the model's own far side,
        //     which the shadow map always marks occluded — shadowed transmission would be black.
        //     The raw receiveShadow condition survives inside the studio patch's shadow-tint
        //     rewrite (a gem forces its shadow factor to 1, which that mix maps to "unshadowed"),
        //     so this one replacement works with or without the studio patch applied first; the
        //     expand below no-ops when the studio patch already expanded the chunk.
            .replace('#include <lights_fragment_begin>', THREE.ShaderChunk.lights_fragment_begin)
            .split('( directLight.visible && receiveShadow ) ?')
            .join('( directLight.visible && receiveShadow && vGemParams.x < 0.5 ) ?');

        if (useTint) shader.fragmentShader = shader.fragmentShader
            .replace('#include <color_fragment>',
                '#include <color_fragment>\n\tdiffuseColor.rgb = mix( diffuseColor.rgb, vRegionTint, step( 0.3, vRegionSurf.w ) );')
            // Glow-region surfaces emit their own colour, so the source area reads hot under ACES
            // instead of a passive tint. (w: 0 none, 0.6 tint, 0.86–1.0 tint + emissive whose
            // gain is encoded in the band — the surface glows at its light's actual level.)
            .replace('#include <emissivemap_fragment>',
                '#include <emissivemap_fragment>\n\ttotalEmissiveRadiance += vRegionTint * ( step( 0.85, vRegionSurf.w ) * ( vRegionSurf.w - 0.86 ) / 0.14 * 6.0 );');
    };
    const prevKey = material.customProgramCacheKey;
    material.customProgramCacheKey = () =>
        (prevKey ? prevKey.call(material) : '') + '|regionsurf-gem' + (useTint ? '+tint' : '');
    return material;
}

function buildRegionColors() {
    return [NEUTRAL, ...regionPalette.map(r => new THREE.Color(r.color))];
}

function resetRegions() {
    regionIndex = null;
    regionColorAttr = null;
    regionSurfAttr = null;
    gemFlagAttr = null;
    paintGrid = null;
    fillTopology = null; // adjacency belongs to the old geometry
    highlightAttr = null;
    isolated = -1;
    if (overlayMesh) { scene.remove(overlayMesh); overlayMesh.material.dispose(); overlayMesh = null; }
    if (fillPreviewLine) {
        scene.remove(fillPreviewLine);
        fillPreviewLine.geometry.dispose();
        fillPreviewLine.material.dispose();
        fillPreviewLine = null;
    }
    fillPreview.face = -1;
}

// The on-model glow: a second mesh sharing the model geometry, drawn additively over the LIT render
// so the isolated region lights up in place. A per-vertex aHighlight (0/1) selects which vertices
// glow; depthFunc LEQUAL + no depth write lets it sit exactly on the surface without z-fighting.
function ensureHighlightOverlay() {
    if (!mesh) return;
    const count = mesh.geometry.attributes.position.count;
    if (!highlightAttr || highlightAttr.count !== count) {
        highlightAttr = new THREE.Float32BufferAttribute(new Float32Array(count), 1);
        highlightAttr.setUsage(THREE.DynamicDrawUsage);
        mesh.geometry.setAttribute('aHighlight', highlightAttr);
    }
    if (!overlayMesh) {
        const material = new THREE.ShaderMaterial({
            uniforms: { uColor: { value: new THREE.Color('#ffffff') } },
            // A solid tint plus a bright fresnel rim, so the region reads as "glowing" even on a
            // bright metallic look where a flat additive tint would wash out.
            vertexShader:
                'attribute float aHighlight;\nvarying float vH;\nvarying vec3 vN;\nvarying vec3 vViewPos;\n' +
                'void main() {\n' +
                '  vH = aHighlight;\n' +
                '  vec4 mv = modelViewMatrix * vec4( position, 1.0 );\n' +
                '  vViewPos = mv.xyz;\n' +
                '  vN = normalMatrix * normal;\n' +
                '  gl_Position = projectionMatrix * mv;\n' +
                '}',
            fragmentShader:
                'varying float vH;\nvarying vec3 vN;\nvarying vec3 vViewPos;\nuniform vec3 uColor;\n' +
                'void main() {\n' +
                '  if ( vH < 0.5 ) discard;\n' +
                '  float fres = pow( 1.0 - max( dot( normalize( vN ), normalize( -vViewPos ) ), 0.0 ), 2.0 );\n' +
                // Alpha-blend (not additive) so the region reads on a bright surface too: it tints
                // toward the region colour, opaque at the fresnel rim for a lit-up edge.
                '  gl_FragColor = vec4( uColor, clamp( 0.55 + 0.45 * fres, 0.0, 1.0 ) );\n' +
                '}',
            transparent: true,
            depthWrite: false,
            // Shares the model's exact geometry; nudge toward the camera so the tint wins the depth
            // test on the front surface but stays occluded on the far side.
            polygonOffset: true,
            polygonOffsetFactor: -1,
            polygonOffsetUnits: -1,
        });
        overlayMesh = new THREE.Mesh(mesh.geometry, material);
        overlayMesh.renderOrder = 2;      // after the lit model
        overlayMesh.frustumCulled = false;
        scene.add(overlayMesh);
    }
    overlayMesh.visible = false;
}

/// A uniform grid bucketing vertices by cell, so the brush only tests vertices near the hit point
/// instead of the whole mesh — the difference between a smooth brush and a frozen one on dense STLs.
/// Built lazily and invalidated whenever the geometry moves (load, reorient).
function buildPaintGrid() {
    const pos = mesh.geometry.attributes.position;
    const n = pos.count;
    const bb = new THREE.Box3().setFromBufferAttribute(pos);
    const size = new THREE.Vector3();
    bb.getSize(size);

    const perAxis = Math.min(100, Math.max(8, Math.round(Math.cbrt(n) / 2)));
    const cell = Math.max(1e-4, Math.max(size.x, size.y, size.z) / perAxis);
    const nx = Math.max(1, Math.floor(size.x / cell) + 1);
    const ny = Math.max(1, Math.floor(size.y / cell) + 1);
    const nz = Math.max(1, Math.floor(size.z / cell) + 1);
    const cellCount = nx * ny * nz;

    const cellOf = (x, y, z) => {
        const ix = Math.min(nx - 1, Math.max(0, Math.floor((x - bb.min.x) / cell)));
        const iy = Math.min(ny - 1, Math.max(0, Math.floor((y - bb.min.y) / cell)));
        const iz = Math.min(nz - 1, Math.max(0, Math.floor((z - bb.min.z) / cell)));
        return (ix * ny + iy) * nz + iz;
    };

    // Counting sort into a flat items array (recompute the cell index rather than store it per vertex).
    const start = new Int32Array(cellCount + 1);
    for (let i = 0; i < n; i++) start[cellOf(pos.getX(i), pos.getY(i), pos.getZ(i)) + 1]++;
    for (let c = 0; c < cellCount; c++) start[c + 1] += start[c];
    const cursor = start.slice(0, cellCount);
    const items = new Int32Array(n);
    for (let i = 0; i < n; i++) items[cursor[cellOf(pos.getX(i), pos.getY(i), pos.getZ(i))]++] = i;

    paintGrid = { bb, cell, nx, ny, nz, start, items };
}

function ensureRegionBuffers() {
    if (!mesh) return;
    const count = mesh.geometry.attributes.position.count;
    if (regionIndex && regionIndex.length === count && regionColorAttr) return;
    regionIndex = new Uint8Array(count);
    const colors = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) { colors[i * 3] = NEUTRAL.r; colors[i * 3 + 1] = NEUTRAL.g; colors[i * 3 + 2] = NEUTRAL.b; }
    regionColorAttr = new THREE.BufferAttribute(colors, 3);
    mesh.geometry.setAttribute('color', regionColorAttr);
    // Normalized bytes (not floats): 4 bytes/vertex keeps the cost negligible on dense prints.
    regionSurfAttr = new THREE.BufferAttribute(new Uint8Array(count * 4), 4, true);
    regionSurfAttr.setUsage(THREE.DynamicDrawUsage);
    mesh.geometry.setAttribute('aRegionSurf', regionSurfAttr);
    // Three normalized bytes per vertex: x = gem flag, y/z = the slot's window/glow encoded into
    // [0,1] (see gemEncode) — per-REGION gem character on the GPU for +2 bytes/vertex.
    gemFlagAttr = new THREE.BufferAttribute(new Uint8Array(count * 3), 3, true);
    gemFlagAttr.setUsage(THREE.DynamicDrawUsage);
    mesh.geometry.setAttribute('aGemFlag', gemFlagAttr);
}

// Encode/decode ranges for the gem attribute — the GLSL decode in applyRegionSurfacePatch MUST
// match: power spans [0.2, 4.0], gain spans [0, 6].
function gemEncode(slot) {
    if (slot?.material !== 'gem') return [0, 0, 0];
    const power = Math.min(4, Math.max(0.2, slot.gemPower ?? gemParams.power));
    const gain = Math.min(6, Math.max(0, slot.gemGain ?? gemParams.gain));
    return [1, (power - 0.2) / 3.8, gain / 6];
}

function repaintColors() {
    if (!regionColorAttr) return;
    // Palette lookups hoisted: this loops over every vertex of a multi-million-vertex print.
    const surf = regionPalette.map((_, k) => surfFor(k + 1));
    const gems = regionPalette.map(r => gemEncode(r));
    const noGem = [0, 0, 0];
    for (let i = 0; i < regionIndex.length; i++) {
        const idx = regionIndex[i];
        const c = regionThreeColors[idx] ?? NEUTRAL;
        regionColorAttr.setXYZ(i, c.r, c.g, c.b);
        const s = idx > 0 ? (surf[idx - 1] ?? [0, 0, 0, 0]) : [0, 0, 0, 0];
        regionSurfAttr.setXYZW(i, s[0], s[1], s[2], s[3]);
        const g = idx > 0 ? (gems[idx - 1] ?? noGem) : noGem;
        gemFlagAttr.setXYZ(i, g[0], g[1], g[2]);
    }
    regionColorAttr.needsUpdate = true;
    regionSurfAttr.needsUpdate = true;
    gemFlagAttr.needsUpdate = true;
}

function paintAt(worldPoint) {
    if (!regionColorAttr) return;
    if (!paintGrid) buildPaintGrid();
    regionsDirty = true;

    const local = mesh.worldToLocal(worldPoint.clone());
    const pos = mesh.geometry.attributes.position;
    const r = brush.radius * modelRadius, r2 = r * r;
    const paintR = strokeRegion; // brush.region for a left stroke, 0 (erase) for a right-drag
    const c = regionThreeColors[paintR] ?? NEUTRAL;
    const s = surfFor(paintR); // the region's own surface (or none), baked alongside its colour
    const gemVals = paintR > 0 ? gemEncode(regionPalette[paintR - 1]) : [0, 0, 0]; // gem params ride per-stroke too
    const highlightVal = paintR === isolated ? 1 : 0; // keep an active glow in sync with new paint

    const g = paintGrid, bb = g.bb;
    const cellRange = (v, lo, hi) => Math.min(hi - 1, Math.max(0, Math.floor((v - lo) / g.cell)));
    const ix0 = cellRange(local.x - r, bb.min.x, g.nx), ix1 = cellRange(local.x + r, bb.min.x, g.nx);
    const iy0 = cellRange(local.y - r, bb.min.y, g.ny), iy1 = cellRange(local.y + r, bb.min.y, g.ny);
    const iz0 = cellRange(local.z - r, bb.min.z, g.nz), iz1 = cellRange(local.z + r, bb.min.z, g.nz);

    for (let ix = ix0; ix <= ix1; ix++)
    for (let iy = iy0; iy <= iy1; iy++)
    for (let iz = iz0; iz <= iz1; iz++) {
        const cellId = (ix * g.ny + iy) * g.nz + iz;
        for (let k = g.start[cellId]; k < g.start[cellId + 1]; k++) {
            const i = g.items[k];
            const dx = pos.getX(i) - local.x, dy = pos.getY(i) - local.y, dz = pos.getZ(i) - local.z;
            if (dx * dx + dy * dy + dz * dz <= r2) {
                if (strokeDiff && !strokeDiff.has(i)) strokeDiff.set(i, regionIndex[i]);
                regionIndex[i] = paintR;
                regionColorAttr.setXYZ(i, c.r, c.g, c.b);
                regionSurfAttr.setXYZW(i, s[0], s[1], s[2], s[3]);
                gemFlagAttr.setXYZ(i, gemVals[0], gemVals[1], gemVals[2]);
                if (highlightAttr) highlightAttr.setX(i, highlightVal);
            }
        }
    }
    regionColorAttr.needsUpdate = true;
    regionSurfAttr.needsUpdate = true;
    gemFlagAttr.needsUpdate = true;
    if (highlightAttr && isolated >= 0) highlightAttr.needsUpdate = true;
}

// ----- Fill tool -------------------------------------------------------------------------------
// Click-to-fill: spreads from the clicked face across the surface while faces keep pointing within
// brush.fillAngle degrees of the clicked face (density-independent, unlike per-edge angles on dense
// prints), and never crosses a truly sharp edge (a real part boundary). Face adjacency is built
// once per mesh from welded vertex positions.

function buildFillTopology() {
    const pos = mesh.geometry.attributes.position;
    const tris = pos.count / 3;

    // Weld coincident vertices so shared edges connect faces (STL soup shares nothing by index).
    let minX = Infinity, minY = Infinity, minZ = Infinity, maxDim = 0;
    for (let i = 0; i < pos.count; i++) {
        minX = Math.min(minX, pos.getX(i)); minY = Math.min(minY, pos.getY(i)); minZ = Math.min(minZ, pos.getZ(i));
    }
    maxDim = modelRadius * 2 || 1;
    const tol = Math.max(maxDim * 1e-5, 1e-9);
    const weldMap = new Map();
    const welded = new Int32Array(pos.count);
    let unique = 0;
    for (let i = 0; i < pos.count; i++) {
        // 2^26-safe packing via multiplication (JS bit-ops are 32-bit only).
        const qx = Math.round((pos.getX(i) - minX) / tol);
        const qy = Math.round((pos.getY(i) - minY) / tol);
        const qz = Math.round((pos.getZ(i) - minZ) / tol);
        const key = (qx * 2097152 + qy) * 2097152 + qz;
        let id = weldMap.get(key);
        if (id === undefined) { id = unique++; weldMap.set(key, id); }
        welded[i] = id;
    }

    // Face normals.
    const normals = new Float32Array(tris * 3);
    for (let t = 0; t < tris; t++) {
        const i = t * 3;
        const ux = pos.getX(i + 1) - pos.getX(i), uy = pos.getY(i + 1) - pos.getY(i), uz = pos.getZ(i + 1) - pos.getZ(i);
        const vx = pos.getX(i + 2) - pos.getX(i), vy = pos.getY(i + 2) - pos.getY(i), vz = pos.getZ(i + 2) - pos.getZ(i);
        let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
        const len = Math.hypot(nx, ny, nz) || 1;
        normals[t * 3] = nx / len; normals[t * 3 + 1] = ny / len; normals[t * 3 + 2] = nz / len;
    }

    // Edge → adjacent face pairs, as compact adjacency lists. Each entry also carries the shared
    // edge's two soup-vertex indices so the fill PREVIEW can draw the boundary it stops at.
    const heads = new Int32Array(tris).fill(-1);
    const next = new Int32Array(tris * 6);
    const targets = new Int32Array(tris * 6);
    const edgeA = new Int32Array(tris * 6);
    const edgeB = new Int32Array(tris * 6);
    let cursor = 0;
    const link = (f1, f2, va, vb) => {
        targets[cursor] = f2; edgeA[cursor] = va; edgeB[cursor] = vb; next[cursor] = heads[f1]; heads[f1] = cursor++;
        targets[cursor] = f1; edgeA[cursor] = va; edgeB[cursor] = vb; next[cursor] = heads[f2]; heads[f2] = cursor++;
    };
    const edgeMap = new Map();
    for (let t = 0; t < tris; t++) {
        for (let k = 0; k < 3; k++) {
            const va = t * 3 + k, vb = t * 3 + (k + 1) % 3;
            const a = welded[va], b = welded[vb];
            if (a === b) continue;
            const key = (a < b ? a : b) * 67108864 + (a < b ? b : a);
            const other = edgeMap.get(key);
            if (other === undefined) edgeMap.set(key, va); // remember one side's soup edge start
            else if (other >= 0) { link((other / 3) | 0, t, other, ((other % 3) === 2 ? other - 2 : other + 1)); edgeMap.set(key, -1); }
        }
    }
    fillTopology = {
        tris, normals, heads, next, targets, edgeA, edgeB,
        visitStamp: new Int32Array(tris), stamp: 0, fillList: new Int32Array(Math.min(tris, 1 << 20)),
    };
}

/// The set of faces a fill from seedFace would cover, marked in visitStamp with a fresh stamp.
/// Returns the count; the face ids are in fillTopology.fillList[0..count-1].
function computeFillSet(seedFace) {
    const topo = fillTopology;
    const { tris, normals, heads, next, targets } = topo;
    if (seedFace < 0 || seedFace >= tris) return 0;

    const cosCone = Math.cos(Math.min(Math.max(brush.fillAngle, 1), 90) * Math.PI / 180);
    // Only near-perpendicular edges stop the fill outright (real part boundaries sit at ~90°);
    // sculpted detail like scales folds up to ~80° face-to-face and must not fence the fill.
    const cosSharp = Math.cos(88 * Math.PI / 180);
    const sx = normals[seedFace * 3], sy = normals[seedFace * 3 + 1], sz = normals[seedFace * 3 + 2];

    const stamp = ++topo.stamp;
    const visited = topo.visitStamp;
    let list = topo.fillList;
    let count = 0;
    const push = f => {
        if (count === list.length) {
            const bigger = new Int32Array(Math.min(tris, list.length * 2));
            bigger.set(list);
            topo.fillList = list = bigger;
        }
        list[count++] = f;
    };

    visited[seedFace] = stamp;
    push(seedFace);
    let head = 0;
    while (head < count) {
        const f = list[head++];
        const fx = normals[f * 3], fy = normals[f * 3 + 1], fz = normals[f * 3 + 2];
        for (let e = heads[f]; e !== -1; e = next[e]) {
            const n = targets[e];
            if (visited[n] === stamp) continue;
            const nx = normals[n * 3], ny = normals[n * 3 + 1], nz = normals[n * 3 + 2];
            if (nx * sx + ny * sy + nz * sz < cosCone) continue;  // left the clicked facing
            if (nx * fx + ny * fy + nz * fz < cosSharp) continue; // crossed a hard part edge
            visited[n] = stamp;
            push(n);
        }
    }
    return count;
}

/// Fills starting at a face (raycast faceIndex) with strokeRegion. Returns faces filled.
function fillFromFace(seedFace) {
    if (!regionColorAttr) return 0;
    if (!fillTopology) buildFillTopology();
    const count = computeFillSet(seedFace);
    if (count === 0) return 0;

    const paintR = strokeRegion;
    const c = regionThreeColors[paintR] ?? NEUTRAL;
    const s = surfFor(paintR);
    const highlightVal = paintR === isolated ? 1 : 0;
    const list = fillTopology.fillList;

    for (let idx = 0; idx < count; idx++) {
        const t = list[idx];
        for (let k = 0; k < 3; k++) {
            const i = t * 3 + k;
            if (strokeDiff && !strokeDiff.has(i)) strokeDiff.set(i, regionIndex[i]);
            regionIndex[i] = paintR;
            regionColorAttr.setXYZ(i, c.r, c.g, c.b);
            regionSurfAttr.setXYZW(i, s[0], s[1], s[2], s[3]);
            if (highlightAttr) highlightAttr.setX(i, highlightVal);
        }
    }

    regionColorAttr.needsUpdate = true;
    regionSurfAttr.needsUpdate = true;
    if (highlightAttr && isolated >= 0) highlightAttr.needsUpdate = true;
    regionsDirty = true;
    return count;
}

// ----- Fill preview: marching-ants boundary of the area a click would fill --------------------

let fillPreviewLine = null;                       // dashed LineSegments overlay
const fillPreview = { face: -1, angle: -1, time: 0 };

function hideFillPreview() {
    if (fillPreviewLine) fillPreviewLine.visible = false;
    fillPreview.face = -1;
}

/// Recomputes the dashed boundary for a hover face (throttled by the caller).
function updateFillPreview(seedFace) {
    if (!fillTopology) buildFillTopology();
    const topo = fillTopology;
    const count = computeFillSet(seedFace);
    if (count === 0) { hideFillPreview(); return; }

    // Boundary = fill-set edges whose neighbour is outside the set; lift each segment slightly
    // along the face normal so the line doesn't z-fight the surface.
    const pos = mesh.geometry.attributes.position;
    const stamp = topo.stamp;
    const lift = modelRadius * 0.004;
    const points = [];
    for (let idx = 0; idx < count; idx++) {
        const f = topo.fillList[idx];
        const nx = topo.normals[f * 3] * lift, ny = topo.normals[f * 3 + 1] * lift, nz = topo.normals[f * 3 + 2] * lift;
        for (let e = topo.heads[f]; e !== -1; e = topo.next[e]) {
            if (topo.visitStamp[topo.targets[e]] === stamp) continue; // interior edge
            const a = topo.edgeA[e], b = topo.edgeB[e];
            points.push(pos.getX(a) + nx, pos.getY(a) + ny, pos.getZ(a) + nz,
                        pos.getX(b) + nx, pos.getY(b) + ny, pos.getZ(b) + nz);
        }
    }

    if (points.length === 0) { hideFillPreview(); return; }
    if (!fillPreviewLine) {
        const material = new THREE.LineDashedMaterial({
            color: 0xffffff, dashSize: modelRadius * 0.02, gapSize: modelRadius * 0.012,
            depthTest: false, transparent: true, opacity: 0.95, toneMapped: false,
        });
        fillPreviewLine = new THREE.LineSegments(new THREE.BufferGeometry(), material);
        fillPreviewLine.renderOrder = 4; // over the model, the glow overlay and the brush ring
        fillPreviewLine.frustumCulled = false;
        scene.add(fillPreviewLine);
    }
    fillPreviewLine.geometry.dispose();
    fillPreviewLine.geometry = new THREE.BufferGeometry();
    fillPreviewLine.geometry.setAttribute('position', new THREE.Float32BufferAttribute(points, 3));
    fillPreviewLine.computeLineDistances(); // dashes need cumulative distances
    fillPreviewLine.visible = true;
}

/// Dev/tuning hook: run a fill outside the pointer pipeline. Returns the filled face count.
export function __debugFill(faceIndex, region, angleDeg) {
    ensureRegionBuffers();
    strokeRegion = region | 0;
    brush.fillAngle = angleDeg;
    strokeDiff = null;
    return fillFromFace(faceIndex | 0);
}

/// Enter/leave paint mode. Paint mode swaps to a LIT vertex-colour material — so the model keeps its
/// shape and surface detail while the coloured regions tint it — rather than a flat unlit fill.
export function setPaintMode(on) {
    paintMode = !!on;
    if (!mesh) return;
    if (paintMode) {
        ensureRegionBuffers();
        repaintColors();
        // The look material may predate the region buffers (unpainted model) — rebuild it now so it
        // carries the region-surface patch when we swap back on exit.
        if (lookMaterial && !lookMaterial.userData.regionPatch && regionSurfAttr) rebuildLookMaterial();
        // MeshStandard (not MeshBasic) so the studio lights still shade the geometry: you can read
        // scales/folds and where the brush is landing. Vertex colours multiply through as the regions.
        // The region-surface patch (no tint — vertex colours already carry the region colour) makes a
        // metal area gleam while you paint, so material changes give immediate feedback.
        mesh.material = ensurePaintMaterial();
    } else {
        // Keep any active region glow — showing it on the LIT model is the whole point.
        if (brushCursor) brushCursor.visible = false;
        hideFillPreview();
        mesh.material = lookMaterial ?? rebuildLookMaterial(); // swap, don't rebuild — the program stays warm
        refreshRegionLights(); // painting may have moved/recoloured a glow light's region
    }
}

export function getPaintMode() { return paintMode; }

export function setBrush(opts) {
    if (opts.region !== undefined) brush.region = opts.region | 0;
    if (opts.radius !== undefined) brush.radius = Math.max(0.005, Math.min(0.4, opts.radius)); // floor low enough for edge cleanup
    if (opts.mode !== undefined) brush.mode = opts.mode === 'fill' ? 'fill' : 'brush';
    if (opts.fillAngle !== undefined) {
        brush.fillAngle = Math.max(1, Math.min(90, opts.fillAngle));
        fillPreview.angle = -1;                                   // slider moved: preview is stale
        if (lastPaintHover) pendingPaintMove = lastPaintHover;    // refresh without a mouse move
    }
    if (brush.mode === 'fill') {
        if (brushCursor) brushCursor.visible = false; // the ring is brush-sized; misleading for fill
    } else {
        hideFillPreview();
    }
}

/// Glow one region in place on whatever's showing (works on the LIT look, not just the flat view).
export function highlightRegion(index) {
    isolated = index | 0;
    if (!regionIndex) return;
    ensureHighlightOverlay();
    for (let i = 0; i < regionIndex.length; i++) highlightAttr.setX(i, regionIndex[i] === isolated ? 1 : 0);
    highlightAttr.needsUpdate = true;
    const col = regionThreeColors[isolated] ?? NEUTRAL;
    overlayMesh.material.uniforms.uColor.value.copy(col);
    overlayMesh.visible = true;
}

export function showAllRegions() {
    isolated = -1;
    if (overlayMesh) overlayMesh.visible = false;
}

export function clearRegions() {
    if (!regionIndex) return;
    const diff = new Map(); // so Clear is undoable too
    for (let i = 0; i < regionIndex.length; i++) if (regionIndex[i] !== 0) diff.set(i, regionIndex[i]);
    pushPaintUndo(diff);
    regionIndex.fill(0);
    isolated = -1;
    if (highlightAttr) { highlightAttr.array.fill(0); highlightAttr.needsUpdate = true; }
    if (overlayMesh) overlayMesh.visible = false;
    repaintColors();
    regionsDirty = true;
    notifyChanged();
}

export function setRegionPalette(json) {
    regionPalette = JSON.parse(json);
    regionThreeColors = buildRegionColors();
    // Rebake unconditionally (not just in paint mode): colours and surfaces feed the LIT look too,
    // so changing an area's material (or colour) shows on the model immediately.
    if (regionIndex) {
        repaintColors();
        refreshRegionLights(); // a recoloured slot recolours its glow light
        regionsDirty = true; // the palette persists inside the mask blob
        // Attribute-only updates reliably appear on the paint material, but have been observed to
        // NOT appear while the LOOK material keeps rendering (headless repro; cause in three's
        // binding internals unproven). Outside paint mode the palette changes only rarely (Revert),
        // so rebuild the look there — the proven-correct path — instead of trusting the update.
        if (!paintMode && mesh) rebuildLookMaterial();
    }
    if (isolated >= 0) highlightRegion(isolated); // refresh the glow colour
}

export function getRegionPalette() {
    return JSON.stringify(regionPalette);
}

/// Dev/tuning: rebuild the model material from explicit pieces, to bisect patch interactions.
/// Replaces the cached look material so subsequent paint-mode toggles don't clobber it.
export function __debugBuildMaterial(opts) {
    if (!mesh) return 'no mesh';
    let material = new THREE.MeshPhysicalMaterial({
        color: new THREE.Color(appearance.color),
        metalness: appearance.metalness, roughness: appearance.roughness,
        clearcoat: appearance.clearcoat,
        envMapIntensity: (appearance.envIntensity ?? 0.4) * ambientEnvScale()
    });
    if (opts.studio) material = applyStudioShaderPatches(material);
    if (opts.region) material = applyRegionSurfacePatch(material, opts.tint !== false);
    lookMaterial?.dispose();
    lookMaterial = material;
    lookMaterial.userData.regionPatch = !!opts.region;
    mesh.material = lookMaterial;
    return material.customProgramCacheKey ? material.customProgramCacheKey() : 'default-key';
}

/// Dev/tuning introspection: is the region-surface pipeline live on the current material?
export function getRegionDebug() {
    let firstUse = -1;
    if (regionSurfAttr) {
        const a = regionSurfAttr.array;
        for (let i = 0; i < a.length; i += 4) if (a[i] !== 0) { firstUse = i / 4; break; }
    }
    const shader = mesh?.material?.userData?.studioShader;
    return JSON.stringify({
        paintMode,
        hasSurfAttr: !!regionSurfAttr,
        firstUseVertex: firstUse,
        materialKey: mesh?.material?.customProgramCacheKey?.() ?? null,
        paletteMaterials: regionPalette.map(r => r.material ?? null),
        // Did the region patch actually land in the compiled source?
        vertHasAttr: shader?.vertexShader?.includes('aRegionSurf') ?? null,
        fragHasMetalMix: shader?.fragmentShader?.includes('vRegionSurf.y') ?? null,
        fragHasTint: shader?.fragmentShader?.includes('vRegionTint') ?? null,
        // …and does the LINKED program actually use the attribute, or was it eliminated?
        activeAttrs: (() => {
            try {
                const prog = renderer?.properties?.get(mesh.material)?.currentProgram;
                return prog ? Object.keys(prog.getAttributes()) : null;
            } catch { return null; }
        })()
    });
}

/// Painted-vertex count per region index (0..N) so the panel can show what's covered.
export function getRegionCounts() {
    const counts = new Array(regionPalette.length + 1).fill(0);
    if (regionIndex) for (let i = 0; i < regionIndex.length; i++) counts[regionIndex[i]]++;
    return JSON.stringify(counts);
}

/// The whole mask as a JSON blob (or null when nothing is painted) — persisted per model.
export function getRegions() {
    if (!regionIndex || !regionIndex.some(v => v !== 0)) return null;
    return JSON.stringify({ version: 1, count: regionIndex.length, palette: regionPalette, data: u8ToBase64(regionIndex) });
}

export function setRegions(json) {
    if (!json || !mesh) return;
    let data;
    try { data = JSON.parse(json); } catch { return; }
    if (!data || data.count !== mesh.geometry.attributes.position.count) return; // stale mask; ignore
    if (Array.isArray(data.palette)) { regionPalette = data.palette; regionThreeColors = buildRegionColors(); }
    ensureRegionBuffers();
    regionIndex.set(base64ToU8(data.data).subarray(0, regionIndex.length));
    repaintColors(); // always: region surfaces show on the lit look, not only in paint mode
    // The look material was built before the region buffers existed — rebuild it so the
    // region-surface patch attaches and saved areas with their own material show right away.
    if (lookMaterial && !lookMaterial.userData.regionPatch) rebuildLookMaterial();
    refreshRegionLights(); // glow lights follow the loaded mask
    regionsDirty = false; // what was just loaded IS the saved state
}

/// Unsaved mask/palette edits? The panel shows Save/Revert while true.
export function isRegionsDirty() {
    return regionsDirty;
}

/// The panel persisted the mask — edits up to this point are no longer "unsaved".
export function markRegionsSaved() {
    regionsDirty = false;
}

/// The panel applied a mask that is NOT what's saved (auto-detect proposal) — offer Save/Revert.
export function markRegionsEdited() {
    regionsDirty = true;
}

function u8ToBase64(u8) {
    let out = '';
    const chunk = 0x8000; // fromCharCode.apply can't take a whole 1M array at once
    for (let i = 0; i < u8.length; i += chunk)
        out += String.fromCharCode.apply(null, u8.subarray(i, i + chunk));
    return btoa(out);
}

function base64ToU8(b64) {
    const bin = atob(b64);
    const u8 = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    return u8;
}

// ---------------------------------------------------------------------------------------------
// Imported STLs (scene props) — bring extra models in to light a composed scene (base + mini +
// scenery). Each is a movable/rotatable mesh, lit like the model but WITHOUT region painting.
// Session-scoped for now: save a screenshot to keep the composition (transforms aren't persisted
// with the light rig yet).
// ---------------------------------------------------------------------------------------------

export function addStl(bytes, name, modelId, transformJson) {
    if (!renderer) return 0;
    const geometry = new STLLoader().parse(bytes.buffer ?? bytes);
    smoothSoupNormals(geometry); // same welded normals as the primary model
    geometry.rotateX(-Math.PI / 2);                       // match the model's Z-up→Y-up convention
    geometry.scale(modelScale, modelScale, modelScale);   // keep size relative to the primary model
    geometry.center();                                    // so it rotates about its own centre
    geometry.computeBoundingSphere();

    const propMesh = new THREE.Mesh(geometry, buildMaterial(false));
    propMesh.castShadow = true;
    propMesh.receiveShadow = true;

    let transform = null;
    if (transformJson) { try { transform = JSON.parse(transformJson); } catch { /* fresh placement */ } }
    if (transform?.position && transform?.quaternion) {
        propMesh.position.fromArray(transform.position);   // restore a saved scene
        propMesh.quaternion.fromArray(transform.quaternion);
    } else {
        // Fresh import: drop it beside the model at ground level, not buried inside the sculpt.
        const r = geometry.boundingSphere.radius;
        propMesh.position.set(modelRadius + r, r, 0);
    }
    scene.add(propMesh);

    const id = ++propSeq;
    propMesh.userData.propId = id;
    props.set(id, { mesh: propMesh, name: name || `part ${id}`, modelId: modelId || null });
    if (!transform) selectProp(id); // fresh add selects; a restored prop does not
    requestShadowUpdate();
    return id;
}

function selectProp(id) {
    const entry = props.get(id);
    if (!entry) return;
    if (selectedLightId !== null) selectLight(null); // the two gizmos are mutually exclusive
    selectedPropId = id;
    propControl.attach(entry.mesh);
}

function deselectProp() {
    selectedPropId = null;
    propControl.detach();
}

/// Select a prop from the panel list.
export function selectPropById(id) { selectProp(id | 0); notifyChanged(); }

/// 'translate' (move) or 'rotate' (turn) for the prop gizmo.
export function setPropMode(mode) {
    propControl.setMode(mode === 'rotate' ? 'rotate' : 'translate');
}

export function deleteProp(id) {
    const entry = props.get(id | 0);
    if (!entry) return;
    if (selectedPropId === (id | 0)) deselectProp();
    scene.remove(entry.mesh);
    entry.mesh.geometry.dispose();
    entry.mesh.material.dispose();
    props.delete(id | 0);
    requestShadowUpdate();
}

/// The scene's imported props for the panel: [{id, name, modelId, selected}].
export function listProps() {
    return JSON.stringify([...props.entries()].map(([id, e]) =>
        ({ id, name: e.name, modelId: e.modelId, selected: id === selectedPropId })));
}

/// The persistable scene: [{modelId, name, position, quaternion}] for props backed by a model.
export function getScene() {
    return JSON.stringify([...props.values()].filter(e => e.modelId).map(e => ({
        modelId: e.modelId,
        name: e.name,
        position: e.mesh.position.toArray(),
        quaternion: e.mesh.quaternion.toArray()
    })));
}

// ---------------------------------------------------------------------------------------------
// Persistence & capture
// ---------------------------------------------------------------------------------------------

export function getSetup() {
    return JSON.stringify({
        version: 3,
        appearance,
        orientation: orientation.toArray(),
        floor: { color: floorColor },
        shadows: shadowsEnabled,
        shadowTint: { color: shadowTintColor, strength: shadowStrength },
        specularLock: { on: specLock.on, position: specLock.camPos.toArray() },
        ambient: {
            intensity: ambientLevel,
            sky: '#' + (hemi?.color.getHexString() ?? '8fa3b4'),
            ground: '#' + (hemi?.groundColor.getHexString() ?? '4a4238')
        },
        lights: JSON.parse(getLightsJson()),
        camera: { position: camera.position.toArray(), target: controls.target.toArray() }
    });
}

export function applySetup(json) {
    captureUndo('load', false);
    const setup = JSON.parse(json);
    withUndoSuppressed(() => {
        if (setup.appearance) setAppearance(setup.appearance);
        if (setup.ambient) setEnvironmentLight(setup.ambient); // v1 rigs simply keep the default
        if (setup.floor) setFloor(setup.floor);
        if (setup.shadows !== undefined) setShadows(setup.shadows);
        if (setup.shadowTint) setShadowTint(setup.shadowTint);
        if (setup.specularLock) {
            specLock.on = !!setup.specularLock.on;
            if (setup.specularLock.position) specLock.camPos.fromArray(setup.specularLock.position);
        }
        if (setup.orientation && mesh) {
            // Bring the CURRENT geometry to the saved orientation, whatever it is now.
            const target = new THREE.Quaternion().fromArray(setup.orientation);
            applyOrientationDelta(target.multiply(orientation.clone().invert()));
        }
        clearLights();
        for (const l of setup.lights ?? [])
            addLight(l.type, l);
    });
    if (setup.camera) {
        camera.position.fromArray(setup.camera.position);
        controls.target.fromArray(setup.camera.target);
    }
    return getLightsJson();
}

/// Current ambient (room light) level, for the UI slider.
export function getAmbientIntensity() {
    return ambientLevel;
}

/// The whole appearance (shader/look fields, colour, PBR params) — the panel re-syncs from this
/// after a rig load or undo, so its controls never stomp a freshly restored look with stale values.
export function getAppearanceJson() {
    return JSON.stringify(appearance);
}

/// How many lights are rendering shadow maps right now (diagnostics; point casters cost 6x).
/// Dev/tuning: live shadow-system state, and a lever to force per-frame shadow updates.
export function __debugShadowState() {
    const casters = [];
    scene.traverse(o => {
        if (o.isLight && o.castShadow) casters.push({
            type: o.type, pos: o.position.toArray().map(v => +v.toFixed(1)),
            near: o.shadow.camera.near, far: o.shadow.camera.far,
            mapNull: !o.shadow.map, visible: o.visible, intensity: +o.intensity.toFixed(1)
        });
    });
    return JSON.stringify({
        casters,
        shadowMap: { enabled: renderer.shadowMap.enabled, autoUpdate: renderer.shadowMap.autoUpdate,
                     needsUpdate: renderer.shadowMap.needsUpdate, type: renderer.shadowMap.type },
        mesh: mesh ? { cast: mesh.castShadow, receive: mesh.receiveShadow, mat: mesh.material.type,
                       visible: mesh.visible } : null,
        ground: ground ? { receive: ground.receiveShadow } : null
    });
}

export function __debugShadowAuto(on) {
    renderer.shadowMap.autoUpdate = !!on;
    renderer.shadowMap.needsUpdate = true;
}

export function getShadowCasterCount() {
    let count = 0;
    for (const entry of lights.values())
        for (const light of entry.lights ?? []) if (light.castShadow) count++;
    return count;
}

export function screenshot() {
    // Captures exclude pure manipulation UI (move arrows, prop gizmos). The light-source orbs are
    // scene content — the root "light source dots" toggle decides whether they appear, same as in
    // the live view (a visible lantern orb in a render is a feature; toggle off for a clean plate).
    const arrowsVisible = axisControl?.visible ?? false;
    if (axisControl) axisControl.visible = false;
    const propArrowsVisible = propControl?.visible ?? false;
    if (propControl) propControl.visible = false;

    requestShadowUpdate();
    renderer.render(scene, camera);
    const dataUrl = renderer.domElement.toDataURL('image/png');

    if (axisControl) axisControl.visible = arrowsVisible;
    if (propControl) propControl.visible = propArrowsVisible;
    return dataUrl;
}

/// Captures the model from `count` evenly-spaced angles around the current view — same rig, same
/// look — so an image model gets several sides of the SAME sculpt to reason about its 3D form.
/// The FIRST shot is the current framing (the angle the mockup should be output from); the rest
/// orbit around it. The camera is restored exactly afterwards. Returns a JSON array of PNG data
/// URLs. Elevation and distance are held; only the azimuth changes, so every view frames the mini.
export function captureAngles(count = 4) {
    if (!renderer || !camera || !controls || count < 1) return JSON.stringify([screenshot()]);
    const target = controls.target.clone();
    const savedPos = camera.position.clone();
    const offset = camera.position.clone().sub(target);
    const radius = offset.length() || 1;
    const startAz = Math.atan2(offset.x, offset.z);
    const y = offset.y;                          // hold elevation
    const horiz = Math.hypot(offset.x, offset.z); // horizontal ring radius
    const shots = [];
    for (let i = 0; i < count; i++) {
        const az = startAz + (i / count) * Math.PI * 2;
        camera.position.set(target.x + horiz * Math.sin(az), target.y + y, target.z + horiz * Math.cos(az));
        camera.lookAt(target);
        camera.updateMatrixWorld();
        shots.push(screenshot());
    }
    camera.position.copy(savedPos);
    camera.lookAt(target);
    controls.update();
    renderer.render(scene, camera);
    return JSON.stringify(shots);
}
