// Chromatory Assembly (Architecture v2 §26c).
// Client-side workbench for assembling multi-part miniature kits: load several STLs, move and
// rotate them, snap mating surfaces together, and export one merged STL. Parts are loaded
// AS-EXPORTED (no recentering) because sculptors usually export parts in the original sculpt's
// coordinate space — for those kits, loading alone reconstructs the model.
//
// The snap is a local registration, not a global search: the user drags a part roughly into
// place and the fit is polished by complementary-surface ICP — point-to-plane, favoring pairs
// whose normals OPPOSE (mating faces look at each other), with penetration pushed out hard.
// No external geometry library: parts are area-weighted surface-sampled and the target's
// samples go into a uniform spatial hash for nearest-point queries.
import * as THREE from 'three';
import { STLLoader } from '../lib/three/STLLoader.js';
import { OrbitControls } from '../lib/three/OrbitControls.js';
import { TransformControls } from '../lib/three/TransformControls.js';
import { createViewCube } from './view-cube.js';

let renderer, scene, camera, controls, canvasEl;
let moveGizmo, rotateGizmo, viewCube;
let selectedId = null;

// Both gizmos attach to this pivot, positioned at the SELECTED PART'S bounding-box center —
// kit parts exported in place have their origin at the sculpt's origin, which can be nowhere
// near the part, and rotating about that swings the part across the scene. Gizmo deltas are
// re-applied to the mesh about the pivot point.
const pivot = new THREE.Object3D();
const pivotPrev = { position: new THREE.Vector3(), quaternion: new THREE.Quaternion() };
let autoSnap = false;
let changeCallback = null;   // UI hook: parts/selection changed from inside the module
let statusCallback = null;   // UI hook: progress text during long operations

const parts = new Map();     // id -> entry (live parts, in scene)
const partStore = new Map(); // id -> entry (includes removed parts, so undo can resurrect)
let nextPartId = 1;

const PALETTE = ['#8fb4d8', '#d8a878', '#9ad89a', '#d89ab4', '#c8c878', '#a89ad8', '#78c8c8', '#d88f8f'];

// ---------------------------------------------------------------------------------------------
// Undo (same snapshot pattern as the Light Studio): transforms + membership, geometry by ref.
// ---------------------------------------------------------------------------------------------

const undoStack = [];
const UNDO_LIMIT = 50;
let restoringUndo = false;
let undoSuppressDepth = 0;

function captureUndo(tag, coalesce) {
    if (restoringUndo || undoSuppressDepth > 0 || !renderer) return;
    const now = performance.now();
    const top = undoStack[undoStack.length - 1];
    if (coalesce && top && top.tag === tag && now - top.time < 1200) { top.time = now; return; }
    const states = [];
    for (const [id, entry] of parts) {
        entry.mesh.updateMatrix();
        states.push({ id, matrix: entry.mesh.matrix.toArray(), visible: entry.mesh.visible, scale: entry.scale });
    }
    undoStack.push({ tag, time: now, states });
    if (undoStack.length > UNDO_LIMIT) undoStack.shift();
}

export function undo() {
    const top = undoStack.pop();
    if (!top) return false;
    restoringUndo = true;
    try {
        const keep = new Set(top.states.map(s => s.id));
        for (const id of [...parts.keys()]) {
            if (!keep.has(id)) detachPart(id);
        }
        for (const state of top.states) {
            let entry = parts.get(state.id);
            if (!entry) {
                entry = partStore.get(state.id);
                if (!entry) continue;
                parts.set(state.id, entry);
                scene.add(entry.mesh);
            }
            entry.mesh.matrix.fromArray(state.matrix);
            entry.mesh.matrix.decompose(entry.mesh.position, entry.mesh.quaternion, entry.mesh.scale);
            entry.mesh.visible = state.visible;
            entry.scale = state.scale;
        }
        if (selectedId !== null && !parts.has(selectedId)) selectPart(null);
        syncPivot();
    }
    finally { restoringUndo = false; }
    return true;
}

// ---------------------------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------------------------

export function init(canvas) {
    canvasEl = canvas;
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x101315);

    camera = new THREE.PerspectiveCamera(45, 1, 0.1, 8000);
    camera.position.set(80, 90, 160);

    controls = new OrbitControls(camera, canvas);
    controls.enableDamping = true;
    controls.minDistance = 5;
    controls.maxDistance = 3000;

    // Simple neutral lighting — this is a workbench, not the light studio.
    scene.add(new THREE.HemisphereLight(0x9fb4c8, 0x4a4238, 0.75));
    const key = new THREE.DirectionalLight(0xffffff, 1.6);
    key.position.set(120, 220, 140);
    scene.add(key);
    const fill = new THREE.DirectionalLight(0x8899bb, 0.5);
    fill.position.set(-140, 60, -100);
    scene.add(fill);

    const grid = new THREE.GridHelper(600, 60, 0x2a3f55, 0x1c2126);
    grid.position.y = -0.01;
    scene.add(grid);

    // One COMBINED control: translate arrows and rotate rings live together (two instances on
    // the same pivot), so there is no mode to switch.
    scene.add(pivot);
    moveGizmo = new TransformControls(camera, canvas);
    moveGizmo.setMode('translate');
    moveGizmo.setSize(0.75);
    moveGizmo.addEventListener('objectChange', applyPivotChange);
    scene.add(moveGizmo);

    rotateGizmo = new TransformControls(camera, canvas);
    rotateGizmo.setMode('rotate');
    rotateGizmo.setSize(1.15);
    rotateGizmo.addEventListener('objectChange', applyPivotChange);
    scene.add(rotateGizmo);

    // A drag must be ONE action — move or rotate, never both. The two gizmos' pick regions
    // overlap (the translate centre handles sit inside the rotate gizmo's free-rotate sphere),
    // and TransformControls re-computes its hovered handle INSIDE its own pointerdown, so gating
    // on the hover state at press time missed any press that arrived without a fresh hover
    // (touch, fast clicks) — BOTH gizmos started dragging and the part moved and rotated at
    // once. Instead: the instant one gizmo starts dragging, the other is disabled for the rest
    // of the gesture (its pointerdown handler then ignores it entirely). The gizmos see the
    // pointerdown in registration order, so translate — created first — wins where they overlap.
    const onGizmoDrag = other => event => {
        controls.enabled = !event.value;
        other.enabled = !event.value;
        if (event.value) captureUndo('gizmo', false);
        else { syncPivot(); afterPartMoved(); }
    };
    moveGizmo.addEventListener('dragging-changed', onGizmoDrag(rotateGizmo));
    rotateGizmo.addEventListener('dragging-changed', onGizmoDrag(moveGizmo));

    resize();
    window.addEventListener('resize', resize);
    window.addEventListener('keydown', onKeyDown);
    canvas.addEventListener('pointerdown', onPointerDown);
    canvas.addEventListener('pointermove', onPointerMove);
    canvas.addEventListener('pointerup', onPointerUp);

    // Safety: a cancelled gesture (browser stole the pointer) never fires the gizmos' own
    // pointerup, so re-enable both here or one could stay locked out.
    const gizmoReset = () => { moveGizmo.enabled = true; rotateGizmo.enabled = true; };
    canvas.addEventListener('pointercancel', gizmoReset, true);

    viewCube = createViewCube(renderer, camera, controls, canvas);

    // Explicit clear: the view cube composites over the top-right corner after the main pass.
    renderer.autoClear = false;
    renderer.setAnimationLoop(() => {
        controls.update();
        renderer.clear();
        renderer.render(scene, camera);
        viewCube.render();
    });
}

export function dispose() {
    renderer?.setAnimationLoop(null);
    window.removeEventListener('resize', resize);
    window.removeEventListener('keydown', onKeyDown);
    viewCube?.dispose();
    viewCube = null;
    renderer?.dispose();
    parts.clear();
    partStore.clear();
    undoStack.length = 0;
    changeCallback = null;
    statusCallback = null;
}

function resize() {
    if (!canvasEl || !renderer) return;
    const w = canvasEl.clientWidth || 800;
    const h = canvasEl.clientHeight || 600;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    if (controls) controls.rotateSpeed = Math.min(2.5, Math.max(0.7, h / 550));
}

export function setChangeCallback(cb) { changeCallback = cb; }
export function setStatusCallback(cb) { statusCallback = cb; }

function notifyChanged() {
    if (!changeCallback) return;
    if (typeof changeCallback === 'function') changeCallback();
    else changeCallback.invokeMethodAsync('OnAssemblyChangedFromCanvas');
}

function status(text) {
    if (text) devlog('status', { text }); // the narration IS a good trace
    if (!statusCallback) return;
    if (typeof statusCallback === 'function') statusCallback(text);
    else statusCallback.invokeMethodAsync('OnAssemblyStatus', text);
}

/// Dev-build action trace (no-op unless dev-log.js registered the hook): rich, parameterized
/// events — anchor coordinates, fit results — that a generic click logger can't see.
function devlog(action, data) {
    window.__chromatoryDevLog?.('assembly:' + action, data);
}

function onKeyDown(event) {
    const target = event.target;
    if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' ||
                   target.tagName === 'SELECT' || target.isContentEditable)) return;

    if ((event.ctrlKey || event.metaKey) && !event.shiftKey && event.key.toLowerCase() === 'z') {
        event.preventDefault();
        // In scissor mode Ctrl+Z means "undo the last cut point" — the cut line lives in the
        // scissor session, not the parts undo stack, so route to the session's own undo.
        if (scissors) { scissorUndo().then(() => notifyChanged()); return; }
        if (undo()) notifyChanged();
    }
    else if (event.key === 'Escape' && scissors) {
        event.preventDefault();
        exitScissors();
        notifyChanged();
    }
    else if ((event.key === 'Delete' || event.key === 'Backspace') && selectedId !== null) {
        if (scissors) return; // deleting the part mid-cut is never what Delete meant
        event.preventDefault();
        removePart(selectedId);
        notifyChanged();
    }
}

// ---------------------------------------------------------------------------------------------
// Parts
// ---------------------------------------------------------------------------------------------

/// Loads one STL AS-EXPORTED — no recentering, so in-place kits assemble by themselves.
export function addPartFromBytes(bytes, name) {
    captureUndo('parts', false);
    const geometry = new STLLoader().parse(bytes.buffer ?? bytes);
    geometry.computeVertexNormals();
    geometry.computeBoundingSphere();
    geometry.computeBoundingBox();

    const id = nextPartId++;
    const color = PALETTE[(id - 1) % PALETTE.length];
    const material = new THREE.MeshStandardMaterial({
        color: new THREE.Color(color), roughness: 0.65, metalness: 0.05
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.userData.partId = id;

    const entry = {
        id, mesh, color,
        name: (name ?? `part ${id}`).replace(/\.stl$/i, ''),
        scale: 1,
        triangles: geometry.attributes.position.count / 3,
        localCenter: geometry.boundingBox.getCenter(new THREE.Vector3()),
        samples: null // lazy: built on first snap involving this part
    };
    parts.set(id, entry);
    partStore.set(id, entry);
    scene.add(mesh);
    if (parts.size <= 2) frameAll();
    return id;
}

/// Hooks a plain file input so shells (including Blazor) never marshal STL bytes themselves.
export function bindFileInput(elementId) {
    const el = document.getElementById(elementId);
    if (!el) return;
    el.addEventListener('change', async () => {
        // Per-file try/catch: a failed 300 MB read (memory pressure, revoked handle) must
        // surface as a status line, never as a silent nothing.
        for (const file of el.files) {
            try {
                status(`reading ${file.name}…`);
                addPartFromBytes(new Uint8Array(await file.arrayBuffer()), file.name);
                status('');
            }
            catch (error) {
                status(`couldn't read ${file.name}: ${error.message}`);
            }
            notifyChanged();
        }
        el.value = '';
    });
}

function detachPart(id) {
    const entry = parts.get(id);
    if (!entry) return;
    if (selectedId === id) selectPart(null);
    scene.remove(entry.mesh);
    parts.delete(id);
}

export function removePart(id) {
    if (!parts.has(id)) return;
    captureUndo('parts', false);
    detachPart(id); // geometry stays in partStore so undo can bring it back
}

export function setPartVisible(id, on) {
    const entry = parts.get(id);
    if (!entry) return;
    captureUndo('visible', false);
    entry.mesh.visible = !!on;
}

export function setPartScale(id, scale) {
    const entry = parts.get(id);
    if (!entry) return;
    captureUndo('scale:' + id, true);
    entry.scale = Math.min(20, Math.max(0.05, scale));
    entry.mesh.scale.setScalar(entry.scale);
    if (id === selectedId) syncPivot();
}

/// Repositions the pivot on the selected part's world bounding-box center and rebases the
/// delta tracking. Called on selection and after any non-gizmo transform of the part.
function syncPivot() {
    const entry = selectedId !== null ? parts.get(selectedId) : null;
    if (!entry) return;
    entry.mesh.updateMatrixWorld();
    pivot.position.copy(entry.localCenter).applyMatrix4(entry.mesh.matrixWorld);
    pivot.quaternion.copy(entry.mesh.quaternion);
    pivotPrev.position.copy(pivot.position);
    pivotPrev.quaternion.copy(pivot.quaternion);
}

/// Applies the pivot's movement since the last event to the mesh — rotation happens ABOUT the
/// pivot point, so parts turn in place no matter where their STL origin sits.
function applyPivotChange() {
    const entry = selectedId !== null ? parts.get(selectedId) : null;
    if (!entry) return;
    const deltaQ = pivot.quaternion.clone().multiply(pivotPrev.quaternion.clone().invert());
    const deltaPos = pivot.position.clone().sub(pivotPrev.position);

    entry.mesh.position.sub(pivotPrev.position).applyQuaternion(deltaQ).add(pivotPrev.position).add(deltaPos);
    entry.mesh.quaternion.premultiply(deltaQ);

    pivotPrev.position.copy(pivot.position);
    pivotPrev.quaternion.copy(pivot.quaternion);
}

export function selectPart(id) {
    const previous = selectedId !== null ? parts.get(selectedId) : null;
    if (previous) previous.mesh.material.emissive.set(0x000000);

    selectedId = id;
    const entry = id !== null ? parts.get(id) : null;
    if (entry) {
        entry.mesh.material.emissive.set(0x1a3a5c);
        syncPivot();
        moveGizmo.attach(pivot);
        rotateGizmo.attach(pivot);
    }
    else {
        selectedId = null;
        moveGizmo.detach();
        rotateGizmo.detach();
    }
}

export function getSelectedId() { return selectedId; }

/// Test hook: rotates the pivot by deg about Y and applies it exactly as a gizmo drag would.
export function debugPivotRotate(deg) {
    pivot.quaternion.premultiply(new THREE.Quaternion().setFromAxisAngle(
        new THREE.Vector3(0, 1, 0), THREE.MathUtils.degToRad(deg)));
    applyPivotChange();
}

/// Where the combined control currently sits (world), for tests and shells.
export function getGizmoAnchor() {
    return moveGizmo.object ? [...pivot.position.toArray()] : null;
}

export function listPartsJson() {
    const list = [];
    for (const [id, entry] of parts) {
        list.push({
            id, name: entry.name, color: entry.color, visible: entry.mesh.visible,
            scale: entry.scale, triangles: entry.triangles, selected: id === selectedId
        });
    }
    return JSON.stringify(list);
}

/// Test/persistence hooks: a part's full local matrix.
export function getPartMatrix(id) {
    const entry = parts.get(id);
    if (!entry) return null;
    entry.mesh.updateMatrix();
    return [...entry.mesh.matrix.toArray()];
}

export function setPartMatrix(id, array16) {
    const entry = parts.get(id);
    if (!entry) return;
    captureUndo('matrix:' + id, true);
    entry.mesh.matrix.fromArray(array16);
    entry.mesh.matrix.decompose(entry.mesh.position, entry.mesh.quaternion, entry.mesh.scale);
    entry.scale = entry.mesh.scale.x;
    if (id === selectedId) syncPivot();
}

export function frameAll() {
    if (parts.size === 0) return;
    const box = new THREE.Box3();
    for (const entry of parts.values()) box.expandByObject(entry.mesh);
    const center = box.getCenter(new THREE.Vector3());
    const radius = Math.max(10, box.getSize(new THREE.Vector3()).length() / 2);
    const dir = camera.position.clone().sub(controls.target).normalize();
    controls.target.copy(center);
    camera.position.copy(center).addScaledVector(dir, radius * 2.4);
    camera.near = radius / 100;
    camera.far = radius * 40;
    camera.updateProjectionMatrix();
}

// Free drag on the camera-facing plane (same interaction as the Light Studio's light handles).
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const dragPlane = new THREE.Plane();
const dragOffset = new THREE.Vector3();
let dragging = null;
let draggedSinceDown = false;

function setPointer(event) {
    const rect = canvasEl.getBoundingClientRect();
    pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
}

function onPointerDown(event) {
    if (moveGizmo?.dragging || rotateGizmo?.dragging) return;
    if (viewCube?.handlePointerDown(event)) return; // the corner widget owns its clicks
    setPointer(event);
    raycaster.setFromCamera(pointer, camera);

    // Scissor mode: clicks place cut anchors on the part being cut — no select, no drag.
    if (scissors) {
        const cutting = parts.get(scissors.partId);
        const hit = cutting ? raycaster.intersectObject(cutting.mesh, false)[0] : null;
        if (hit) scissorClick(hit.point);
        return;
    }

    const meshes = [...parts.values()].filter(e => e.mesh.visible).map(e => e.mesh);
    const hit = raycaster.intersectObjects(meshes, false)[0];
    if (!hit) {
        if (selectedId !== null) { selectPart(null); notifyChanged(); }
        return;
    }
    selectPart(hit.object.userData.partId);
    notifyChanged();
    captureUndo('drag', false);
    dragging = hit.object.userData.partId;
    draggedSinceDown = false;
    dragPlane.setFromNormalAndCoplanarPoint(
        camera.getWorldDirection(new THREE.Vector3()).negate(), hit.point);
    dragOffset.copy(hit.object.position).sub(hit.point);
    controls.enabled = false;
    canvasEl.setPointerCapture(event.pointerId);
}

function onPointerMove(event) {
    if (viewCube?.handlePointerMove(event)) return;
    if (dragging === null) return;
    setPointer(event);
    raycaster.setFromCamera(pointer, camera);
    const point = new THREE.Vector3();
    if (raycaster.ray.intersectPlane(dragPlane, point)) {
        const entry = parts.get(dragging);
        if (entry) {
            entry.mesh.position.copy(point).add(dragOffset);
            draggedSinceDown = true;
            if (dragging === selectedId) syncPivot(); // the combined control rides along
        }
    }
}

function onPointerUp(event) {
    if (viewCube?.handlePointerUp(event)) return;
    const moved = dragging !== null && draggedSinceDown;
    dragging = null;
    controls.enabled = true;
    if (moved) afterPartMoved();
}

function afterPartMoved() {
    if (!autoSnap || selectedId === null) return;
    // Deferred so the pointer gesture fully settles before the (possibly ~1s) fit runs.
    setTimeout(() => { snapSelected(true).then(() => notifyChanged()); }, 30);
}

export function setAutoSnap(on) { autoSnap = !!on; }
export function getAutoSnap() { return autoSnap; }

// ---------------------------------------------------------------------------------------------
// Snap: complementary-surface point-to-plane ICP over surface samples + a spatial hash.
// ---------------------------------------------------------------------------------------------

const MOVING_SAMPLES = 2600;
const TARGET_SAMPLES = 42000;

/// Area-weighted surface sampling in LOCAL space; face normals ride along.
function buildSamples(entry, count) {
    const pos = entry.mesh.geometry.attributes.position;
    const triCount = pos.count / 3;
    const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3();
    const ab = new THREE.Vector3(), ac = new THREE.Vector3(), n = new THREE.Vector3();

    let totalArea = 0;
    const areas = new Float64Array(triCount);
    for (let t = 0; t < triCount; t++) {
        a.fromBufferAttribute(pos, t * 3);
        b.fromBufferAttribute(pos, t * 3 + 1);
        c.fromBufferAttribute(pos, t * 3 + 2);
        const area = ab.subVectors(b, a).cross(ac.subVectors(c, a)).length() / 2;
        areas[t] = area;
        totalArea += area;
    }

    const points = new Float32Array(count * 3);
    const normals = new Float32Array(count * 3);
    let emitted = 0, carry = 0;
    for (let t = 0; t < triCount && emitted < count; t++) {
        carry += count * areas[t] / totalArea;
        let emit = Math.floor(carry);
        carry -= emit;
        if (emit === 0) continue;
        a.fromBufferAttribute(pos, t * 3);
        b.fromBufferAttribute(pos, t * 3 + 1);
        c.fromBufferAttribute(pos, t * 3 + 2);
        n.copy(ab.subVectors(b, a)).cross(ac.subVectors(c, a)).normalize();
        while (emit-- > 0 && emitted < count) {
            const r1 = Math.sqrt(Math.random());
            const r2 = Math.random();
            const u = 1 - r1, v = r1 * (1 - r2), w = r1 * r2;
            points[emitted * 3] = a.x * u + b.x * v + c.x * w;
            points[emitted * 3 + 1] = a.y * u + b.y * v + c.y * w;
            points[emitted * 3 + 2] = a.z * u + b.z * v + c.z * w;
            normals[emitted * 3] = n.x;
            normals[emitted * 3 + 1] = n.y;
            normals[emitted * 3 + 2] = n.z;
            emitted++;
        }
    }
    return { points, normals, count: emitted, area: totalArea };
}

function getSamples(entry, count) {
    if (!entry.samples || entry.samples.count < count * 0.9) entry.samples = buildSamples(entry, count);
    return entry.samples;
}

/// Uniform spatial hash over a sample set (target-local space).
function buildHash(samples, cellSize) {
    const map = new Map();
    for (let i = 0; i < samples.count; i++) {
        const key = `${Math.floor(samples.points[i * 3] / cellSize)},${Math.floor(samples.points[i * 3 + 1] / cellSize)},${Math.floor(samples.points[i * 3 + 2] / cellSize)}`;
        let bucket = map.get(key);
        if (!bucket) { bucket = []; map.set(key, bucket); }
        bucket.push(i);
    }
    return {
        cellSize, map, samples,
        nearest(x, y, z, maxDist) {
            const cx = Math.floor(x / cellSize), cy = Math.floor(y / cellSize), cz = Math.floor(z / cellSize);
            const reach = Math.max(1, Math.ceil(maxDist / cellSize));
            let best = -1, bestSq = maxDist * maxDist;
            for (let ix = cx - reach; ix <= cx + reach; ix++)
                for (let iy = cy - reach; iy <= cy + reach; iy++)
                    for (let iz = cz - reach; iz <= cz + reach; iz++) {
                        const bucket = this.map.get(`${ix},${iy},${iz}`);
                        if (!bucket) continue;
                        for (const i of bucket) {
                            const dx = this.samples.points[i * 3] - x;
                            const dy = this.samples.points[i * 3 + 1] - y;
                            const dz = this.samples.points[i * 3 + 2] - z;
                            const d = dx * dx + dy * dy + dz * dz;
                            if (d < bestSq) { bestSq = d; best = i; }
                        }
                    }
            return best < 0 ? null : { index: best, dist: Math.sqrt(bestSq) };
        }
    };
}

/// Solves A x = b for a symmetric 6x6 system (Gaussian elimination, partial pivot).
function solve6(A, b) {
    const n = 6;
    const M = A.map((row, i) => [...row, b[i]]);
    for (let col = 0; col < n; col++) {
        let pivot = col;
        for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[pivot][col])) pivot = r;
        if (Math.abs(M[pivot][col]) < 1e-12) return null;
        [M[col], M[pivot]] = [M[pivot], M[col]];
        for (let r = col + 1; r < n; r++) {
            const f = M[r][col] / M[col][col];
            for (let k = col; k <= n; k++) M[r][k] -= f * M[col][k];
        }
    }
    const x = new Array(n).fill(0);
    for (let r = n - 1; r >= 0; r--) {
        let s = M[r][n];
        for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k];
        x[r] = s / M[r][r];
    }
    return x;
}

/// Fits the SELECTED part against the best nearby part. Returns a report; on failure the part
/// is left exactly where it was. auto=true uses a tighter capture radius so releasing a part
/// far from anything never yanks it across the scene.
export async function snapSelected(auto = false) {
    const moving = selectedId !== null ? parts.get(selectedId) : null;
    if (!moving) return { snapped: false, reason: 'select a part first' };
    const others = [...parts.values()].filter(e => e.id !== moving.id && e.mesh.visible);
    if (others.length === 0) return { snapped: false, reason: 'nothing to snap to' };

    status('computing fit…');
    await new Promise(r => setTimeout(r, 15)); // let the status paint before the math blocks

    moving.mesh.updateMatrixWorld();
    const mRadius = moving.mesh.geometry.boundingSphere.radius * moving.scale;
    // Manual snap searches wider (the user ASKED for a fit); auto stays tight so releasing a
    // part somewhere deliberate never yanks it onto a neighbour.
    const captureR = (auto ? 0.08 : 0.18) * mRadius;

    // A part that has served as a snap TARGET carries dense samples; when it later MOVES,
    // probe a strided subset so the fit cost stays bounded regardless of history.
    const mSamples = getSamples(moving, MOVING_SAMPLES);
    const mStride = Math.max(1, Math.floor(mSamples.count / MOVING_SAMPLES));

    // Pick the target with the most probe points inside the capture radius.
    let target = null, targetHash = null, bestContacts = 0;
    const probe = new THREE.Vector3();
    for (const candidate of others) {
        candidate.mesh.updateMatrixWorld();
        const toLocal = candidate.mesh.matrixWorld.clone().invert();
        const toTarget = toLocal.multiply(moving.mesh.matrixWorld);
        const tSamples = getSamples(candidate, TARGET_SAMPLES);
        const spacing = Math.sqrt(tSamples.area / tSamples.count);
        const hash = buildHash(tSamples, Math.max(captureR / candidate.scale, spacing * 2));
        let contacts = 0;
        for (let i = 0; i < mSamples.count; i += mStride * 4) { // quarter of the probes ranks fine
            probe.set(mSamples.points[i * 3], mSamples.points[i * 3 + 1], mSamples.points[i * 3 + 2])
                .applyMatrix4(toTarget);
            if (hash.nearest(probe.x, probe.y, probe.z, captureR / candidate.scale)) contacts++;
        }
        if (contacts > bestContacts) { bestContacts = contacts; target = candidate; targetHash = hash; }
    }
    if (!target || bestContacts < 12) {
        status('');
        return { snapped: false, reason: 'no nearby surface — move the part closer first' };
    }

    // ICP in TARGET-LOCAL space. The moving part's pose is refined as localDelta * original.
    const original = moving.mesh.matrix.clone();
    const tSamples = target.samples;
    const spacing = Math.sqrt(tSamples.area / tSamples.count);
    const targetToWorld = target.mesh.matrixWorld.clone();
    const worldToTarget = targetToWorld.clone().invert();

    let current = worldToTarget.clone().multiply(moving.mesh.matrixWorld); // moving-local -> target-local
    const normalMat = new THREE.Matrix3();
    const p = new THREE.Vector3(), n = new THREE.Vector3(), q = new THREE.Vector3(), m = new THREE.Vector3();
    const centroid = new THREE.Vector3();
    let contactCount = 0;

    const iterations = 40;
    for (let iter = 0; iter < iterations; iter++) {
        const radius = (captureR / target.scale) * (1 - 0.8 * iter / (iterations - 1)); // anneal inward
        normalMat.getNormalMatrix(current);

        const pairs = [];
        centroid.set(0, 0, 0);
        for (let i = 0; i < mSamples.count; i += mStride) {
            p.set(mSamples.points[i * 3], mSamples.points[i * 3 + 1], mSamples.points[i * 3 + 2]).applyMatrix4(current);
            const hit = targetHash.nearest(p.x, p.y, p.z, radius);
            if (!hit) continue;
            n.set(mSamples.normals[i * 3], mSamples.normals[i * 3 + 1], mSamples.normals[i * 3 + 2])
                .applyMatrix3(normalMat).normalize();
            m.set(tSamples.normals[hit.index * 3], tSamples.normals[hit.index * 3 + 1], tSamples.normals[hit.index * 3 + 2]);
            const opposition = -n.dot(m);
            if (opposition < -0.2) continue;             // same-facing surfaces don't mate
            q.set(tSamples.points[hit.index * 3], tSamples.points[hit.index * 3 + 1], tSamples.points[hit.index * 3 + 2]);
            let weight = 0.15 + Math.max(0, opposition);
            const side = (p.x - q.x) * m.x + (p.y - q.y) * m.y + (p.z - q.z) * m.z;
            if (side < 0) weight *= 5;                   // interpenetration: push out hard
            pairs.push({ px: p.x, py: p.y, pz: p.z, qx: q.x, qy: q.y, qz: q.z, mx: m.x, my: m.y, mz: m.z, weight, dist: hit.dist });
            centroid.add(p);
        }
        contactCount = pairs.length;
        if (contactCount < 12) break;
        centroid.divideScalar(contactCount);

        const A = Array.from({ length: 6 }, () => new Array(6).fill(0));
        const bVec = new Array(6).fill(0);
        for (const pair of pairs) {
            const px = pair.px - centroid.x, py = pair.py - centroid.y, pz = pair.pz - centroid.z;
            const ax = py * pair.mz - pz * pair.my;      // cross(p - c, m)
            const ay = pz * pair.mx - px * pair.mz;
            const az = px * pair.my - py * pair.mx;
            const row = [ax, ay, az, pair.mx, pair.my, pair.mz];
            const residual = (pair.px - pair.qx) * pair.mx + (pair.py - pair.qy) * pair.my + (pair.pz - pair.qz) * pair.mz;
            for (let i = 0; i < 6; i++) {
                bVec[i] -= pair.weight * row[i] * residual;
                for (let j = i; j < 6; j++) A[i][j] += pair.weight * row[i] * row[j];
            }
        }
        for (let i = 0; i < 6; i++) for (let j = 0; j < i; j++) A[i][j] = A[j][i];
        for (let i = 0; i < 6; i++) A[i][i] += 1e-9;

        const x = solve6(A, bVec);
        if (!x) break;
        const omega = new THREE.Vector3(x[0], x[1], x[2]).multiplyScalar(0.8); // damped step
        const trans = new THREE.Vector3(x[3], x[4], x[5]).multiplyScalar(0.8);
        const angle = Math.min(omega.length(), THREE.MathUtils.degToRad(8));
        const delta = new THREE.Matrix4()
            .makeTranslation(centroid.x + trans.x, centroid.y + trans.y, centroid.z + trans.z)
            .multiply(new THREE.Matrix4().makeRotationAxis(
                omega.length() > 1e-9 ? omega.clone().normalize() : new THREE.Vector3(1, 0, 0), angle))
            .multiply(new THREE.Matrix4().makeTranslation(-centroid.x, -centroid.y, -centroid.z));
        current = delta.multiply(current);

        if (angle < THREE.MathUtils.degToRad(0.02) && trans.length() < mRadius * 1e-4 && iter > 10) break;
    }

    // Acceptance is judged on a dedicated final sweep over the true CONTACT set (pairs within
    // a couple of sample spacings) — the annealed ICP radius still admits off-contact pairs
    // that would unfairly inflate the score.
    const tolerance = Math.max(1.8 * spacing, 0.004 * mRadius / target.scale);
    let meanDist = Infinity;
    {
        const contactRadius = Math.max(2.5 * spacing, tolerance * 1.5);
        let sum = 0, count = 0;
        for (let i = 0; i < mSamples.count; i += mStride) {
            p.set(mSamples.points[i * 3], mSamples.points[i * 3 + 1], mSamples.points[i * 3 + 2]).applyMatrix4(current);
            const hit = targetHash.nearest(p.x, p.y, p.z, contactRadius);
            if (!hit) continue;
            sum += hit.dist;
            count++;
        }
        contactCount = count;
        if (count > 0) meanDist = sum / count;
    }
    const proposedWorld = targetToWorld.clone().multiply(current);
    const proposedLocal = moving.mesh.parent
        ? proposedWorld // parts sit directly in the scene, so world == local
        : proposedWorld;
    const originalPos = new THREE.Vector3().setFromMatrixPosition(original);
    const proposedPos = new THREE.Vector3().setFromMatrixPosition(proposedLocal);
    const drift = originalPos.distanceTo(proposedPos);
    const qOriginal = new THREE.Quaternion().setFromRotationMatrix(original);
    const qProposed = new THREE.Quaternion().setFromRotationMatrix(proposedLocal);
    const rotDrift = qOriginal.angleTo(qProposed);

    status('');
    if (contactCount < 60 || meanDist > tolerance) {
        devlog('snap-rejected', { contacts: contactCount, gap: +meanDist.toFixed(3), tolerance: +tolerance.toFixed(3) });
        return {
            snapped: false,
            reason: `no confident fit (contact ${contactCount}, gap ${meanDist.toFixed(2)} > ${tolerance.toFixed(2)})`
        };
    }
    if (drift > 0.45 * mRadius || rotDrift > THREE.MathUtils.degToRad(40)) {
        devlog('snap-rejected', { reason: 'wandered', drift: +drift.toFixed(3) });
        return { snapped: false, reason: 'fit wandered too far — place the part closer to where it belongs' };
    }

    devlog('snap', { part: moving.name, target: target.name, gap: +meanDist.toFixed(3), contacts: contactCount });
    captureUndo('snap', false);
    moving.mesh.matrix.copy(proposedLocal);
    moving.mesh.matrix.decompose(moving.mesh.position, moving.mesh.quaternion, moving.mesh.scale);
    if (moving.id === selectedId) syncPivot();
    return { snapped: true, gap: meanDist, contacts: contactCount };
}

// ---------------------------------------------------------------------------------------------
// Scissor cut: click points on the selected part; the service routes each leg along the
// SHARPEST creases between clicks (MeshScissor A*), the canvas draws the accumulated cut line,
// and once the path fully separates the surface the cut executes server-side. All geometry
// stays on the service; the canvas only places anchors and draws polylines.
// ---------------------------------------------------------------------------------------------

let scissors = null; // { sessionId, urlBase, headerName, headerValue, partId, overlay, markers, busy }

async function scissorPost(path, body) {
    const response = await fetch(`${scissors.urlBase}/${path}`, {
        method: 'POST',
        headers: { [scissors.headerName]: scissors.headerValue, 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    });
    if (!response.ok) {
        let reason = `${response.status}`;
        try { reason = (await response.json()).detail ?? reason; } catch { }
        throw new Error(reason);
    }
    return await response.json();
}

function scissorMarkerSize(entry) {
    return entry.mesh.geometry.boundingSphere.radius * 0.012;
}

const scissorChain = () => scissors.chains[scissors.chains.length - 1];
const scissorAnchorCount = () => scissors.chains.reduce((sum, c) => sum + c.markers.length, 0);

function scissorAddMarker(entry, localPoint, first) {
    const marker = new THREE.Mesh(
        new THREE.SphereGeometry(scissorMarkerSize(entry), 12, 12),
        new THREE.MeshBasicMaterial({ color: first ? 0xffe640 : 0xff4040, depthTest: false }));
    marker.position.copy(localPoint);
    marker.renderOrder = 30;
    scissors.overlay.add(marker);
    scissorChain().markers.push(marker);
}

function scissorAddPolyline(points) {
    const geometry = new THREE.BufferGeometry().setFromPoints(
        points.map(p => new THREE.Vector3(p[0], p[1], p[2])));
    const line = new THREE.Line(geometry,
        new THREE.LineBasicMaterial({ color: 0xff3030, depthTest: false }));
    line.renderOrder = 29;
    scissors.overlay.add(line);
    scissorChain().lines.push(line);
}

/// The next click starts a SEPARATE cut line — some separations need several (a handle wants a
/// loop at each end; a staff held in two hands wants a cut at each hand).
export function scissorNewLine() {
    if (!scissors) return;
    scissors.pendingNewChain = true;
    status('scissors: next click starts a NEW cut line');
}

/// Enters scissor mode on the SELECTED part: uploads its local-space STL once, then clicks
/// place anchors instead of moving parts. Returns a JSON status for the shell.
export async function enterScissors(urlBase, headerName, headerValue) {
    const entry = selectedId !== null ? parts.get(selectedId) : null;
    if (!entry) return JSON.stringify({ ok: false, reason: 'select a part first' });
    if (scissors) exitScissors();
    status('preparing scissors (uploading part + building the crease graph)…');
    try {
        const response = await fetch(`${urlBase}/begin`, {
            method: 'POST',
            headers: { [headerName]: headerValue, 'Content-Type': 'application/octet-stream' },
            body: exportPartLocalStl(entry)
        });
        if (!response.ok) throw new Error(`${response.status}`);
        const result = await response.json();

        const overlay = new THREE.Group();
        entry.mesh.add(overlay); // local coords ride the part's transform
        scissors = {
            sessionId: result.sessionId, urlBase, headerName, headerValue,
            partId: entry.id, overlay, chains: [], pendingNewChain: false, busy: false
        };
        moveGizmo.detach();
        rotateGizmo.detach();
        status('scissors: click points along the crease to cut on');
        return JSON.stringify({ ok: true });
    }
    catch (error) {
        status('');
        return JSON.stringify({ ok: false, reason: error.message });
    }
}

export function exitScissors() {
    if (!scissors) return;
    const entry = parts.get(scissors.partId);
    entry?.mesh.remove(scissors.overlay);
    scissors.overlay.traverse(o => { o.geometry?.dispose?.(); o.material?.dispose?.(); });
    fetch(`${scissors.urlBase}/${scissors.sessionId}`, {
        method: 'DELETE', headers: { [scissors.headerName]: scissors.headerValue }
    }).catch(() => { });
    scissors = null;
    status('');
}

export function isScissorsActive() { return scissors !== null; }

/// Canvas click while in scissor mode: raycast the part, snap server-side, draw the new leg.
async function scissorClick(hitPoint) {
    if (scissors.busy) return;
    const entry = parts.get(scissors.partId);
    if (!entry) return;
    scissors.busy = true;
    try {
        const local = entry.mesh.worldToLocal(hitPoint.clone());
        const result = await scissorPost('anchor',
            { sessionId: scissors.sessionId, x: local.x, y: local.y, z: local.z,
              newChain: scissors.pendingNewChain });
        scissors.pendingNewChain = false;
        devlog('scissor-anchor', {
            x: +local.x.toFixed(3), y: +local.y.toFixed(3), z: +local.z.toFixed(3),
            anchors: result.anchors, chains: result.chains,
            separates: result.separates, components: result.components
        });
        if (result.startsChain || scissors.chains.length === 0)
            scissors.chains.push({ markers: [], lines: [] });
        scissorAddMarker(entry, new THREE.Vector3(...result.anchor), scissorChain().markers.length === 0);
        if (result.polyline) scissorAddPolyline(result.polyline);
        scissors.separates = result.separates;
        const lineNote = result.chains > 1 ? ` (${result.chains} lines)` : '';
        status(`scissors: ${result.anchors} point(s)${lineNote} — ${result.separates
            ? 'the cut SEPARATES the part ✓ ready to cut'
            : 'not separating yet (close the loop, keep going, or add a New line)'}`);
        notifyChanged(); // the shell re-reads scissor state for its buttons
    }
    catch (error) { status(`scissors: ${error.message}`); }
    finally { scissors.busy = false; }
}

/// Routes from the last anchor back to the first, completing a loop.
export async function scissorClose() {
    if (!scissors || scissors.busy) return JSON.stringify({ ok: false, reason: 'not in scissor mode' });
    scissors.busy = true;
    try {
        const result = await scissorPost('close', { sessionId: scissors.sessionId });
        devlog('scissor-close', { separates: result.separates, components: result.components });
        if (result.polyline) scissorAddPolyline(result.polyline);
        scissors.separates = result.separates;
        scissors.pendingNewChain = true; // a closed loop is complete — clicking again starts a new line
        status(`scissors: loop closed — ${result.separates
            ? 'SEPARATES ✓ ready to cut'
            : 'still not separating (another line may be needed — e.g. a handle wants a loop at BOTH ends)'}`);
        return JSON.stringify({ ok: true, separates: result.separates });
    }
    catch (error) {
        status(`scissors: ${error.message}`);
        return JSON.stringify({ ok: false, reason: error.message });
    }
    finally { scissors.busy = false; }
}

export async function scissorUndo() {
    if (!scissors || scissors.busy) return JSON.stringify({ ok: false });
    scissors.busy = true;
    try {
        const result = await scissorPost('undo', { sessionId: scissors.sessionId });
        const chain = scissorChain();
        if (chain) {
            const marker = chain.markers.pop();
            if (marker) scissors.overlay.remove(marker);
            if (chain.lines.length > Math.max(0, chain.markers.length - 1)) {
                const line = chain.lines.pop();
                if (line) scissors.overlay.remove(line);
            }
            if (chain.markers.length === 0) scissors.chains.pop();
        }
        scissors.pendingNewChain = false; // resume the line the undo landed on
        scissors.separates = result.separates;
        status(`scissors: ${result.anchors} point(s)${result.chains > 1 ? ` (${result.chains} lines)` : ''}${result.separates ? ' — SEPARATES ✓' : ''}`);
        return JSON.stringify({ ok: true, separates: result.separates });
    }
    catch (error) { return JSON.stringify({ ok: false, reason: error.message }); }
    finally { scissors.busy = false; }
}

export function scissorState() {
    return JSON.stringify({
        active: scissors !== null,
        anchors: scissors ? scissorAnchorCount() : 0,
        chains: scissors?.chains.length ?? 0,
        separates: scissors?.separates ?? false
    });
}

/// Executes the cut: the part is replaced by the separated pieces at the same transform.
export async function scissorCut(cap) {
    if (!scissors || scissors.busy) return JSON.stringify({ ok: false, reason: 'not in scissor mode' });
    scissors.busy = true;
    status('cutting…');
    try {
        const result = await scissorPost('cut', { sessionId: scissors.sessionId, cap: !!cap });
        const entry = parts.get(scissors.partId);
        if (!entry) throw new Error('the part disappeared mid-cut');
        entry.mesh.updateMatrix();
        const matrix = [...entry.mesh.matrix.toArray()];
        const baseName = entry.name;

        // The session is consumed server-side; tear down the overlay without the DELETE call.
        entry.mesh.remove(scissors.overlay);
        scissors.overlay.traverse(o => { o.geometry?.dispose?.(); o.material?.dispose?.(); });
        scissors = null;

        captureUndo('scissor-cut', false);
        undoSuppressDepth++;
        const added = [];
        try {
            detachPart(entry.id);
            for (let i = 0; i < result.parts.length; i++) {
                const bytes = Uint8Array.from(atob(result.parts[i].stl), ch => ch.charCodeAt(0));
                const id = addPartFromBytes(bytes, `${baseName} · ${String.fromCharCode(65 + (i % 26))}`);
                setPartMatrix(id, matrix);
                added.push(id);
            }
        }
        finally { undoSuppressDepth--; }
        if (added.length > 0) selectPart(added[0]);
        devlog('scissor-cut', { pieces: added.length, from: baseName });
        status(`cut into ${added.length} pieces ✓ (Ctrl+Z undoes the whole cut)`);
        return JSON.stringify({ ok: true, count: added.length });
    }
    catch (error) {
        status(`scissors: ${error.message}`);
        return JSON.stringify({ ok: false, reason: error.message });
    }
    finally { if (scissors) scissors.busy = false; }
}

// ---------------------------------------------------------------------------------------------
// Seam detection + split (server-side geometry; the canvas only previews and re-adds parts).
// The selected part's LOCAL-space STL goes to the service, which finds the crease loops where
// separate pieces meet. Preview-first: detect colours the part per candidate piece; split
// replaces the part with the returned pieces at the exact same transform.
// ---------------------------------------------------------------------------------------------

const SEAM_COLORS = ['#ff4040', '#40d440', '#4080ff', '#ffe640', '#ff40ff', '#40e8e8',
                     '#ff8c1a', '#a060ff', '#f0f0f0', '#8a5a30'];

/// One part's geometry as binary STL. Without a matrix: LOCAL space (so split pieces can be
/// re-added under the part's original matrix and land exactly in place). With one: baked —
/// the file matches what's on screen, which is what a per-part download wants.
function exportPartStl(entry, matrix = null) {
    const pos = entry.mesh.geometry.attributes.position;
    const triCount = pos.count / 3;
    const buffer = new ArrayBuffer(84 + triCount * 50);
    const out = new Uint8Array(buffer);
    out.set(new TextEncoder().encode('Chromatory part'), 0);
    new DataView(buffer).setUint32(80, triCount, true);
    const scratch = new Float32Array(12);
    const scratchBytes = new Uint8Array(scratch.buffer, 0, 48);
    const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3();
    const ab = new THREE.Vector3(), ac = new THREE.Vector3(), n = new THREE.Vector3();
    let offset = 84;
    for (let t = 0; t < triCount; t++) {
        a.fromBufferAttribute(pos, t * 3);
        b.fromBufferAttribute(pos, t * 3 + 1);
        c.fromBufferAttribute(pos, t * 3 + 2);
        if (matrix) { a.applyMatrix4(matrix); b.applyMatrix4(matrix); c.applyMatrix4(matrix); }
        n.copy(ab.subVectors(b, a)).cross(ac.subVectors(c, a)).normalize();
        scratch[0] = n.x; scratch[1] = n.y; scratch[2] = n.z;
        scratch[3] = a.x; scratch[4] = a.y; scratch[5] = a.z;
        scratch[6] = b.x; scratch[7] = b.y; scratch[8] = b.z;
        scratch[9] = c.x; scratch[10] = c.y; scratch[11] = c.z;
        out.set(scratchBytes, offset);
        offset += 50;
    }
    return out;
}

const exportPartLocalStl = entry => exportPartStl(entry);

/// Downloads ONE part as a binary STL with its current transform baked — scale, rotation and
/// position as seen on screen — so cut pieces go straight to the slicer.
export function downloadPart(id) {
    const entry = parts.get(id);
    if (!entry) return;
    entry.mesh.updateMatrixWorld();
    const bytes = exportPartStl(entry, entry.mesh.matrixWorld);
    const blob = new Blob([bytes], { type: 'model/stl' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = entry.name.replace(/[^\w\-. ·]+/g, '_') + '.stl';
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 10000);
}

async function postPartStl(entry, url, headerName, headerValue) {
    const response = await fetch(url, {
        method: 'POST',
        headers: { [headerName]: headerValue, 'Content-Type': 'application/octet-stream' },
        body: exportPartLocalStl(entry)
    });
    if (!response.ok) {
        let reason = `${response.status}`;
        try { reason = (await response.json()).detail ?? reason; } catch { }
        throw new Error(reason);
    }
    return await response.json();
}

/// Detects seams in the selected part and colours it per candidate piece (preview only —
/// geometry and parts are untouched). Returns the detection summary for the shell's UI.
export async function detectSeams(url, headerName, headerValue) {
    const entry = selectedId !== null ? parts.get(selectedId) : null;
    if (!entry) return JSON.stringify({ ok: false, reason: 'select a part first' });
    status('detecting seams…');
    try {
        const result = await postPartStl(entry, url, headerName, headerValue);
        const labels = Uint8Array.from(atob(result.labels), ch => ch.charCodeAt(0));

        const pos = entry.mesh.geometry.attributes.position;
        const colors = new Float32Array(pos.count * 3);
        const c = new THREE.Color();
        for (let t = 0; t < labels.length; t++) {
            c.set(SEAM_COLORS[(labels[t] - 1 + SEAM_COLORS.length) % SEAM_COLORS.length]);
            for (let k = 0; k < 3; k++) {
                colors[(t * 3 + k) * 3] = c.r;
                colors[(t * 3 + k) * 3 + 1] = c.g;
                colors[(t * 3 + k) * 3 + 2] = c.b;
            }
        }
        entry.mesh.geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        entry.seamPreview = true;
        entry.mesh.material.vertexColors = true;
        entry.mesh.material.color.set('#ffffff');
        entry.mesh.material.needsUpdate = true;

        return JSON.stringify({
            ok: true, seamEdges: result.seamEdges,
            parts: result.parts.map((p, i) => ({
                ...p, color: SEAM_COLORS[(p.label - 1) % SEAM_COLORS.length]
            }))
        });
    }
    catch (error) {
        return JSON.stringify({ ok: false, reason: error.message });
    }
    finally { status(''); }
}

/// Drops the seam preview colouring from every part (called on cancel and after a split).
export function clearSeamPreview() {
    for (const entry of parts.values()) {
        if (!entry.seamPreview) continue;
        entry.seamPreview = false;
        entry.mesh.geometry.deleteAttribute('color');
        entry.mesh.material.vertexColors = false;
        entry.mesh.material.color.set(entry.color);
        entry.mesh.material.needsUpdate = true;
    }
}

/// Splits the selected part along its detected seams: the part is replaced by the returned
/// pieces, each under the ORIGINAL part's transform so nothing moves on screen.
export async function splitSelected(url, headerName, headerValue) {
    const entry = selectedId !== null ? parts.get(selectedId) : null;
    if (!entry) return JSON.stringify({ ok: false, reason: 'select a part first' });
    status('splitting along seams…');
    try {
        const result = await postPartStl(entry, url, headerName, headerValue);
        entry.mesh.updateMatrix();
        const matrix = [...entry.mesh.matrix.toArray()];
        const baseName = entry.name;

        clearSeamPreview();
        // One undo entry for the whole operation: undoing a split brings the original back
        // and removes the pieces in a single Ctrl+Z.
        captureUndo('split', false);
        undoSuppressDepth++;
        const added = [];
        try {
            detachPart(entry.id);
            for (let i = 0; i < result.parts.length; i++) {
                const bytes = Uint8Array.from(atob(result.parts[i].stl), ch => ch.charCodeAt(0));
                const id = addPartFromBytes(bytes, `${baseName} · ${String.fromCharCode(65 + (i % 26))}`);
                setPartMatrix(id, matrix);
                added.push(id);
            }
        }
        finally { undoSuppressDepth--; }
        if (added.length > 0) selectPart(added[0]);
        return JSON.stringify({ ok: true, count: added.length });
    }
    catch (error) {
        return JSON.stringify({ ok: false, reason: error.message });
    }
    finally { status(''); }
}

// ---------------------------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------------------------

/// Bridge mode: instead of nudging the mating surfaces (which distorts them), leave both parts
/// exactly as sculpted and ADD new geometry that fills the enclosed gap. For every triangle that
/// faces another part across the gap, a small closed prism is emitted spanning from just inside
/// this part to just inside the other (its 3 corners projected onto the other surface). The union
/// of those prisms is a solid collar filling the seam — the slicer prints one piece. Each gap is
/// bridged once (only from the lower-indexed part) so the fill isn't built twice. Returns a flat
/// Float32Array of world-space triangle corners (9 floats per triangle).
function buildBridgeTriangles(visible, tol) {
    const overlap = tol * 0.5;
    const hashes = visible.map(entry => {
        const ws = worldSamples(entry, STITCH_SAMPLES);
        const spacing = Math.sqrt(ws.area / ws.count);
        return buildHash(ws, Math.max(tol, spacing * 2));
    });
    const boxes = visible.map(entry => { entry.mesh.updateMatrixWorld(); return new THREE.Box3().setFromObject(entry.mesh); });
    const otherBox = visible.map((_, pi) => {
        const box = new THREE.Box3();
        for (let j = 0; j < visible.length; j++) if (j !== pi) box.union(boxes[j]);
        return box.expandByScalar(tol);
    });

    const out = [];
    const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3();
    const ab = new THREE.Vector3(), ac = new THREE.Vector3(), fn = new THREE.Vector3(), g = new THREE.Vector3(), dir = new THREE.Vector3();
    const top = [new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3()];
    const bot = [new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3()];
    const corners = [a, b, c];

    function nearestOn(j, p, maxDist) {
        const hit = hashes[j].nearest(p.x, p.y, p.z, maxDist);
        if (!hit) return null;
        const s = hashes[j].samples, i = hit.index;
        return { x: s.points[i * 3], y: s.points[i * 3 + 1], z: s.points[i * 3 + 2],
                 nx: s.normals[i * 3], ny: s.normals[i * 3 + 1], nz: s.normals[i * 3 + 2], dist: hit.dist };
    }
    const emit = (v0, v1, v2) => out.push(v0.x, v0.y, v0.z, v1.x, v1.y, v1.z, v2.x, v2.y, v2.z);

    for (let pi = 0; pi < visible.length; pi++) {
        const entry = visible[pi];
        entry.mesh.updateMatrixWorld();
        const M = entry.mesh.matrixWorld;
        const pos = entry.mesh.geometry.attributes.position;
        for (let t = 0; t < entry.triangles; t++) {
            a.fromBufferAttribute(pos, t * 3).applyMatrix4(M);
            b.fromBufferAttribute(pos, t * 3 + 1).applyMatrix4(M);
            c.fromBufferAttribute(pos, t * 3 + 2).applyMatrix4(M);
            g.set((a.x + b.x + c.x) / 3, (a.y + b.y + c.y) / 3, (a.z + b.z + c.z) / 3);
            if (!otherBox[pi].containsPoint(g)) continue;

            // The nearest OTHER part whose surface faces this triangle across a gap ≤ tol.
            let jBest = -1, centre = null, bestDist = tol;
            for (let j = 0; j < hashes.length; j++) {
                if (j === pi) continue;
                const s = nearestOn(j, g, tol);
                if (!s || s.dist >= bestDist) continue;
                dir.set(g.x - s.x, g.y - s.y, g.z - s.z);
                const len = dir.length() || 1e-9;
                if ((s.nx * dir.x + s.ny * dir.y + s.nz * dir.z) / len > 0.25) { jBest = j; centre = s; bestDist = s.dist; }
            }
            if (jBest < pi) continue; // each gap bridged once, from the lower-indexed part

            fn.copy(ab.subVectors(b, a)).cross(ac.subVectors(c, a)).normalize(); // outward face normal
            for (let k = 0; k < 3; k++) {
                top[k].copy(corners[k]).addScaledVector(fn, -overlap); // embed slightly into THIS part
                const s = nearestOn(jBest, corners[k], tol * 2) || centre; // project onto the other part
                bot[k].set(s.x - s.nx * overlap, s.y - s.ny * overlap, s.z - s.nz * overlap); // embed into it
            }
            emit(top[0], top[1], top[2]);            // caps
            emit(bot[2], bot[1], bot[0]);
            for (let k = 0; k < 3; k++) {            // side walls
                const n = (k + 1) % 3;
                emit(top[k], top[n], bot[n]);
                emit(top[k], bot[n], bot[k]);
            }
        }
    }
    return out;
}

/// Exact byte size the merged export will have (84-byte header + 50 per triangle). Nudge-stitching
/// only MOVES seam vertices, so the size is unchanged; bridge mode ADDS triangles, so this is a
/// lower bound there (the bridge collar is small next to the parts).
export function exportStlSize() {
    let totalTris = 0;
    for (const entry of parts.values()) if (entry.mesh.visible) totalTris += entry.triangles;
    return 84 + totalTris * 50;
}

const STITCH_SAMPLES = 90000; // denser than the ICP target: the gap fit wants finer surface detail

/// World-space surface samples (points + normals) for a part, reusing the ICP sampler.
function worldSamples(entry, count) {
    entry.mesh.updateMatrixWorld();
    const local = getSamples(entry, count);
    const m = entry.mesh.matrixWorld;
    const nm = new THREE.Matrix3().getNormalMatrix(m);
    const points = new Float32Array(local.count * 3);
    const normals = new Float32Array(local.count * 3);
    const v = new THREE.Vector3(), n = new THREE.Vector3();
    for (let i = 0; i < local.count; i++) {
        v.set(local.points[i * 3], local.points[i * 3 + 1], local.points[i * 3 + 2]).applyMatrix4(m);
        n.set(local.normals[i * 3], local.normals[i * 3 + 1], local.normals[i * 3 + 2]).applyMatrix3(nm).normalize();
        points[i * 3] = v.x; points[i * 3 + 1] = v.y; points[i * 3 + 2] = v.z;
        normals[i * 3] = n.x; normals[i * 3 + 1] = n.y; normals[i * 3 + 2] = n.z;
    }
    return { points, normals, count: local.count, area: local.area };
}

/// Gap-stitcher: closes the intentional clearance between mating parts so the merged STL prints as
/// one solid instead of two pieces with a slicer-visible seam. A vertex is moved ONLY when it sits
/// within `tol` of another part's surface AND that surface FACES it (opposing normals) — i.e. it's
/// across a real slot-together gap, not just near a neighbour. Such a vertex is pushed onto the far
/// surface and a hair past it (overlap), so the slicer's per-layer union fuses the two. The
/// displacement is a pure function of the world position (cached per point), so every triangle
/// sharing that vertex moves identically — the mesh stays watertight, no cracks. Interior/outer
/// vertices are far from any other part, so the model's real dimensions and detail are untouched.
function buildStitcher(visible, tol) {
    const overlap = tol * 0.5;
    const hashes = visible.map(entry => {
        const ws = worldSamples(entry, STITCH_SAMPLES);
        const spacing = Math.sqrt(ws.area / ws.count);
        return buildHash(ws, Math.max(tol, spacing * 2));
    });
    // Per part, the bounding box of every OTHER part (expanded by tol): a vertex outside it can't be
    // near a seam, so it skips the (relatively costly) nearest-surface query entirely.
    const boxes = visible.map(entry => {
        entry.mesh.updateMatrixWorld();
        return new THREE.Box3().setFromObject(entry.mesh);
    });
    const otherBox = visible.map((_, pi) => {
        const box = new THREE.Box3();
        for (let j = 0; j < visible.length; j++) if (j !== pi) box.union(boxes[j]);
        return box.expandByScalar(tol);
    });

    const cache = new Map();
    const QUANT = 1e-4;
    const dir = new THREE.Vector3();

    function compute(p, pi) {
        let best = null;
        let bestDist = tol;
        for (let j = 0; j < hashes.length; j++) {
            if (j === pi) continue;
            const hit = hashes[j].nearest(p.x, p.y, p.z, tol);
            if (!hit || hit.dist >= bestDist) continue;
            const s = hashes[j].samples;
            const sx = s.points[hit.index * 3], sy = s.points[hit.index * 3 + 1], sz = s.points[hit.index * 3 + 2];
            const nx = s.normals[hit.index * 3], ny = s.normals[hit.index * 3 + 1], nz = s.normals[hit.index * 3 + 2];
            // The other surface must face p: its outward normal points from S toward p.
            dir.set(p.x - sx, p.y - sy, p.z - sz);
            const len = dir.length() || 1e-9;
            if ((nx * dir.x + ny * dir.y + nz * dir.z) / len > 0.25)
                { best = { sx, sy, sz, dist: hit.dist }; bestDist = hit.dist; }
        }
        if (!best) return null;
        dir.set(best.sx - p.x, best.sy - p.y, best.sz - p.z);
        const len = dir.length() || 1e-9;
        // Both facing seams move toward each other, so each travels HALF the closure — they meet
        // near the middle of the gap and overlap by `overlap`, rather than each crossing the whole
        // gap and interpenetrating twice as far.
        const push = Math.min(best.dist + overlap, tol + overlap) / 2;
        return dir.multiplyScalar(push / len).clone();
    }

    return {
        displace(p, pi) {
            if (!otherBox[pi].containsPoint(p)) return; // nowhere near a seam
            const k = `${Math.round(p.x / QUANT)},${Math.round(p.y / QUANT)},${Math.round(p.z / QUANT)}`;
            let d = cache.get(k);
            if (d === undefined) { d = compute(p, pi); cache.set(k, d); }
            if (d) p.add(d);
        }
    };
}

/// One binary STL of every visible part, transforms baked in. Written via a per-triangle
/// aligned scratch block instead of per-field DataView calls — assembled kits run to millions
/// of triangles and the naive path blocked the main thread for the better part of a minute.
/// (STL is little-endian; typed arrays are little-endian on every platform we run on. The
/// 2-byte attribute field stays zero — ArrayBuffers are born zeroed.)
/// <param name="stitchTolMm">When > 0, close intentional slot-together gaps up to this many model
/// units between mating parts, so the export prints as one solid. 0 keeps every part exactly where
/// it sits.</param>
/// <param name="mode">'nudge' moves the mating surfaces to overlap (see <see cref="buildStitcher"/>);
/// 'bridge' leaves both parts untouched and adds filler geometry across the gap
/// (see <see cref="buildBridgeTriangles"/>).</param>
export function exportStlBytes(stitchTolMm = 0, mode = 'nudge') {
    const visible = [...parts.values()].filter(e => e.mesh.visible);
    const active = stitchTolMm > 0 && visible.length > 1;
    const stitch = active && mode === 'nudge' ? buildStitcher(visible, stitchTolMm) : null;
    const bridge = active && mode === 'bridge' ? buildBridgeTriangles(visible, stitchTolMm) : null;
    const bridgeTris = bridge ? bridge.length / 9 : 0;

    let totalTris = bridgeTris;
    for (const entry of visible) totalTris += entry.triangles;

    const buffer = new ArrayBuffer(84 + totalTris * 50);
    const out = new Uint8Array(buffer);
    out.set(new TextEncoder().encode('Chromatory assembly export'), 0);
    new DataView(buffer).setUint32(80, totalTris, true);

    const scratch = new Float32Array(12);
    const scratchBytes = new Uint8Array(scratch.buffer, 0, 48);
    const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3();
    const ab = new THREE.Vector3(), ac = new THREE.Vector3(), n = new THREE.Vector3();
    let offset = 84;
    const writeTri = () => {
        n.copy(ab.subVectors(b, a)).cross(ac.subVectors(c, a)).normalize();
        scratch[0] = n.x; scratch[1] = n.y; scratch[2] = n.z;
        scratch[3] = a.x; scratch[4] = a.y; scratch[5] = a.z;
        scratch[6] = b.x; scratch[7] = b.y; scratch[8] = b.z;
        scratch[9] = c.x; scratch[10] = c.y; scratch[11] = c.z;
        out.set(scratchBytes, offset);
        offset += 50;
    };
    for (let pi = 0; pi < visible.length; pi++) {
        const entry = visible[pi];
        entry.mesh.updateMatrixWorld();
        const matrix = entry.mesh.matrixWorld;
        const pos = entry.mesh.geometry.attributes.position;
        for (let t = 0; t < entry.triangles; t++) {
            a.fromBufferAttribute(pos, t * 3).applyMatrix4(matrix);
            b.fromBufferAttribute(pos, t * 3 + 1).applyMatrix4(matrix);
            c.fromBufferAttribute(pos, t * 3 + 2).applyMatrix4(matrix);
            if (stitch) { stitch.displace(a, pi); stitch.displace(b, pi); stitch.displace(c, pi); }
            writeTri();
        }
    }
    for (let i = 0; i < bridgeTris; i++) {
        a.set(bridge[i * 9], bridge[i * 9 + 1], bridge[i * 9 + 2]);
        b.set(bridge[i * 9 + 3], bridge[i * 9 + 4], bridge[i * 9 + 5]);
        c.set(bridge[i * 9 + 6], bridge[i * 9 + 7], bridge[i * 9 + 8]);
        writeTri();
    }
    return out;
}

/// Uploads the merged STL (e.g. as a Chromatory project model) without the bytes ever passing
/// through the .NET side — Blazor supplies the URL and auth header, fetch does the rest.
export async function uploadMerged(url, headerName, headerValue, filename, stitchTolMm = 0, mode = 'nudge') {
    status(stitchTolMm > 0 ? 'closing joint gaps + building the merged STL…' : 'building the merged STL…');
    await new Promise(r => setTimeout(r, 15)); // let the status paint before the bake blocks
    const bytes = exportStlBytes(stitchTolMm, mode);
    status(`uploading ${(bytes.byteLength / 1048576).toFixed(0)} MB…`);
    try {
        const response = await fetch(url + (url.includes('?') ? '&' : '?') + 'name=' + encodeURIComponent(filename), {
            method: 'POST',
            headers: { [headerName]: headerValue, 'Content-Type': 'application/octet-stream' },
            body: bytes
        });
        if (!response.ok) throw new Error(`upload failed: ${response.status}`);
        return await response.text();
    }
    finally { status(''); }
}

export function screenshot() {
    renderer.render(scene, camera);
    return renderer.domElement.toDataURL('image/png');
}
