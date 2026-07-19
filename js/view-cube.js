// Navigation cube (top-right of the canvas): a little labelled cube that rotates with the
// camera so you always know which way is up — and clicking a face (TOP, FRONT, LEFT…) animates
// the camera to that axis view around the current orbit target, keeping the distance.
// Self-contained and reusable: hand it the renderer/camera/controls/canvas, call render()
// after the main pass each frame, and give handlePointerDown() first refusal on clicks.
// NOTE: the host must run with renderer.autoClear = false and clear explicitly — the cube pass
// composites over the corner (clearDepth only), it must not blank it.
import * as THREE from 'three';

const SIZE = 92;    // css px
const MARGIN = 10;

// BoxGeometry material order: +X, -X, +Y, -Y, +Z, -Z. The -Y face's UVs render a canvas
// texture upside-down at a bottom view, so its label is drawn pre-rotated.
const FACES = [
    { label: 'RIGHT', dir: [1, 0, 0] },
    { label: 'LEFT', dir: [-1, 0, 0] },
    { label: 'TOP', dir: [0, 1, 0.0002] },     // epsilon keeps look-down valid with a Y-up camera
    { label: 'BOTTOM', dir: [0, -1, 0.0002], flip: true },
    { label: 'FRONT', dir: [0, 0, 1] },
    { label: 'BACK', dir: [0, 0, -1] },
];

function faceTexture(label, flip) {
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = 128;
    const g = canvas.getContext('2d');
    g.fillStyle = '#232b33';
    g.fillRect(0, 0, 128, 128);
    g.strokeStyle = '#46566a';
    g.lineWidth = 5;
    g.strokeRect(3, 3, 122, 122);
    g.fillStyle = '#d6dfe9';
    g.font = '600 24px system-ui, sans-serif';
    g.textAlign = 'center';
    g.textBaseline = 'middle';
    if (flip) { g.translate(64, 64); g.rotate(Math.PI); g.translate(-64, -64); }
    g.fillText(label, 64, 66);
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    return texture;
}

export function createViewCube(renderer, camera, controls, canvasEl) {
    const scene = new THREE.Scene();
    const cubeCamera = new THREE.OrthographicCamera(-1.75, 1.75, 1.75, -1.75, 0.1, 10);
    cubeCamera.position.set(0, 0, 5);
    const cube = new THREE.Mesh(
        new THREE.BoxGeometry(2, 2, 2),
        FACES.map(f => new THREE.MeshBasicMaterial({ map: faceTexture(f.label, f.flip) })));
    scene.add(cube);
    scene.add(new THREE.LineSegments(
        new THREE.EdgesGeometry(cube.geometry),
        new THREE.LineBasicMaterial({ color: 0x6b7d92 })));

    const raycaster = new THREE.Raycaster();
    let tween = null;
    let drag = null; // { startX, startY, moved } — dragging the cube orbits the camera

    const rect = () => ({ x: canvasEl.clientWidth - SIZE - MARGIN, y: MARGIN, w: SIZE, h: SIZE });

    function inRect(event) {
        const bounds = canvasEl.getBoundingClientRect();
        const px = event.clientX - bounds.left;
        const py = event.clientY - bounds.top;
        const r = rect();
        return px >= r.x && px <= r.x + r.w && py >= r.y && py <= r.y + r.h
            ? { px, py, r } : null;
    }

    function orbitBy(dx, dy) {
        const offset = camera.position.clone().sub(controls.target);
        const spherical = new THREE.Spherical().setFromVector3(offset);
        spherical.theta -= dx * 0.012;
        spherical.phi = Math.max(0.03, Math.min(Math.PI - 0.03, spherical.phi - dy * 0.012));
        offset.setFromSpherical(spherical);
        camera.position.copy(controls.target).add(offset);
        camera.lookAt(controls.target);
    }

    function stepTween() {
        tween.t = Math.min(1, tween.t + 1 / 18);
        const t = tween.t < 0.5 ? 2 * tween.t * tween.t : 1 - (-2 * tween.t + 2) ** 2 / 2; // ease in-out
        const q = new THREE.Quaternion().slerpQuaternions(new THREE.Quaternion(), tween.turn, t);
        const dir = tween.from.clone().applyQuaternion(q);
        camera.position.copy(controls.target).addScaledVector(dir, tween.radius);
        camera.lookAt(controls.target);
        if (tween.t >= 1) tween = null;
    }

    /// Call once per frame AFTER the main scene render.
    function render() {
        if (tween) stepTween();
        // Both edge lines and the cube follow the inverse camera orientation, so the cube shows
        // the WORLD axes as seen from the camera.
        scene.children.forEach(o => o.quaternion.copy(camera.quaternion).invert());
        const r = rect();
        const y = canvasEl.clientHeight - r.y - r.h; // three's viewport origin is bottom-left
        renderer.clearDepth();
        renderer.setScissorTest(true);
        renderer.setViewport(r.x, y, r.w, r.h);
        renderer.setScissor(r.x, y, r.w, r.h);
        renderer.render(scene, cubeCamera);
        renderer.setScissorTest(false);
        renderer.setViewport(0, 0, canvasEl.clientWidth, canvasEl.clientHeight);
    }

    function snapTo(dir) {
        const from = camera.position.clone().sub(controls.target);
        const fromDir = from.clone().normalize();
        // Clicking the face you're already looking at flips to the OPPOSITE view — at a
        // straight-on view the neighbours are edge-on and unclickable, so the aligned face
        // doubles as "look from the other side".
        if (fromDir.dot(dir) > 0.985) dir = dir.clone().negate();
        tween = {
            from: fromDir,
            turn: new THREE.Quaternion().setFromUnitVectors(fromDir, dir.clone().normalize()),
            radius: from.length(),
            t: 0
        };
    }

    /// Returns true when the event belongs to the widget (the host must then ignore it).
    /// A press starts a possible drag-orbit; the release decides click (snap) vs drag.
    function handlePointerDown(event) {
        const at = inRect(event);
        if (!at) return false;
        drag = { x: event.clientX, y: event.clientY, moved: false };
        canvasEl.setPointerCapture?.(event.pointerId);
        controls.enabled = false; // OrbitControls must not fight the widget's own orbit
        return true;
    }

    /// Returns true while a widget drag is orbiting the camera.
    function handlePointerMove(event) {
        if (!drag) return false;
        const dx = event.clientX - drag.x;
        const dy = event.clientY - drag.y;
        if (!drag.moved && dx * dx + dy * dy < 16) return true; // ±4px is still a click
        drag.moved = true;
        orbitBy(dx, dy);
        drag.x = event.clientX;
        drag.y = event.clientY;
        return true;
    }

    /// Returns true when the release ended a widget gesture (click → snap; drag → done).
    function handlePointerUp(event) {
        if (!drag) return false;
        const wasClick = !drag.moved;
        drag = null;
        controls.enabled = true;
        if (wasClick) {
            const at = inRect(event);
            if (at) {
                raycaster.setFromCamera(new THREE.Vector2(
                    ((at.px - at.r.x) / at.r.w) * 2 - 1,
                    -((at.py - at.r.y) / at.r.h) * 2 + 1), cubeCamera);
                const hit = raycaster.intersectObject(cube, false)[0];
                if (hit?.face) snapTo(new THREE.Vector3(...FACES[hit.face.materialIndex].dir).normalize());
            }
        }
        return true;
    }

    function dispose() {
        cube.geometry.dispose();
        cube.material.forEach(m => { m.map?.dispose(); m.dispose(); });
    }

    return { render, handlePointerDown, handlePointerMove, handlePointerUp, dispose };
}
