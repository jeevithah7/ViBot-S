// ViBot-S | 3D Antigravity SLAM Simulation Engine
// Three.js r128 — 6-DOF Nav · Point Cloud · LiDAR · SLAM · Path Planning

'use strict';

/* ═══════════════════ LOADER SEQUENCE ═══════════════════ */
const LOAD_STEPS = [
  'Loading 3D environment...',
  'Building geometry meshes...',
  'Initializing LiDAR sensor...',
  'Compiling SLAM engine...',
  'Planning navigation paths...',
  'Spinning up antigravity drive...',
  'System ready.',
];
let loadPct = 0;
const fill   = document.getElementById('loaderFill');
const lstat  = document.getElementById('loaderStatus');
function advanceLoader(i) {
  if (i >= LOAD_STEPS.length) { finishLoader(); return; }
  lstat.textContent = LOAD_STEPS[i];
  loadPct = Math.round((i + 1) / LOAD_STEPS.length * 100);
  fill.style.width = loadPct + '%';
  setTimeout(() => advanceLoader(i + 1), 340);
}
function finishLoader() {
  document.getElementById('loader').style.opacity = '0';
  document.getElementById('loader').style.transition = 'opacity 0.6s';
  setTimeout(() => {
    document.getElementById('loader').classList.add('hidden');
    document.getElementById('app').classList.remove('hidden');
    initApp();
  }, 650);
}
advanceLoader(0);

/* ═══════════════════ GLOBALS ═══════════════════════════ */
let scene, camera, renderer, clock;
let robot, rotorGroup, sensorRing;
let pointCloudGeom, pointCloudMat, pointCloudMesh;
let lidarLines, navGridHelper, pathLine, trailLine;
let fovCone, meshRecon;
let camMode = 'orbit';
let orbitAngle = 0.3, orbitPhi = 0.6, orbitR = 22;
let isRunning = false, isPaused = false, simTime = 0;
let speed = 1.5;
let pathWaypoints = [], currentWP = 0;
let trailPoints = [];
let lastFPSTime = 0, frameCount = 0, fps = 60;

/* Robot state — 6 DOF */
const RS = {
  x: 0, y: 1.8, z: 0,
  vx: 0, vy: 0, vz: 0,
  yaw: 0, pitch: 0, roll: 0,
  vyaw: 0
};

/* Key state */
const KEYS = {};
document.addEventListener('keydown', e => { KEYS[e.code] = true; });
document.addEventListener('keyup',   e => { KEYS[e.code] = false; });

/* Sensor distances */
const SENSORS = { front:5, rear:5, left:3, right:4, up:6, down:1.8 };

/* Toggle state */
const T = {
  pointCloud: true, mesh: false, lidar: true,
  navGrid: true, path: true, fov: true,
  trail: true, wireframe: false
};

/* ═══════════════════ ENVIRONMENT MAP ════════════════════ */
// occupancy: 0=free, 1=wall, cells are 1m
const MAP_W = 30, MAP_D = 24;
const CELL = 1.0;
const occupancy = [];
for (let r = 0; r < MAP_D; r++) {
  occupancy[r] = [];
  for (let c = 0; c < MAP_W; c++) {
    const edge = r===0||r===MAP_D-1||c===0||c===MAP_W-1;
    // inner walls
    const w1 = (r>5&&r<8&&c>3&&c<14);
    const w2 = (r>14&&r<17&&c>16&&c<27);
    const w3 = (c===10&&r>0&&r<10);
    const w4 = (c===20&&r>12&&r<24);
    const w5 = (r===11&&c>5&&c<22);
    // pillars
    const p1 = (r>7&&r<10&&c>7&&c<10);
    const p2 = (r>7&&r<10&&c>19&&c<22);
    const p3 = (r>14&&r<17&&c>7&&c<10);
    occupancy[r][c] = (edge||w1||w2||w3||w4||w5||p1||p2||p3) ? 1 : 0;
  }
}

function isFree(r, c) {
  if (r < 0 || r >= MAP_D || c < 0 || c >= MAP_W) return false;
  return occupancy[r][c] === 0;
}

/* ═══════════════════ OBSTACLE SAFETY BUFFER ═══════════════ */
const SAFETY_BUFFER = 1;
const inflatedOccupancy = [];
(function buildInflatedGrid() {
  for (let r = 0; r < MAP_D; r++) {
    inflatedOccupancy[r] = [];
    for (let c = 0; c < MAP_W; c++) inflatedOccupancy[r][c] = 0;
  }
  for (let r = 0; r < MAP_D; r++) {
    for (let c = 0; c < MAP_W; c++) {
      if (!occupancy[r][c]) continue;
      for (let dr = -SAFETY_BUFFER; dr <= SAFETY_BUFFER; dr++) {
        for (let dc = -SAFETY_BUFFER; dc <= SAFETY_BUFFER; dc++) {
          const nr = r + dr, nc = c + dc;
          if (nr >= 0 && nr < MAP_D && nc >= 0 && nc < MAP_W) {
            inflatedOccupancy[nr][nc] = 1;
          }
        }
      }
    }
  }
})();

function isSafeForPath(r, c) {
  if (r < 0 || r >= MAP_D || c < 0 || c >= MAP_W) return false;
  return inflatedOccupancy[r][c] === 0;
}

/* ═══════════════════ A* PATH PLANNER ═══════════════════ */
function astar(sr, sc, er, ec, useSafeGrid) {
  if (useSafeGrid === undefined) useSafeGrid = true;
  const checkFn = useSafeGrid ? isSafeForPath : isFree;
  const key = (r, c) => r * MAP_W + c;
  const h = (r, c) => Math.abs(r - er) + Math.abs(c - ec);
  const open = new Map();
  const closed = new Set();
  const came = new Map();
  const g = new Map();
  if (!isFree(sr, sc) || !isFree(er, ec)) {
    if (useSafeGrid) return astar(sr, sc, er, ec, false);
    return [];
  }
  g.set(key(sr, sc), 0);
  open.set(key(sr, sc), h(sr, sc));
  const dirs = [[0,1],[0,-1],[1,0],[-1,0],[1,1],[-1,-1],[1,-1],[-1,1]];
  while (open.size) {
    let bestK = null, bestF = Infinity;
    open.forEach((f, k) => { if (f < bestF) { bestF = f; bestK = k; } });
    if (bestK === null) break;
    const cr = Math.floor(bestK / MAP_W), cc = bestK % MAP_W;
    if (cr === er && cc === ec) {
      const path = [];
      let cur = bestK;
      while (came.has(cur)) { const r=Math.floor(cur/MAP_W),c=cur%MAP_W; path.unshift([r,c]); cur=came.get(cur); }
      path.unshift([sr,sc]);
      return path;
    }
    open.delete(bestK);
    closed.add(bestK);
    for (const [dr,dc] of dirs) {
      const nr=cr+dr, nc=cc+dc;
      const isGoal = (nr === er && nc === ec);
      if (isGoal ? !isFree(nr,nc) : !checkFn(nr,nc)) continue;
      const nk = key(nr,nc);
      if (closed.has(nk)) continue;
      const ng = (g.get(bestK)||0) + (Math.abs(dr)+Math.abs(dc)===2?1.41:1);
      if (ng < (g.get(nk)||Infinity)) {
        g.set(nk, ng);
        came.set(nk, bestK);
        open.set(nk, ng + h(nr,nc));
      }
    }
  }
  if (useSafeGrid) return astar(sr, sc, er, ec, false);
  return [];
}

/* ═══════════════════ THREE.JS INIT ═════════════════════ */
function initApp() {
  const canvas = document.getElementById('threeCanvas');
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xb0b8cc);
  scene.fog = new THREE.FogExp2(0xb0b8cc, 0.015);

  clock = new THREE.Clock();

  camera = new THREE.PerspectiveCamera(55, 1, 0.1, 200);
  camera.position.set(0, 14, 22);
  camera.lookAt(0, 0, 0);

  buildEnvironment();
  buildRobot();
  buildPointCloud();
  buildLidarViz();
  buildNavGrid();
  buildPathViz();
  buildTrailViz();
  buildFOVCone();
  buildMeshRecon();
  buildLights();
  planMission();
  bindUI();

  // Force resize after layout settles
  requestAnimationFrame(() => { requestAnimationFrame(() => {
    onResize();
    renderer.render(scene, camera);
  }); });
  window.addEventListener('resize', onResize);
  requestAnimationFrame(loop);

  toast('ViBot-S SLAM ready', 'info');
}

/* ═══════════════════ ENVIRONMENT ════════════════════════ */
function buildEnvironment() {
  // Floor
  const floorGeo = new THREE.PlaneGeometry(MAP_W, MAP_D);
  const floorMat = new THREE.MeshStandardMaterial({
    color: 0x999999, roughness: 0.8, metalness: 0.1,
  });
  const floor = new THREE.Mesh(floorGeo, floorMat);
  floor.rotation.x = -Math.PI / 2;
  floor.receiveShadow = true;
  scene.add(floor);

  // Grid overlay on floor
  const gridH = new THREE.GridHelper(MAP_W, MAP_W, 0xffffff, 0xd0d0d0);
  gridH.material.opacity = 0.4; gridH.material.transparent = true;
  scene.add(gridH);

  // Walls and obstacles from occupancy map
  const wallMat = new THREE.MeshStandardMaterial({ color:0xd0d0d0, roughness:0.9, metalness:0.0 });
  const wallMatE = new THREE.MeshStandardMaterial({ color:0xb0b0b0, roughness:0.9 });
  for (let r = 0; r < MAP_D; r++) {
    for (let c = 0; c < MAP_W; c++) {
      if (!occupancy[r][c]) continue;
      const isBorder = r===0||r===MAP_D-1||c===0||c===MAP_W-1;
      const geo = new THREE.BoxGeometry(CELL, isBorder?3:4, CELL);
      const mesh = new THREE.Mesh(geo, isBorder ? wallMatE : wallMat);
      mesh.position.set(c - MAP_W/2 + 0.5, (isBorder?3:4)/2, r - MAP_D/2 + 0.5);
      mesh.castShadow = true; mesh.receiveShadow = true;
      scene.add(mesh);
    }
  }
}

/* ═══════════════════ ROBOT MODEL ════════════════════════ */
function buildRobot() {
  robot = new THREE.Group();

  // Bottom Base
  const baseGeo = new THREE.BoxGeometry(0.35, 0.04, 0.2);
  const baseMat = new THREE.MeshStandardMaterial({ color: 0x222222, metalness:0.8, roughness:0.2 });
  const base = new THREE.Mesh(baseGeo, baseMat);
  base.position.y = 0.15;
  robot.add(base);

  // Wheels
  const wheelGeo = new THREE.CylinderGeometry(0.12, 0.12, 0.06, 16);
  const wheelMat = new THREE.MeshStandardMaterial({ color: 0x101530, roughness: 0.9 });
  const whL = new THREE.Mesh(wheelGeo, wheelMat);
  whL.rotation.x = Math.PI / 2;
  whL.position.set(-0.16, 0.12, 0);
  whL.userData.isWheel = true;
  robot.add(whL);
  
  const whR = new THREE.Mesh(wheelGeo, wheelMat);
  whR.rotation.x = Math.PI / 2;
  whR.position.set(0.16, 0.12, 0);
  whR.userData.isWheel = true;
  robot.add(whR);

  // Mid plate
  const midGeo = new THREE.BoxGeometry(0.35, 0.02, 0.22);
  const mid = new THREE.Mesh(midGeo, baseMat);
  mid.position.y = 0.28;
  robot.add(mid);
  
  // Top Arduino Deck
  const topGeo = new THREE.BoxGeometry(0.30, 0.03, 0.25);
  const topMat = new THREE.MeshStandardMaterial({ color: 0x004d40, roughness: 0.6 });
  const top = new THREE.Mesh(topGeo, topMat);
  top.position.y = 0.40;
  robot.add(top);

  // Standoffs
  const stMat = new THREE.MeshStandardMaterial({ color: 0xaaaaaa, metalness: 1 });
  for(let x of [-0.14, 0.14]) {
    for(let z of [-0.08, 0.08]) {
      const stand = new THREE.Mesh(new THREE.CylinderGeometry(0.015, 0.015, 0.26), stMat);
      stand.position.set(x, 0.27, z);
      robot.add(stand);
    }
  }

  // LiDAR Pod
  const lidarPod = new THREE.Mesh(
    new THREE.CylinderGeometry(0.06,0.06,0.08,8),
    new THREE.MeshStandardMaterial({ color:0x111111 })
  );
  lidarPod.position.y = 0.46;
  robot.add(lidarPod);

  // Glowing ring for sensor visualization
  const ringGeo = new THREE.TorusGeometry(0.3, 0.015, 8, 32);
  sensorRing = new THREE.Mesh(ringGeo, new THREE.MeshBasicMaterial({ color: 0x00f5ff }));
  sensorRing.rotation.x = Math.PI / 2;
  sensorRing.position.y = 0.05;
  robot.add(sensorRing);
  
  rotorGroup = new THREE.Group(); // Mock group to prevent errors

  robot.position.set(RS.x, RS.y, RS.z);
  scene.add(robot);
}

/* ═══════════════════ POINT CLOUD ════════════════════════ */
function buildPointCloud() {
  const MAX_PTS = 20000;
  pointCloudGeom = new THREE.BufferGeometry();
  const pos = new Float32Array(MAX_PTS * 3);
  const col = new Float32Array(MAX_PTS * 3);
  pointCloudGeom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  pointCloudGeom.setAttribute('color',    new THREE.BufferAttribute(col, 3));
  pointCloudGeom.setDrawRange(0, 0);
  pointCloudMat = new THREE.PointsMaterial({ size:0.04, vertexColors:true, transparent:true, opacity:0.9, sizeAttenuation:true });
  pointCloudMesh = new THREE.Points(pointCloudGeom, pointCloudMat);
  scene.add(pointCloudMesh);
}

let pcCount = 0;
function addPointCloudHit(wx, wy, wz, dist) {
  if (pcCount >= 20000) return;
  const pa = pointCloudGeom.attributes.position.array;
  const ca = pointCloudGeom.attributes.color.array;
  const i3 = pcCount * 3;
  pa[i3]=wx; pa[i3+1]=wy; pa[i3+2]=wz;
  
  // Photogrammetry style: Greyscale/White dense cloud based on depth
  const depthFactor = 1.0 - Math.min(dist / 14, 0.9);
  const intns = 0.6 + depthFactor * 0.4 + (Math.random() * 0.1);
  ca[i3]   = intns;
  ca[i3+1] = intns;
  ca[i3+2] = intns;
  
  pcCount++;
  pointCloudGeom.setDrawRange(0, pcCount);
  pointCloudGeom.attributes.position.needsUpdate = true;
  pointCloudGeom.attributes.color.needsUpdate = true;
}

/* ═══════════════════ LIDAR VISUALIZATION ════════════════ */
function buildLidarViz() {
  const N = 120;
  const positions = new Float32Array(N * 2 * 3);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions,3));
  const mat = new THREE.LineBasicMaterial({ color:0x00ff88, transparent:true, opacity:0.4 });
  lidarLines = new THREE.LineSegments(geo, mat);
  scene.add(lidarLines);
}

function updateLidar() {
  if (!lidarLines) return;
  const N = 120; // Denser LiDAR sweep
  const pos = lidarLines.geometry.attributes.position.array;
  for (let i = 0; i < N; i++) {
    const ang = (i / N) * Math.PI * 2 + RS.yaw;
    const reach = 8.0 + Math.random() * 0.5;
    const hit = castRay(RS.x, RS.y, RS.z, Math.cos(ang), 0, Math.sin(ang), reach);
    const i6 = i * 6;
    if (i6+5 < pos.length) {
      pos[i6]   = RS.x; pos[i6+1] = RS.y; pos[i6+2] = RS.z;
      pos[i6+3] = hit.x; pos[i6+4] = hit.y; pos[i6+5] = hit.z;
    }
    if (Math.random() < 0.25) addPointCloudHit(hit.x, hit.y + (Math.random()-0.5)*1.5, hit.z, hit.d);
  }
  lidarLines.geometry.attributes.position.needsUpdate = true;
}

function castRay(ox, oy, oz, dx, dy, dz, maxD) {
  const step = 0.25;
  for (let t = step; t <= maxD; t += step) {
    const wx = ox + dx*t, wz = oz + dz*t;
    const gc = Math.floor(wx + MAP_W/2), gr = Math.floor(wz + MAP_D/2);
    if (gc<0||gc>=MAP_W||gr<0||gr>=MAP_D||occupancy[gr][gc]) {
      return { x: ox+dx*t, y: oy+dy*t, z: oz+dz*t, d: t };
    }
  }
  return { x: ox+dx*maxD, y: oy+dy*maxD, z: oz+dz*maxD, d: maxD };
}

/* ═══════════════════ NAV GRID ══════════════════════════ */
function buildNavGrid() {
  navGridHelper = new THREE.GridHelper(MAP_W, MAP_W, 0x003366, 0x001a33);
  navGridHelper.material.opacity = 0.4; navGridHelper.material.transparent = true;
  navGridHelper.position.y = 0.02;
  scene.add(navGridHelper);
}

/* ═══════════════════ PATH VISUALIZATION ════════════════ */
function buildPathViz() {
  const geo = new THREE.BufferGeometry();
  pathLine = new THREE.Line(geo, new THREE.LineBasicMaterial({ color:0xff6b35, linewidth:2 }));
  scene.add(pathLine);
}

function updatePathViz() {
  if (!pathWaypoints.length) {
    pathLine.geometry.setFromPoints([]);
    return;
  }
  const pts = pathWaypoints.map(([r,c]) =>
    new THREE.Vector3(c - MAP_W/2 + 0.5, RS.y, r - MAP_D/2 + 0.5));
  pathLine.geometry.setFromPoints(pts);
}

/* ═══════════════════ TRAIL ══════════════════════════════ */
function buildTrailViz() {
  const geo = new THREE.BufferGeometry();
  trailLine = new THREE.Line(geo, new THREE.LineBasicMaterial({ color:0xff3366, transparent:true, opacity:0.8, linewidth:3 }));
  scene.add(trailLine);
}

function updateTrailViz() {
  if (trailPoints.length < 2) return;
  const pts = trailPoints.map(p => new THREE.Vector3(p.x, 0.2, p.z));
  trailLine.geometry.setFromPoints(pts);
}

/* ═══════════════════ FOV CONE ═══════════════════════════ */
function buildFOVCone() {
  const geo = new THREE.ConeGeometry(2, 4, 12, 1, true);
  const mat = new THREE.MeshBasicMaterial({ color:0x00f5ff, transparent:true, opacity:0.07, side:THREE.DoubleSide });
  fovCone = new THREE.Mesh(geo, mat);
  fovCone.rotation.x = Math.PI/2;
  fovCone.position.y = 0.4;
  scene.add(fovCone);
}

/* ═══════════════════ MESH RECONSTRUCTION ═══════════════ */
function buildMeshRecon() {
  meshRecon = new THREE.Group();
  // Simulated semantic/feature extraction: Red polylines connecting detected features
  const createPolyFeature = (cx, cz, w) => {
    const pts = [];
    for(let i=0; i<6; i++) {
       pts.push(new THREE.Vector3(cx + (Math.random()-0.5)*w, 0.1, cz + (Math.random()-0.5)*w));
    }
    pts.push(pts[0]); // close loop
    const geo = new THREE.BufferGeometry().setFromPoints(pts);
    const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ color: 0xff3366, linewidth: 2 }));
    
    // add vertex spheres
    pts.forEach(p => {
       const sphere = new THREE.Mesh(new THREE.SphereGeometry(0.12, 8, 8), new THREE.MeshBasicMaterial({ color: 0xffffff }));
       sphere.position.copy(p);
       line.add(sphere);
    });
    return line;
  };

  meshRecon.add(createPolyFeature(-5, 5, 4));
  meshRecon.add(createPolyFeature(8, -8, 5));
  meshRecon.add(createPolyFeature(-2, -6, 3));
  meshRecon.add(createPolyFeature(4, 12, 4));
  
  scene.add(meshRecon);
}

/* ═══════════════════ LIGHTS ════════════════════════════ */
function buildLights() {
  scene.add(new THREE.AmbientLight(0xffffff, 0.8));
  const dir = new THREE.DirectionalLight(0xfffaed, 1.2);
  dir.position.set(5, 12, 5);
  dir.castShadow = true;
  dir.shadow.mapSize.width = 1024;
  dir.shadow.mapSize.height = 1024;
  scene.add(dir);
  
  const dir2 = new THREE.DirectionalLight(0xb1ccff, 0.6);
  dir2.position.set(-8, 8, -5);
  scene.add(dir2);
}

/* ═══════════════════ MISSION PLAN ══════════════════════ */
const WAYPOINTS_DEF = [
  [2,2],[2,8],[4,8],[4,15],[2,15],[2,25],[5,25],[9,25],[13,25],[13,18],[18,18],[18,12],[20,12],[20,6],[18,6],[14,6],[14,2],[4,2]
];
function planMission() {
  pathWaypoints = [];
  
  // Custom Waypoint parsing
  const inpS = document.getElementById('inpStart');
  const inpE = document.getElementById('inpEnd');
  let pts = [...WAYPOINTS_DEF];
  
  if ((inpE && inpE.value) || (inpS && inpS.value)) {
    // Determine end coordinates:
    let ec, er;
    if (inpE && inpE.value) {
      const endParts = inpE.value.split(',');
      if (endParts.length < 2) {
         toast('Invalid End format. Use: X,Y', 'warn'); return;
      }
      ec = parseInt(endParts[0].trim());
      er = parseInt(endParts[1].trim());
      if (isNaN(ec) || isNaN(er)) {
         toast('Invalid End numbers', 'warn'); return;
      }
    } else {
      toast('Please provide an End coordinate', 'warn'); return;
    }

    // Determine start coordinates:
    let sc, sr;
    if (inpS && inpS.value) {
      const startParts = inpS.value.split(',');
      if (startParts.length < 2) {
         toast('Invalid Start format. Use: X,Y', 'warn'); return;
      }
      sc = parseInt(startParts[0].trim());
      sr = parseInt(startParts[1].trim());
      if (isNaN(sc) || isNaN(sr)) {
         toast('Invalid Start numbers', 'warn'); return;
      }
    } else {
      // Use bot's current logical location
      sc = Math.max(0, Math.min(MAP_W-1, Math.round(RS.x + MAP_W/2 - 0.5)));
      sr = Math.max(0, Math.min(MAP_D-1, Math.round(RS.z + MAP_D/2 - 0.5)));
    }
    
    pts = [[sr, sc], [er, ec]];
  }

  let pathFound = true;
  for (let i = 0; i < pts.length-1; i++) {
    const [sr,sc] = pts[i], [er,ec] = pts[i+1];
    const seg = astar(sr,sc,er,ec);
    if (seg.length === 0) {
      toast(`Cannot reach (${ec}, ${er}) - Coordinates are blocked by walls!`, 'error');
      addLog(`[PLAN] Error: Destination (${ec}, ${er}) is enclosed in an obstacle.`, 'error');
      pathFound = false;
      continue;
    }
    if (i===0) pathWaypoints.push(...seg);
    else        pathWaypoints.push(...seg.slice(1));
  }
  
  if (!pathFound || pathWaypoints.length === 0) {
     pathWaypoints = [];
     currentWP = 0;
     updatePathViz();
     addLog(`[PLAN] Mission aborted - No valid path`, 'warn');
     return false;
  }
  // Teleport ONLY if the user explicitly typed a start coordinate
  if (pathWaypoints.length && inpS && inpS.value) {
    const [sr, sc] = pathWaypoints[0];
    RS.x = sc - MAP_W/2 + 0.5;
    RS.z = sr - MAP_D/2 + 0.5;
    RS.y = 0; // Grounded 2-wheel robot
  }
  currentWP = 0;
  if(document.getElementById('pathProg')) document.getElementById('pathProg').textContent = `0/${pathWaypoints.length}`;
  updatePathViz();
  addLog(`[PLAN] ${pathWaypoints.length} waypoints computed`, 'ok');
  
  const st = document.getElementById('statusText');
  if (st && st.textContent === 'COMPLETE') st.textContent = 'READY';
  
  return true;
}

/* ═══════════════════ SENSOR UPDATE ═════════════════════ */
function updateSensors() {
  const dirs = [
    ['front', Math.cos(RS.yaw), 0, Math.sin(RS.yaw)],
    ['rear', -Math.cos(RS.yaw), 0, -Math.sin(RS.yaw)],
    ['left', -Math.sin(RS.yaw), 0, Math.cos(RS.yaw)],
    ['right', Math.sin(RS.yaw), 0, -Math.cos(RS.yaw)],
    ['up', 0, 1, 0],
    ['down', 0,-1, 0],
  ];
  dirs.forEach(([name, dx, dy, dz]) => {
    const h = castRay(RS.x, RS.y, RS.z, dx, dy, dz, 8);
    SENSORS[name] = h.d;
    const pct = Math.min(h.d / 8, 1) * 100;
    const el = document.getElementById('s'+name.charAt(0).toUpperCase()+name.slice(1));
    const el2 = document.getElementById('s'+name.charAt(0).toUpperCase()+name.slice(1)+'Val');
    if (el) el.style.width = pct + '%';
    if (el2) el2.textContent = h.d.toFixed(1)+'m';
  });

  const risk = (SENSORS.front < 1.2 || SENSORS.left < 0.8 || SENSORS.right < 0.8) ? 'HIGH' : SENSORS.front < 2.5 ? 'MED' : 'LOW';
  const re = document.getElementById('collRisk');
  if (re) { re.textContent = risk; re.className = 'state-val' + (risk==='HIGH'?' red':risk==='MED'?' orange':' green'); }
}

/* ═══════════════════ MINIMAP ════════════════════════════ */
const mmCanvas = document.getElementById('minimapCanvas');
const mmCtx    = mmCanvas ? mmCanvas.getContext('2d') : null;
function drawMinimap() {
  if (!mmCtx) return;
  const cw = mmCanvas.width, ch = mmCanvas.height;
  const csx = cw / MAP_W, csy = ch / MAP_D;
  mmCtx.clearRect(0,0,cw,ch);
  // Map cells
  for (let r=0;r<MAP_D;r++) for (let c=0;c<MAP_W;c++) {
    mmCtx.fillStyle = occupancy[r][c] ? '#ff3366' : '#0a2040';
    mmCtx.fillRect(c*csx, r*csy, csx, csy);
  }
  // Path
  if (pathWaypoints.length) {
    mmCtx.strokeStyle='#ff6b35'; mmCtx.lineWidth=1; mmCtx.beginPath();
    pathWaypoints.forEach(([r,c],i) => i===0?mmCtx.moveTo((c+0.5)*csx,(r+0.5)*csy):mmCtx.lineTo((c+0.5)*csx,(r+0.5)*csy));
    mmCtx.stroke();
  }
  // Robot
  const rc = RS.x + MAP_W/2, rr = RS.z + MAP_D/2;
  mmCtx.fillStyle='#00f5ff'; mmCtx.beginPath(); mmCtx.arc(rc*csx,rr*csy,3,0,Math.PI*2); mmCtx.fill();
  // heading arrow
  mmCtx.strokeStyle='#00f5ff'; mmCtx.lineWidth=1.5; mmCtx.beginPath();
  mmCtx.moveTo(rc*csx,rr*csy);
  mmCtx.lineTo((rc+Math.cos(RS.yaw)*2)*csx,(rr+Math.sin(RS.yaw)*2)*csy);
  mmCtx.stroke();
}

/* ═══════════════════ ROBOT POV ══════════════════════════ */
const povCanvas = document.getElementById('povCanvas');
const povCtx    = povCanvas ? povCanvas.getContext('2d') : null;
function drawPOV() {
  if (!povCtx) return;
  const w=povCanvas.width, h=povCanvas.height;
  povCtx.clearRect(0,0,w,h);
  // Sky gradient
  const sky = povCtx.createLinearGradient(0,0,0,h*0.55);
  sky.addColorStop(0,'#010810'); sky.addColorStop(1,'#021830');
  povCtx.fillStyle=sky; povCtx.fillRect(0,0,w,h*0.55);
  // Floor
  const flr = povCtx.createLinearGradient(0,h*0.55,0,h);
  flr.addColorStop(0,'#040e1e'); flr.addColorStop(1,'#010810');
  povCtx.fillStyle=flr; povCtx.fillRect(0,h*0.55,w,h*0.45);
  // Horizon line
  povCtx.strokeStyle='rgba(0,245,255,0.3)'; povCtx.lineWidth=1;
  povCtx.beginPath(); povCtx.moveTo(0,h*0.55); povCtx.lineTo(w,h*0.55); povCtx.stroke();
  // Perspective grid
  povCtx.strokeStyle='rgba(0,245,255,0.12)'; povCtx.lineWidth=0.5;
  for(let i=1;i<8;i++){
    const y=h*0.55+i*(h*0.45/8);
    povCtx.beginPath(); povCtx.moveTo(0,y); povCtx.lineTo(w,y); povCtx.stroke();
    const vp=w/2, spread=(i/8)*(w*0.4);
    povCtx.beginPath(); povCtx.moveTo(vp,h*0.55); povCtx.lineTo(vp-spread,y); povCtx.stroke();
    povCtx.beginPath(); povCtx.moveTo(vp,h*0.55); povCtx.lineTo(vp+spread,y); povCtx.stroke();
  }
  // HUD crosshair
  const cx=w/2, cy=h/2;
  povCtx.strokeStyle='rgba(0,245,255,0.6)'; povCtx.lineWidth=1;
  [[-20,0,20,0],[0,-20,0,20]].forEach(([x1,y1,x2,y2]) => {
    povCtx.beginPath(); povCtx.moveTo(cx+x1,cy+y1); povCtx.lineTo(cx+x2,cy+y2); povCtx.stroke();
  });
  povCtx.strokeStyle='rgba(0,245,255,0.4)'; povCtx.beginPath();
  povCtx.arc(cx,cy,6,0,Math.PI*2); povCtx.stroke();
  // Obstacle indicators
  const dirs2 = [{label:'F',d:SENSORS.front},{label:'R',d:SENSORS.right},{label:'L',d:SENSORS.left}];
  dirs2.forEach((s,i) => {
    const t=(s.d<2)?1:(s.d<4)?0.5:0.1;
    const col=`rgba(255,${Math.floor(51+t*204)},${Math.floor(t*50)},${0.6+t*0.4})`;
    povCtx.fillStyle=col; povCtx.font='bold 9px JetBrains Mono,monospace';
    povCtx.fillText(`${s.label}:${s.d.toFixed(1)}m`,(i*88)+8,h-8);
  });
  // Altitude bar right side (Removed or repurposed)
  const dToGoal = document.getElementById('distGoal') ? parseFloat(document.getElementById('distGoal').textContent) || 0 : 0;
  const dPct = Math.min(1, dToGoal / 30);
  povCtx.fillStyle='rgba(0,128,255,0.3)';
  povCtx.fillRect(w-14, h*(1-dPct), 6, h*dPct);
  povCtx.strokeStyle='#0080ff'; povCtx.strokeRect(w-14,0,6,h);
  // Update POV labels
  if (document.getElementById('povYaw'))   document.getElementById('povYaw').textContent   = (RS.yaw*180/Math.PI).toFixed(1)+'°';
}

/* ═══════════════════ AUTONOMOUS NAVIGATION ═════════════ */
let replanCooldown = 0;
let replanCount = 0;
function navigateAuto(dt) {
  if (!pathWaypoints.length || currentWP >= pathWaypoints.length) {
    if (isRunning) { isRunning=false; addLog('[NAV] Mission complete!','ok'); toast('Mission complete!','info'); document.getElementById('statusText').textContent='COMPLETE'; }
    return;
  }
  if (replanCooldown > 0) replanCooldown -= dt;

  const [tr, tc] = pathWaypoints[currentWP];
  const tx = tc - MAP_W/2 + 0.5, tz = tr - MAP_D/2 + 0.5;
  const dx = tx - RS.x, dz = tz - RS.z;
  const dist = Math.sqrt(dx*dx+dz*dz);

  if (dist < 0.3) { currentWP++; return; }

  // ─── Collision Detection & Dynamic Re-planning ───
  const nearObstacle = SENSORS.front < 1.5 || SENSORS.left < 0.7 || SENSORS.right < 0.7;
  if (nearObstacle && replanCooldown <= 0) {
    replanCooldown = 2.0;
    const curC = Math.max(0, Math.min(MAP_W-1, Math.round(RS.x + MAP_W/2 - 0.5)));
    const curR = Math.max(0, Math.min(MAP_D-1, Math.round(RS.z + MAP_D/2 - 0.5)));
    const [goalR, goalC] = pathWaypoints[pathWaypoints.length - 1];
    addLog('[REPLAN] Obstacle detected — re-routing...', 'warn');
    document.getElementById('navMode').textContent = 'REPLAN';
    const newPath = astar(curR, curC, goalR, goalC);
    if (newPath.length > 0) {
      pathWaypoints = newPath;
      currentWP = 0;
      replanCount++;
      updatePathViz();
      addLog(`[REPLAN] New path: ${newPath.length} waypoints (replan #${replanCount})`, 'ok');
    } else {
      addLog('[REPLAN] No alternate path — backing up', 'warn');
      RS.x -= Math.cos(RS.yaw) * 0.15;
      RS.z -= Math.sin(RS.yaw) * 0.15;
    }
    setTimeout(() => { document.getElementById('navMode').textContent = 'AUTO'; }, 800);
    return;
  }

  // ─── Smooth Yaw Interpolation ───
  const targetYaw = Math.atan2(dz, dx);
  let dyaw = targetYaw - RS.yaw;
  while (dyaw >  Math.PI) dyaw -= Math.PI*2;
  while (dyaw < -Math.PI) dyaw += Math.PI*2;
  RS.yaw += dyaw * Math.min(dt*4, 0.15);

  // ─── Movement with dynamic speed near obstacles ───
  const spd = speed * dt * Math.min(dist, 1);
  let moveFactor = 1.0;
  if (SENSORS.front < 1.0) moveFactor = 0.25;
  else if (SENSORS.front < 2.0) moveFactor = 0.5;
  else if (SENSORS.front < 3.0) moveFactor = 0.75;
  RS.x += Math.cos(RS.yaw) * spd * moveFactor;
  RS.z += Math.sin(RS.yaw) * spd * moveFactor;
  RS.y = 0;

  // ─── Progress ───
  const pct = Math.round(currentWP / pathWaypoints.length * 100);
  document.getElementById('pathProg').textContent = `${currentWP}/${pathWaypoints.length}`;
  document.getElementById('missionProg').style.width = pct+'%';
  document.getElementById('missionPct').textContent = pct+'%';
  document.getElementById('mapCov').textContent = Math.min(pct+12,99)+'%';

  const goal = pathWaypoints[pathWaypoints.length-1];
  const gx=goal[1]-MAP_W/2+0.5, gz=goal[0]-MAP_D/2+0.5;
  const distG=Math.sqrt((RS.x-gx)**2+(RS.z-gz)**2);
  document.getElementById('distGoal').textContent = distG.toFixed(1)+'m';
  document.getElementById('wpLabel').textContent = `WP ${currentWP}`;
}

/* Manual control removed — fully autonomous navigation */

/* ═══════════════════ CAMERA LOGIC ══════════════════════ */
function updateCamera(dt) {
  if (camMode==='orbit') {
    orbitAngle += dt * 0.18;
    const cx=Math.cos(orbitAngle)*orbitR*Math.cos(orbitPhi), cy=Math.sin(orbitPhi)*orbitR*0.7, cz=Math.sin(orbitAngle)*orbitR*Math.cos(orbitPhi);
    camera.position.set(RS.x+cx, RS.y+cy, RS.z+cz);
    camera.lookAt(RS.x, RS.y, RS.z);
  } else if (camMode==='pov') {
    camera.position.set(RS.x, RS.y+0.25, RS.z);
    camera.rotation.set(0, RS.yaw+Math.PI/2+Math.PI, 0, 'YXZ');
  } else if (camMode==='top') {
    camera.position.set(RS.x, 18, RS.z);
    camera.lookAt(RS.x, 0, RS.z);
  } else if (camMode==='cinema') {
    const lead = 8;
    camera.position.set(RS.x-Math.cos(RS.yaw)*lead, RS.y+5, RS.z-Math.sin(RS.yaw)*lead);
    camera.lookAt(RS.x, RS.y, RS.z);
  }
}

/* ═══════════════════ MAIN LOOP ═════════════════════════ */
function loop() {
  requestAnimationFrame(loop);
  try {
    const dt = Math.min(clock.getDelta(), 0.05);
    simTime += dt;

    // FPS
    frameCount++;
    if (simTime - lastFPSTime > 0.5) {
      fps = Math.round(frameCount / (simTime-lastFPSTime));
      frameCount=0; lastFPSTime=simTime;
      const fE = document.getElementById('fpsCounter');
      if(fE) fE.textContent=fps;
    }

    if (!isPaused) {
      if (isRunning) navigateAuto(dt);

      // Clamp to valid space
      RS.x = Math.max(-MAP_W/2+1, Math.min(MAP_W/2-1, RS.x));
      RS.z = Math.max(-MAP_D/2+1, Math.min(MAP_D/2-1, RS.z));
      RS.y = 0;
    }

    // Animate robot
    if (robot) {
      robot.position.set(RS.x, RS.y, RS.z);
      robot.rotation.y = -RS.yaw; // Diff drive visual matching
      robot.children.forEach(c => {
         if (c.userData.isWheel) {
            c.rotation.y += Math.sqrt(RS.vx*RS.vx+RS.vz*RS.vz) * dt * 5;
         }
      });
    }
    if (rotorGroup) rotorGroup.rotation.y += dt * 18;
    if (sensorRing) sensorRing.material.color.setHSL((simTime*0.3)%1, 1, 0.55);

    // Trail
    trailPoints.push({x:RS.x,y:RS.y,z:RS.z});
    if (trailPoints.length>400) trailPoints.shift();
    if (T.trail) updateTrailViz();

    // Sensors + LiDAR
    updateSensors();
    if (T.lidar) updateLidar();

    // FOV cone
    if (T.fov && fovCone) {
      fovCone.position.set(RS.x, RS.y, RS.z);
      fovCone.rotation.y = -RS.yaw - Math.PI/2;
    }
    if (fovCone) fovCone.visible = T.fov;
    if (pointCloudMesh) pointCloudMesh.visible = T.pointCloud;
    if (navGridHelper) navGridHelper.visible = T.navGrid;
    if (pathLine) pathLine.visible = T.path;
    if (trailLine) trailLine.visible = T.trail;
    if (lidarLines) lidarLines.visible = T.lidar;
    if (meshRecon) meshRecon.visible = T.mesh;

    updateCamera(dt);
    drawMinimap();
    drawPOV();
    updateTelemetry();

    // Sim time display
    const mins=Math.floor(simTime/60), secs=Math.floor(simTime%60);
    const stE = document.getElementById('simTime');
    if (stE) stE.textContent=`${String(mins).padStart(2,'0')}:${String(secs).padStart(2,'0')}`;

    renderer.render(scene, camera);
  } catch (err) {
    console.error(err);
    addLog('[SYS ERR] ' + err.message, 'warn');
  }
}

/* ═══════════════════ TELEMETRY UPDATE ══════════════════ */
function updateTelemetry() {
  document.getElementById('posX').textContent = RS.x.toFixed(2);
  document.getElementById('posY').textContent = RS.z.toFixed(2);
  document.getElementById('heading').textContent = ((RS.yaw*180/Math.PI+360)%360).toFixed(1);
  document.getElementById('velocity').textContent = (speed*0.8).toFixed(2);
  document.getElementById('angVel').textContent    = (RS.vyaw*180/Math.PI).toFixed(2);
  document.getElementById('pointCount').textContent = pcCount;
  // Replan counters
  const rcEl = document.getElementById('replanCount');
  const rsEl = document.getElementById('replanState');
  if (rcEl) rcEl.textContent = replanCount;
  if (rsEl) rsEl.textContent = replanCount;
  if (rcEl && replanCooldown > 0) { rcEl.style.color = 'var(--accent-orange)'; rcEl.style.textShadow = '0 0 8px rgba(255,107,53,0.8)'; }
  else if (rcEl) { rcEl.style.color = ''; rcEl.style.textShadow = ''; }
  // Battery drain
  const bat=Math.max(0,98-simTime*0.02).toFixed(0)+'%';
  const be=document.getElementById('battery');
  if(be){be.textContent=bat; be.className='state-val'+(parseInt(bat)<30?' red':parseInt(bat)<60?' orange':' green');}
}

/* ═══════════════════ UI BINDINGS ════════════════════════ */
function bindUI() {
  // Sim controls
  document.getElementById('btnStart').onclick=()=>{
    const success = planMission();
    if (!success) return; // aborted due to invalid coords
    isRunning=true;isPaused=false;
    document.getElementById('statusText').textContent='NAVIGATING';
    document.getElementById('navMode').textContent='AUTO';
    addLog('[SIM] Autonomous navigation started','ok');
    toast('Navigation started');
  };
  document.getElementById('btnPause').onclick=()=>{isPaused=!isPaused;document.getElementById('statusText').textContent=isPaused?'PAUSED':'NAVIGATING';};
  document.getElementById('btnStop').onclick=()=>{isRunning=false;isPaused=false;document.getElementById('statusText').textContent='STOPPED';};
  document.getElementById('btnRestart').onclick=()=>{document.getElementById('inpStart').value='';document.getElementById('inpEnd').value='';RS.x=0;RS.y=0;RS.z=0;RS.yaw=0;trailPoints=[];pcCount=0;pointCloudGeom.setDrawRange(0,0);currentWP=0;isRunning=false;isPaused=false;replanCooldown=0;replanCount=0;planMission();toast('Simulation restarted');};
  
  // Drop Obstacle Button
  const btnDrop = document.getElementById('btnDropObstacle');
  if (btnDrop) btnDrop.onclick = () => {
    const dist = 3.0;
    const ox = RS.x + Math.cos(RS.yaw) * dist;
    const oz = RS.z + Math.sin(RS.yaw) * dist;
    const c = Math.floor(ox + MAP_W/2);
    const r = Math.floor(oz + MAP_D/2);
    if (r >= 0 && r < MAP_D && c >= 0 && c < MAP_W) {
      if (occupancy[r][c] === 1) return; // already an obstacle there
      occupancy[r][c] = 1;
      for (let dr = -SAFETY_BUFFER; dr <= SAFETY_BUFFER; dr++) {
        for (let dc = -SAFETY_BUFFER; dc <= SAFETY_BUFFER; dc++) {
          const nr = r + dr, nc = c + dc;
          if (nr >= 0 && nr < MAP_D && nc >= 0 && nc < MAP_W) inflatedOccupancy[nr][nc] = 1;
        }
      }
      const geo = new THREE.BoxGeometry(CELL, 4, CELL);
      const wallMat = new THREE.MeshStandardMaterial({ color:0xff6b35, roughness:0.9 });
      const mesh = new THREE.Mesh(geo, wallMat);
      mesh.position.set(c - MAP_W/2 + 0.5, 2, r - MAP_D/2 + 0.5);
      mesh.castShadow = true; mesh.receiveShadow = true;
      scene.add(mesh);
      addLog(`[SIM] Obstacle dropped at (${c}, ${r})`, 'warn');
    }
  };
  
  // Real-time map update if waypoints are typed
  const updateMapCb = () => { if(!isRunning) { planMission(); } };
  document.getElementById('inpStart').addEventListener('change', updateMapCb);
  document.getElementById('inpEnd').addEventListener('change', updateMapCb);

  // Minimap Click-to-Pick
  if (mmCanvas) {
    mmCanvas.style.cursor = 'crosshair';
    mmCanvas.onclick = (e) => {
      const rect = mmCanvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const col = Math.floor((x / mmCanvas.width) * MAP_W);
      const row = Math.floor((y / mmCanvas.height) * MAP_D);
      
      if (e.ctrlKey) {
        document.getElementById('inpStart').value = `${col},${row}`;
        toast(`Start set to ${col},${row}`);
      } else {
        document.getElementById('inpEnd').value = `${col},${row}`;
        toast(`Goal set to ${col},${row}`);
      }
      
      if (!isRunning) planMission();
    };
    
    mmCanvas.onmousemove = (e) => {
      const rect = mmCanvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const col = Math.floor((x / mmCanvas.width) * MAP_W);
      const row = Math.floor((y / mmCanvas.height) * MAP_D);
      const scaleEl = document.getElementById('mmScale');
      if (scaleEl) scaleEl.textContent = `COORD: ${col},${row}`;
    };
    mmCanvas.onmouseleave = () => {
      const scaleEl = document.getElementById('mmScale');
      if (scaleEl) scaleEl.textContent = `1:1`;
    };
  }

  // Speed
  const ss=document.getElementById('speedSlider');
  if(ss) ss.oninput=()=>{speed=parseFloat(ss.value);document.getElementById('speedVal').textContent=speed.toFixed(1)+'×';};

  // Manual direction buttons removed — fully autonomous

  // Camera
  const camBtns = {'camOrbit':'orbit','camPOV':'pov','camTop':'top','camCinema':'cinema'};
  Object.entries(camBtns).forEach(([id,mode])=>{
    document.getElementById(id).onclick=()=>{
      camMode=mode;document.getElementById('modeLabel').textContent=mode.toUpperCase();
      document.querySelectorAll('.cam-btn').forEach(b=>b.classList.remove('active'));
      document.getElementById(id).classList.add('active');
    };
  });

  // Layer toggles
  const tMap={'togPointCloud':'pointCloud','togMesh':'mesh','togLidar':'lidar','togNavGrid':'navGrid','togPath':'path','togFOV':'fov','togTrail':'trail','togWireframe':'wireframe'};
  Object.entries(tMap).forEach(([id,key])=>{
    const el=document.getElementById(id);
    if(el) el.onchange=()=>{
      T[key]=el.checked;
      if(key==='wireframe') scene.traverse(o=>{if(o.isMesh&&o!==robot)o.material.wireframe=el.checked;});
    };
  });

  // Cinematic / Overlay buttons
  document.getElementById('btnCinematic').onclick=()=>{camMode='cinema';document.getElementById('modeLabel').textContent='CINEMA';};
  document.getElementById('btnOverlay').onclick=()=>{
    const ov=document.getElementById('btnOverlay');
    ov.classList.toggle('active');
    ['minimap','povPanel'].forEach(id=>document.getElementById(id).style.display=ov.classList.contains('active')?'':'none');
  };
}

/* ═══════════════════ RESIZE ════════════════════════════ */
function onResize() {
  const w = window.innerWidth;
  const h = window.innerHeight;
  renderer.setSize(w, h, false);
  renderer.domElement.style.width = w + 'px';
  renderer.domElement.style.height = h + 'px';
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

/* ═══════════════════ UTILITIES ════════════════════════ */
let toastEl;
function toast(msg, type='info') {
  if (!toastEl) {
    toastEl=document.createElement('div');
    toastEl.className='toast-container';
    document.body.appendChild(toastEl);
  }
  const t=document.createElement('div');
  t.className='toast'; t.textContent=(type==='info'?'ℹ ':'⚠ ')+msg;
  toastEl.appendChild(t);
  setTimeout(()=>t.remove(),3100);
}
function addLog(msg, type='') {
  const lb=document.getElementById('slamLog');
  if(!lb)return;
  const e=document.createElement('div');
  e.className='log-entry'+(type?' '+type:'');
  e.textContent=msg;
  lb.appendChild(e);
  lb.scrollTop=lb.scrollHeight;
  if(lb.children.length>40) lb.children[0].remove();
}
