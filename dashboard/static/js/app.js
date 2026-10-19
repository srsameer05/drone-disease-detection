/* ══════════════════════════════════════════
   Disease Drone — Dashboard JS (v0.2)
   ══════════════════════════════════════════ */

const API = '/api';
const CARTO_API_KEY = 'cb1_3uub_1_02c503dacb2fec94d14c7576';
const DISEASE_COLORS = {};
const DISEASE_LABELS = {};
const COLOR_PALETTE = ['#ef4444', '#f59e0b', '#d97706', '#8b5cf6', '#3b82f6', '#ec4899', '#14b8a6', '#f43f5e', '#84cc16', '#a855f7', '#06b6d4', '#eab308', '#f97316', '#10b981', '#3f6212', '#6366f1', '#d946ef', '#f472b6', '#cbd5e1'];

// ── Global State ──
let map;
let drawControl;
let drawnItems;
let detectionMarkers = L.layerGroup();
let sprayZoneLayer = L.layerGroup();
let healthHeatLayer = null;
let scanAreaLayer = L.layerGroup();
let flightPathLayer = L.layerGroup();
let droneMarker = null;
let droneTrail = null;
let droneTrailPoints = [];

let currentTab = 'missions';
let mapMode = 'view';   // 'view' | 'draw' | 'wizard-draw'
let allDetections = []; // cached for filtering
let ws = null;
let wsReconnectTimer = null;
let followingDrone = false;

// Wizard state
const wizard = {
    step: 1,
    polygon: null,
    waypoints: null,
    stats: null,
    missionId: null,
};

// Mission detail state
let currentMissionDetail = null;

// ══════════════════════════════════════════
// INIT
// ══════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
    initMap();
    initTabs();
    initLayerToggles();
    initWebSocket();
    
    loadClasses().then(() => {
        loadAllData();
        setInterval(loadStats, 15000);
        setInterval(pollTelemetry, 1000);
    });

    // Live confidence slider update
    const confSlider = document.getElementById('uploadConf');
    const confValue = document.getElementById('uploadConfValue');
    if (confSlider && confValue) {
        confSlider.addEventListener('input', () => {
            confValue.textContent = parseFloat(confSlider.value).toFixed(2);
        });
    }
});

async function loadClasses() {
    try {
        const classesObj = await fetch(`${API}/classes`).then(r => r.json());
        const classNames = Object.values(classesObj);
        
        let colorIdx = 0;
        classNames.forEach(c => {
            if (c.toLowerCase() === 'healthy') {
                DISEASE_COLORS[c] = '#10b981';
            } else {
                DISEASE_COLORS[c] = COLOR_PALETTE[colorIdx % COLOR_PALETTE.length];
                colorIdx++;
            }
            DISEASE_LABELS[c] = c.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
        });

        const filterSel = document.getElementById('detFilterDisease');
        if (filterSel) {
            classNames.forEach(c => {
                const opt = document.createElement('option');
                opt.value = c;
                opt.textContent = DISEASE_LABELS[c];
                filterSel.appendChild(opt);
            });
        }

        const legendList = document.getElementById('dynamicLegendList');
        if (legendList) {
            const order = Object.keys(DISEASE_LABELS).sort((a,b) => a==='healthy'? 1 : b==='healthy'? -1 : a.localeCompare(b));
            order.forEach(c => {
                const div = document.createElement('div');
                div.className = 'legend-item';
                div.dataset.disease = c;
                div.innerHTML = `
                    <span class="legend-dot" style="background-color: ${DISEASE_COLORS[c]}"></span>
                    ${DISEASE_LABELS[c]}
                    <span class="legend-count" id="legend-count-${c}">0</span>
                `;
                legendList.appendChild(div);
            });
            initLegendFilter();
        }
    } catch (e) {
        console.error('Failed to load classes', e);
    }
}

async function pollTelemetry() {
    try {
        const res = await fetch('/api/telemetry');
        if (!res.ok) return;
        const data = await res.json();
        
        // Update connection modal & header UI
        const statusEl = document.getElementById('dcCurrentStatus');
        const btn = document.getElementById('dcConnectBtn');
        const scoutDot = document.getElementById('scoutDot');
        
        if (data.connected) {
            if (statusEl) {
                statusEl.textContent = 'Connected';
                statusEl.classList.add('success');
            }
            if (btn) {
                btn.textContent = 'Connected';
                btn.disabled = true;
            }
            if (scoutDot) {
                scoutDot.classList.add('connected');
                scoutDot.classList.remove('offline');
            }
        } else {
            if (statusEl) {
                statusEl.textContent = 'Connecting...';
                statusEl.classList.remove('success');
            }
            if (btn) {
                btn.textContent = 'Connecting...';
                btn.disabled = false;
            }
            if (scoutDot) {
                scoutDot.classList.remove('connected');
                scoutDot.classList.add('offline');
            }
        }
    } catch(e) {}
}

// ══════════════════════════════════════════
// MAP
// ══════════════════════════════════════════

function initMap() {
    map = L.map('map', {
        center: [12.9716, 77.5946],
        zoom: 15,
        zoomControl: false,
    });

    L.control.zoom({ position: 'bottomleft' }).addTo(map);

    const tileUrl = CARTO_API_KEY
        ? `https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png?key=${CARTO_API_KEY}`
        : 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png';

    L.tileLayer(tileUrl, {
        attribution: '&copy; OSM &copy; CARTO',
        maxZoom: 20,
    }).addTo(map);

    document.querySelector('.leaflet-tile-pane').style.filter = 'none';

    detectionMarkers.addTo(map);
    sprayZoneLayer.addTo(map);
    scanAreaLayer.addTo(map);
    flightPathLayer.addTo(map);

    drawnItems = new L.FeatureGroup();
    map.addLayer(drawnItems);

    drawControl = new L.Control.Draw({
        draw: {
            polygon: { shapeOptions: { color: '#3b82f6', fillColor: '#3b82f620', weight: 2 }, allowIntersection: false },
            rectangle: { shapeOptions: { color: '#3b82f6', fillColor: '#3b82f620', weight: 2 } },
            circle: false, circlemarker: false, marker: false, polyline: false,
        },
        edit: { featureGroup: drawnItems },
    });

    map.on(L.Draw.Event.CREATED, (e) => {
        // Capture mode BEFORE setMapMode mutates it
        const wasWizardDraw = (mapMode === 'wizard-draw');

        drawnItems.clearLayers();
        // Make the drawn polygon non-interactive so it doesn't block
        // clicks on detection markers underneath.
        if (e.layer.setStyle) {
            e.layer.options.interactive = false;
            if (e.layer._path) e.layer._path.style.pointerEvents = 'none';
        }
        drawnItems.addLayer(e.layer);

        setMapMode('view');

        if (wasWizardDraw) {
            captureWizardArea(e.layer);
        }
        toast('Scan area captured', 'success');
    });
}

function setMapMode(mode) {
    mapMode = mode;
    const label = document.getElementById('mapModeLabel');
    const toolbar = document.getElementById('mapToolbar');
    if (mode === 'draw' || mode === 'wizard-draw') {
        try { map.addControl(drawControl); } catch(e) {}
        label.textContent = mode === 'wizard-draw' ? 'DRAW SCAN AREA — return to wizard when done' : 'DRAW SCAN AREA';
        label.classList.add('visible');
        toolbar.classList.add('hidden');
    } else {
        try { map.removeControl(drawControl); } catch(e) {}
        label.classList.remove('visible');
        toolbar.classList.remove('hidden');
    }
}

function toggleDrawMode() {
    setMapMode(mapMode === 'draw' ? 'view' : 'draw');
}

// ══════════════════════════════════════════
// TABS & LAYER TOGGLES
// ══════════════════════════════════════════

function initTabs() {
    document.querySelectorAll('.panel-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            const target = tab.dataset.tab;
            document.querySelectorAll('.panel-tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.panel-section').forEach(s => s.classList.remove('active'));
            tab.classList.add('active');
            document.getElementById(`section-${target}`).classList.add('active');
            currentTab = target;
            if (target === 'missions') loadMissionHistory();
        });
    });
}

function initLayerToggles() {
    document.getElementById('toggleDetections').addEventListener('change', (e) => {
        if (e.target.checked) detectionMarkers.addTo(map);
        else map.removeLayer(detectionMarkers);
    });
    document.getElementById('toggleZones').addEventListener('change', (e) => {
        if (e.target.checked) sprayZoneLayer.addTo(map);
        else map.removeLayer(sprayZoneLayer);
    });
    document.getElementById('toggleHeatmap').addEventListener('change', (e) => {
        if (!healthHeatLayer) {
            toast('No health data — load demo first', 'info');
            e.target.checked = false;
            return;
        }
        if (e.target.checked) healthHeatLayer.addTo(map);
        else map.removeLayer(healthHeatLayer);
    });
    document.getElementById('toggleFlightPath').addEventListener('change', (e) => {
        if (e.target.checked) flightPathLayer.addTo(map);
        else map.removeLayer(flightPathLayer);
    });
    document.getElementById('toggleTrail').addEventListener('change', (e) => {
        if (droneTrail) {
            if (e.target.checked) droneTrail.addTo(map);
            else map.removeLayer(droneTrail);
        }
    });
}

function initLegendFilter() {
    document.querySelectorAll('.legend-item').forEach(item => {
        item.addEventListener('click', () => {
            const disease = item.dataset.disease;
            if (!disease) return;
            const filterSel = document.getElementById('detFilterDisease');
            filterSel.value = filterSel.value === disease ? '' : disease;
            // Switch to detections tab
            document.querySelector('.panel-tab[data-tab="detections"]').click();
            renderDetections();
        });
    });
}

// ══════════════════════════════════════════
// WEBSOCKET — Live Telemetry
// ══════════════════════════════════════════

function initWebSocket() {
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${proto}//${window.location.host}/ws/telemetry`;

    try {
        ws = new WebSocket(url);
    } catch(e) {
        console.error('WebSocket failed:', e);
        return;
    }

    ws.onopen = () => {
        document.getElementById('wsIndicator').classList.add('connected');
        if (wsReconnectTimer) { clearTimeout(wsReconnectTimer); wsReconnectTimer = null; }
    };

    ws.onmessage = (event) => {
        try {
            const msg = JSON.parse(event.data);
            if (msg.type === 'telemetry') {
                updateTelemetry(msg.data);
            } else if (msg.type === 'mission_complete') {
                toast(`Mission #${msg.mission_id} complete`, 'success');
                loadAllData();
            } else if (msg.type === 'mission_aborted') {
                toast(`Mission #${msg.mission_id} aborted`, 'error');
                loadAllData();
            } else if (msg.type === 'data_cleared') {
                // Server-initiated clear — tear down everything immediately
                tearDownFlightVisuals();
            }
        } catch(e) {
            console.error('WS parse error:', e);
        }
    };

    ws.onclose = () => {
        document.getElementById('wsIndicator').classList.remove('connected');
        // Auto-reconnect
        wsReconnectTimer = setTimeout(initWebSocket, 3000);
    };

    ws.onerror = () => {
        document.getElementById('wsIndicator').classList.remove('connected');
    };
}

// Track the last mission ID we saw so the user can Restart from the stopped HUD
let _lastFlightMissionId = null;
// Preview mode: HUD is showing a mission that the user has selected but not yet started
let _previewMode = false;
let _previewMission = null;

function updateTelemetry(data) {
    const hud = document.getElementById('telemetryHud');
    const liveControls = document.getElementById('hudLiveControls');
    const stoppedControls = document.getElementById('hudStoppedControls');
    const previewControls = document.getElementById('hudPreviewControls');

    // Inactive AND not paused = the flight has ended (stop/abort/complete).
    // Keep the HUD visible in "stopped state" so the user can see the final
    // stats and choose to Restart or Dismiss. Only Clear Data or Dismiss
    // actually removes the marker/trail.
    if (!data.active && !data.paused) {
        // If we're in preview mode (user clicked a mission to inspect/control it),
        // keep the preview HUD visible — don't get overridden by sim's idle state.
        if (_previewMode) return;

        // If we have a mission_id, the flight is in "stopped" state.
        // If mission_id is null too (post-clear/reset), tear down completely.
        if (data.mission_id == null && data.lat == null) {
            hud.classList.remove('visible');
            if (droneMarker) { map.removeLayer(droneMarker); droneMarker = null; }
            if (droneTrail)  { map.removeLayer(droneTrail);  droneTrail  = null; }
            droneTrailPoints = [];
            _lastDroneState = { type: null, heading: 0, innerEl: null };
            _stopDroneAnimation();
            if (data.source === 'simulator') {
                document.getElementById('scoutDot').classList.add('offline');
                document.getElementById('treatDot').classList.add('offline');
            }
            return;
        }

        // Stopped state — keep HUD visible, swap controls
        hud.classList.add('visible');
        liveControls.style.display = 'none';
        previewControls.style.display = 'none';
        stoppedControls.style.display = 'flex';

        // Mode badge
        const modeEl = document.getElementById('hudMode');
        modeEl.classList.remove('stopped', 'aborted', 'complete');
        if (data.aborted) {
            modeEl.textContent = 'ABORTED';
            modeEl.classList.add('aborted');
        } else if (data.progress >= 0.999) {
            modeEl.textContent = 'COMPLETE';
            modeEl.classList.add('complete');
        } else {
            modeEl.textContent = 'STOPPED';
            modeEl.classList.add('stopped');
        }

        // Stop the rAF interpolation but leave the marker visible
        // at its final position for context.
        _stopDroneAnimation();
        return;
    }

    // Active flight — exit preview mode
    _previewMode = false;
    _previewMission = null;

    hud.classList.add('visible');
    liveControls.style.display = 'flex';
    stoppedControls.style.display = 'none';
    previewControls.style.display = 'none';
    document.getElementById('hudMode').classList.remove('stopped', 'aborted', 'complete');

    // Remember the mission so Restart works after stop
    if (data.mission_id) _lastFlightMissionId = data.mission_id;

    // Title and icon based on drone type
    const isTreatment = data.drone_type === 'spray' || data.drone_type === 'treatment';
    document.getElementById('hudTitle').textContent = isTreatment ? 'TREATMENT DRONE' : 'SCOUT DRONE';
    document.getElementById('hudDroneIcon').textContent = isTreatment ? '💧' : '🛰';

    document.getElementById('hudMode').textContent = data.mode || 'AUTO';
    document.getElementById('hudAlt').textContent = (data.alt || 0).toFixed(1);
    document.getElementById('hudSpeed').textContent = (data.groundspeed || 0).toFixed(1);
    document.getElementById('hudHeading').textContent = Math.round(data.heading || 0);
    document.getElementById('hudBattery').textContent = Math.round(data.battery || 0);

    // Compass needle rotation
    document.getElementById('compassNeedle').style.transform =
        `translate(-50%, -100%) rotate(${data.heading || 0}deg)`;

    // Mission stats — elapsed/eta/wpt
    document.getElementById('hudElapsed').textContent = formatTimeShort(data.elapsed_s || 0);
    document.getElementById('hudEta').textContent = data.eta_s ? formatTimeShort(data.eta_s) : '--';
    document.getElementById('hudWpt').textContent =
        `${(data.waypoint_index || 0) + 1}/${data.waypoint_count || 0}`;
    document.getElementById('hudDistNext').textContent =
        (data.distance_to_next_m != null ? data.distance_to_next_m : 0).toFixed(0);
    document.getElementById('hudDistTraveled').textContent =
        (data.distance_traveled_m != null ? data.distance_traveled_m : 0).toFixed(0);
    document.getElementById('hudDistTotal').textContent =
        (data.total_distance_m != null ? data.total_distance_m : 0).toFixed(0);

    // Battery bar
    const batFill = document.getElementById('batteryFill');
    const batPct = data.battery || 0;
    batFill.style.width = batPct + '%';
    batFill.classList.remove('low', 'critical');
    if (batPct < 20) batFill.classList.add('critical');
    else if (batPct < 40) batFill.classList.add('low');

    // Progress — use 1 decimal precision when below 10% so the display
    // visibly increments on long missions at 1× speed (otherwise rounding
    // makes the % appear stuck at 0 for many ticks).
    const pctRaw = (data.progress || 0) * 100;
    const pctText = pctRaw < 10 ? pctRaw.toFixed(1) + '%' : Math.round(pctRaw) + '%';
    document.getElementById('hudProgress').textContent = pctText;
    document.getElementById('progressFill').style.width = pctRaw + '%';

    // Pause/Resume button toggle
    const pauseBtn = document.getElementById('hudPauseBtn');
    const resumeBtn = document.getElementById('hudResumeBtn');
    if (data.paused) {
        pauseBtn.style.display = 'none';
        resumeBtn.style.display = '';
    } else {
        pauseBtn.style.display = '';
        resumeBtn.style.display = 'none';
    }

    // Highlight active speed multiplier button
    const mult = data.speed_multiplier || 1;
    document.querySelectorAll('.hud-speed-btn').forEach(btn => {
        btn.classList.toggle('active', parseFloat(btn.dataset.mult) === mult);
    });

    // Drone position on map (only update if we have coords and not paused-stationary)
    if (data.lat && data.lon) {
        updateDroneMarker(data.lat, data.lon, data.heading, data.drone_type);

        if (data.drone_type === 'scout') {
            document.getElementById('scoutDot').classList.remove('offline');
        } else {
            document.getElementById('treatDot').classList.remove('offline');
        }

        // Periodic stats refresh while flying
        if (Math.random() < 0.05) loadStats();
    }
}

function formatTimeShort(seconds) {
    if (seconds == null || isNaN(seconds)) return '--';
    seconds = Math.max(0, Math.round(seconds));
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;
    if (h > 0) return `${h}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`;
    return `${m}:${String(s).padStart(2,'0')}`;
}

// ══════════════════════════════════════════
// DRONE MARKER & TRAIL
// ══════════════════════════════════════════

// Cache last drone state to avoid pointless DOM writes
let _lastDroneState = { type: null, heading: 0, innerEl: null };

// Inline SVG drone icons (top-down view, 24x24 viewBox).
const DRONE_ICONS = {
    scout: `<svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" width="16" height="16"><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2" fill="white"/></svg>`,
    treatment: `<svg viewBox="0 0 24 24" fill="white" stroke="white" stroke-width="1.5" stroke-linejoin="round" width="16" height="16"><path d="M12 2.5C8 9 6 12 6 15a6 6 0 0 0 12 0c0-3-2-6-6-12.5z"/></svg>`,
};

// ── Smooth interpolation between WS telemetry updates ──
//
// WS pushes telemetry every 500ms. Without interpolation the marker
// "teleports" — especially obvious in lawnmower patterns where short
// row-turn segments compound the jitter.
//
// We use time-based linear interpolation: each WS frame defines a
// (from, to, startTime) triplet, and the rAF loop renders the marker
// at `from + (to - from) * (elapsed/500ms)` clamped to 1.0. This produces
// continuous motion that always reaches the target exactly when the next
// WS frame arrives.
const WS_INTERVAL_MS = 500;
const _droneCurrent  = { lat: null, lon: null, heading: 0 };
const _droneInterpFrom = { lat: 0, lon: 0, heading: 0 };
const _droneInterpTo   = { lat: 0, lon: 0, heading: 0 };
let _droneInterpStart = 0;
let _droneRafId = null;

function shortestAngleDelta(from, to) {
    // Returns the signed angle (in degrees) from `from` to `to` taking
    // the shortest path. Range: (-180, 180].
    let d = ((to - from + 540) % 360) - 180;
    return d;
}

function lerpAngle(from, to, t) {
    return from + shortestAngleDelta(from, to) * t;
}

function updateDroneMarker(lat, lon, heading, type) {
    const isTreatment = type === 'spray' || type === 'treatment';
    const droneClass = isTreatment ? 'drone-marker treatment' : 'drone-marker';
    const icon = isTreatment ? DRONE_ICONS.treatment : DRONE_ICONS.scout;

    if (!droneMarker) {
        const html = `<div class="${droneClass}" style="transform: rotate(${heading || 0}deg);">${icon}</div>`;
        const divIcon = L.divIcon({
            html: html,
            className: 'drone-marker-wrapper',
            iconSize: [32, 32],
            iconAnchor: [16, 16],
        });
        droneMarker = L.marker([lat, lon], {
            icon: divIcon,
            zIndexOffset: 1000,
            interactive: false,
            keyboard: false,
        }).addTo(map);
        
        // Auto-center the map on the drone's first known location
        map.setView([lat, lon], 18);

        // Snap on first frame — no interpolation source yet
        _droneCurrent.lat = lat; _droneCurrent.lon = lon; _droneCurrent.heading = heading || 0;
        _droneInterpFrom.lat = lat; _droneInterpFrom.lon = lon; _droneInterpFrom.heading = heading || 0;
        _droneInterpTo.lat = lat; _droneInterpTo.lon = lon; _droneInterpTo.heading = heading || 0;
        _droneInterpStart = performance.now();
        _lastDroneState = { type: type, heading: heading || 0, innerEl: null };

        if (!_droneRafId) _droneRafId = requestAnimationFrame(_droneAnimateFrame);
    } else {
        // New WS frame — set up the next interpolation segment.
        // From = current displayed position (smooth handoff, no jumps)
        // To   = newly received position
        _droneInterpFrom.lat = _droneCurrent.lat;
        _droneInterpFrom.lon = _droneCurrent.lon;
        _droneInterpFrom.heading = _droneCurrent.heading;

        _droneInterpTo.lat = lat;
        _droneInterpTo.lon = lon;
        _droneInterpTo.heading = heading || 0;

        _droneInterpStart = performance.now();

        // Update icon class if the type changed (rare)
        if (!_lastDroneState.innerEl) {
            const el = droneMarker.getElement();
            if (el) _lastDroneState.innerEl = el.querySelector('.drone-marker');
        }
        if (_lastDroneState.innerEl && _lastDroneState.type !== type) {
            _lastDroneState.innerEl.className = droneClass;
            _lastDroneState.innerEl.innerHTML = icon;
            _lastDroneState.type = type;
        }
    }

    // Trail — only append if position actually moved (deduplicates pause ticks)
    const lastPt = droneTrailPoints[droneTrailPoints.length - 1];
    if (!lastPt || lastPt[0] !== lat || lastPt[1] !== lon) {
        droneTrailPoints.push([lat, lon]);
        if (droneTrailPoints.length > 300) droneTrailPoints.shift();

        if (!droneTrail) {
            droneTrail = L.polyline(droneTrailPoints, {
                color: '#06b6d4', weight: 2, opacity: 0.7, dashArray: '4 4',
                interactive: false,
            });
            if (document.getElementById('toggleTrail').checked) droneTrail.addTo(map);
        } else {
            droneTrail.setLatLngs(droneTrailPoints);
        }
    }
}

// rAF interpolation loop — drives the marker at display refresh rate (~60fps)
// using time-based linear interpolation between consecutive WS frames.
function _droneAnimateFrame() {
    if (!droneMarker) {
        _droneRafId = requestAnimationFrame(_droneAnimateFrame);
        return;
    }

    // t = fraction of WS interval elapsed since last frame, clamped [0, 1]
    const elapsed = performance.now() - _droneInterpStart;
    const t = Math.min(1.0, elapsed / WS_INTERVAL_MS);

    _droneCurrent.lat = _droneInterpFrom.lat + (_droneInterpTo.lat - _droneInterpFrom.lat) * t;
    _droneCurrent.lon = _droneInterpFrom.lon + (_droneInterpTo.lon - _droneInterpFrom.lon) * t;
    _droneCurrent.heading = lerpAngle(_droneInterpFrom.heading, _droneInterpTo.heading, t);

    // Apply to Leaflet (no Leaflet animation — we drive every frame ourselves)
    droneMarker.setLatLng([_droneCurrent.lat, _droneCurrent.lon]);

    if (_lastDroneState.innerEl) {
        _lastDroneState.innerEl.style.transform = `rotate(${_droneCurrent.heading}deg)`;
    }

    // Follow mode — pan the map at the rAF cadence (no panTo animation conflicts)
    if (followingDrone) {
        map.setView([_droneCurrent.lat, _droneCurrent.lon], map.getZoom(), {
            animate: false,
            reset: false,
        });
    }

    _droneRafId = requestAnimationFrame(_droneAnimateFrame);
}

function _stopDroneAnimation() {
    if (_droneRafId) {
        cancelAnimationFrame(_droneRafId);
        _droneRafId = null;
    }
    _droneCurrent.lat = null; _droneCurrent.lon = null; _droneCurrent.heading = 0;
    _droneInterpFrom.lat = 0; _droneInterpFrom.lon = 0; _droneInterpFrom.heading = 0;
    _droneInterpTo.lat = 0; _droneInterpTo.lon = 0; _droneInterpTo.heading = 0;
    _droneInterpStart = 0;
}

function followDrone() {
    followingDrone = !followingDrone;
    const btn = document.getElementById('followBtn');
    btn.classList.toggle('btn-primary', followingDrone);
    if (followingDrone) {
        if (droneMarker) {
            map.flyTo(droneMarker.getLatLng(), 17, { duration: 0.6 });
        }
        toast('Following drone', 'info');
    } else {
        toast('Stopped following', 'info');
    }
}

function clearTrail() {
    droneTrailPoints = [];
    if (droneTrail) {
        droneTrail.setLatLngs([]);
    }
    toast('Trail cleared', 'info');
}

// ══════════════════════════════════════════
// DATA LOADING
// ══════════════════════════════════════════

async function loadAllData() {
    await Promise.all([
        loadStats(),
        loadDetections(),
        loadSprayZones(),
        loadTreatments(),
        loadHealth(),
        loadMissionHistory(),
    ]);
}

async function loadStats() {
    try {
        const stats = await fetch(`${API}/stats`).then(r => r.json());
        animateValue('statMissions', stats.total_missions || 0);
        animateValue('statActive', stats.active_missions || 0);
        animateValue('statDetections', stats.total_detections || 0);
        animateValue('statPending', stats.pending_zones || 0);

        const dist = stats.disease_distribution || {};
        Object.keys(DISEASE_COLORS).forEach(d => {
            const el = document.getElementById(`legend-count-${d}`);
            if (el) el.textContent = dist[d] || 0;
        });

        renderDistChart(dist);

        const healthFill = document.getElementById('healthFill');
        const healthValue = document.getElementById('healthValue');
        if (stats.avg_health != null) {
            const pct = Math.round(stats.avg_health * 100);
            healthFill.style.width = pct + '%';
            healthFill.style.background = pct > 70 ? 'var(--success)' : pct > 40 ? 'var(--warning)' : 'var(--danger)';
            healthValue.textContent = pct + '%';
        }
    } catch(e) {
        console.error('loadStats:', e);
    }
}

function renderDistChart(dist) {
    const container = document.getElementById('distChart');
    const total = Object.values(dist).reduce((a, b) => a + b, 0) || 1;
    const order = Object.keys(DISEASE_LABELS).sort((a,b) => a==='healthy'? 1 : b==='healthy'? -1 : a.localeCompare(b));
    container.innerHTML = order.map(d => {
        const count = dist[d] || 0;
        const pct = (count / total) * 100;
        return `
            <div class="dist-row">
                <div class="dist-label">${DISEASE_LABELS[d] || d}</div>
                <div class="dist-bar">
                    <div class="dist-bar-fill" style="width: ${pct}%; background: ${DISEASE_COLORS[d] || '#64748b'};"></div>
                </div>
                <div class="dist-count">${count}</div>
            </div>
        `;
    }).join('');
}

async function loadDetections() {
    try {
        allDetections = await fetch(`${API}/detections`).then(r => r.json());
        renderDetections();
    } catch(e) {
        console.error('loadDetections:', e);
    }
}

function renderDetections() {
    detectionMarkers.clearLayers();
    const container = document.getElementById('detectionList');
    container.innerHTML = '';

    const search = (document.getElementById('detSearch')?.value || '').toLowerCase();
    const filterDisease = document.getElementById('detFilterDisease')?.value || '';

    let filtered = allDetections;
    if (filterDisease) filtered = filtered.filter(d => d.class_name === filterDisease);
    if (search) filtered = filtered.filter(d =>
        (d.class_name || '').toLowerCase().includes(search) ||
        (`${d.lat || ''} ${d.lon || ''}`).includes(search)
    );

    if (filtered.length === 0) {
        container.innerHTML = `<div class="empty-state"><p>No detections match.</p></div>`;
        return;
    }

    filtered.forEach((det, i) => {
        if (det.lat && det.lon) {
            const color = DISEASE_COLORS[det.class_name] || '#64748b';
            const marker = L.circleMarker([det.lat, det.lon], {
                radius: 6 + det.confidence * 4,
                color: color,
                fillColor: color,
                fillOpacity: 0.6,
                weight: 1.5,
            }).bindPopup(`
                <strong>${DISEASE_LABELS[det.class_name] || det.class_name}</strong><br>
                Confidence: ${(det.confidence * 100).toFixed(1)}%<br>
                <small>${det.lat.toFixed(5)}, ${det.lon.toFixed(5)}</small>
            `);
            detectionMarkers.addLayer(marker);
        }

        const confClass = det.confidence > 0.8 ? 'high' : det.confidence > 0.6 ? 'medium' : 'low';
        const card = document.createElement('div');
        card.className = 'detection-card';
        card.style.animationDelay = `${Math.min(i, 20) * 0.02}s`;
        card.innerHTML = `
            <div class="detection-header">
                <div class="detection-disease">
                    <span class="legend-dot" style="background-color: ${DISEASE_COLORS[det.class_name] || '#64748b'}"></span>
                    ${DISEASE_LABELS[det.class_name] || det.class_name}
                </div>
                <span class="detection-conf ${confClass}">${(det.confidence * 100).toFixed(1)}%</span>
            </div>
            <div class="detection-meta">
                ${det.lat ? `<span>${det.lat.toFixed(4)}, ${det.lon.toFixed(4)}</span>` : ''}
                <span>${formatTime(det.detected_at)}</span>
            </div>
        `;
        if (det.lat && det.lon) {
            card.addEventListener('click', () => {
                map.flyTo([det.lat, det.lon], 17, { duration: 0.8 });
            });
        }
        container.appendChild(card);
    });
}

async function loadSprayZones() {
    try {
        const data = await fetch(`${API}/spray-zones`).then(r => r.json());
        sprayZoneLayer.clearLayers();
        const container = document.getElementById('zoneList');
        container.innerHTML = '';

        if (data.length === 0) {
            container.innerHTML = `<div class="empty-state"><p>No spray zones generated.</p></div>`;
            return;
        }

        data.forEach((zone) => {
            const color = DISEASE_COLORS[zone.disease_type] || '#64748b';

            if (zone.geometry) {
                const geo = typeof zone.geometry === 'string' ? JSON.parse(zone.geometry) : zone.geometry;
                const polygon = L.polygon(geo, {
                    color: color, fillColor: color, fillOpacity: 0.2, weight: 2,
                    dashArray: zone.status === 'pending' ? '6 4' : null,
                }).bindPopup(`
                    <strong>Spray Zone #${zone.id}</strong><br>
                    Disease: ${DISEASE_LABELS[zone.disease_type] || zone.disease_type}<br>
                    Severity: ${(zone.severity * 100).toFixed(0)}%<br>
                    Status: ${zone.status}
                `);
                sprayZoneLayer.addLayer(polygon);
            }

            const card = document.createElement('div');
            card.className = 'zone-card';
            card.innerHTML = `
                <div class="zone-header">
                    <div class="zone-disease" style="color: ${color}">
                        ${DISEASE_LABELS[zone.disease_type] || zone.disease_type || 'Unknown'}
                    </div>
                    <span class="badge badge-${zone.status}">${zone.status}</span>
                </div>
                <div class="detection-meta">
                    <span>Severity: ${(zone.severity * 100).toFixed(0)}%</span>
                    <span>${formatTime(zone.created_at)}</span>
                </div>
                ${zone.status === 'pending' ? `
                <div class="zone-actions">
                    <button class="btn btn-success btn-sm" onclick="approveZone(${zone.id})">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
                        Approve
                    </button>
                    <button class="btn btn-danger btn-sm" onclick="rejectZone(${zone.id})">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
                        Reject
                    </button>
                </div>` : ''}
            `;
            if (zone.center_lat && zone.center_lon) {
                card.querySelector('.zone-header').style.cursor = 'pointer';
                card.querySelector('.zone-header').addEventListener('click', () => {
                    map.flyTo([zone.center_lat, zone.center_lon], 17, { duration: 0.8 });
                });
            }
            container.appendChild(card);
        });
    } catch(e) {
        console.error('loadSprayZones:', e);
    }
}

async function loadTreatments() {
    try {
        const data = await fetch(`${API}/treatments`).then(r => r.json());
        const container = document.getElementById('treatmentList');
        container.innerHTML = '';

        if (data.length === 0) {
            container.innerHTML = `<div class="empty-state"><p>No treatments logged.</p></div>`;
            return;
        }

        data.forEach(t => {
            const card = document.createElement('div');
            card.className = 'treatment-card';
            card.innerHTML = `
                <div class="treatment-header">
                    <span class="treatment-chemical">${t.chemical || 'Unknown'}</span>
                    <span class="badge badge-treated">${t.disease_type || 'treated'}</span>
                </div>
                <div class="treatment-meta">
                    Duration: ${t.spray_duration ? t.spray_duration.toFixed(1) + 's' : 'N/A'}
                    &nbsp;|&nbsp; ${formatTime(t.treated_at)}
                    &nbsp;|&nbsp; ${t.lat.toFixed(4)}, ${t.lon.toFixed(4)}
                </div>
            `;
            container.appendChild(card);
        });
    } catch(e) {
        console.error('loadTreatments:', e);
    }
}

async function loadHealth() {
    try {
        const data = await fetch(`${API}/health`).then(r => r.json());
        if (data.length === 0) return;

        if (healthHeatLayer) map.removeLayer(healthHeatLayer);

        const points = data.map(p => [p.lat, p.lon, 1 - p.health_score]);

        healthHeatLayer = L.heatLayer(points, {
            radius: 30, blur: 20, maxZoom: 18, max: 1.0,
            gradient: { 0.0: '#10b981', 0.3: '#84cc16', 0.5: '#f59e0b', 0.7: '#ef4444', 1.0: '#dc2626' },
        });

        if (document.getElementById('toggleHeatmap').checked) {
            healthHeatLayer.addTo(map);
        }
    } catch(e) {
        console.error('loadHealth:', e);
    }
}

// ══════════════════════════════════════════
// MISSION HISTORY
// ══════════════════════════════════════════

async function loadMissionHistory() {
    try {
        const type = document.getElementById('missionFilterType').value;
        const status = document.getElementById('missionFilterStatus').value;
        let url = `${API}/missions/summary`;
        const params = [];
        if (type) params.push(`mission_type=${type}`);
        if (status) params.push(`status=${status}`);
        if (params.length) url += '?' + params.join('&');

        const data = await fetch(url).then(r => r.json());
        const container = document.getElementById('missionList');
        container.innerHTML = '';

        if (data.length === 0) {
            container.innerHTML = `<div class="empty-state">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                    <path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/>
                </svg>
                <p>No missions match.<br>Create one to start.</p>
            </div>`;
            return;
        }

        data.forEach((m, i) => {
            const isSpray = m.type === 'spray';
            const iconHtml = isSpray
                ? `<svg viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" stroke-width="1" stroke-linejoin="round"><path d="M12 3c-2.5 4-4 6.5-4 9a4 4 0 0 0 8 0c0-2.5-1.5-5-4-9z"/></svg>`
                : `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="3" fill="currentColor"/></svg>`;

            const card = document.createElement('div');
            card.className = 'mission-card';
            card.style.animationDelay = `${Math.min(i, 20) * 0.03}s`;
            card.innerHTML = `
                <div class="mission-card-header">
                    <div class="mission-card-id">
                        <div class="mission-icon ${isSpray ? 'spray' : ''}">${iconHtml}</div>
                        Mission #${m.id} · ${m.type}
                    </div>
                    <span class="badge badge-${m.status}">${m.status.replace('_', ' ')}</span>
                </div>
                <div class="mission-card-meta">
                    <span class="pill">${m.detection_count || 0} dets</span>
                    <span class="pill">${m.zone_count || 0} zones</span>
                    <span class="pill">${m.treatment_count || 0} sprays</span>
                    <span style="margin-left:auto;">${formatTime(m.created_at)}</span>
                </div>
            `;
            card.addEventListener('click', () => openMissionDetail(m.id));
            container.appendChild(card);
        });
    } catch(e) {
        console.error('loadMissionHistory:', e);
    }
}

// ══════════════════════════════════════════
// MISSION WIZARD
// ══════════════════════════════════════════

function openWizard() {
    wizard.step = 1;
    wizard.polygon = null;
    wizard.waypoints = null;
    wizard.stats = null;
    wizard.missionId = null;
    document.getElementById('wizardAreaStatus').textContent = 'No area drawn yet.';
    document.getElementById('wizardAreaStatus').classList.remove('success');
    document.getElementById('wizardModal').classList.add('visible');
    updateWizardStep();
}

function closeWizard() {
    document.getElementById('wizardModal').classList.remove('visible');
    setMapMode('view');
}

function updateWizardStep() {
    document.querySelectorAll('.wizard-step').forEach(s => {
        const step = parseInt(s.dataset.step);
        s.classList.toggle('active', step === wizard.step);
        s.classList.toggle('done', step < wizard.step);
    });
    document.querySelectorAll('.wizard-pane').forEach(p => {
        p.classList.toggle('active', parseInt(p.dataset.pane) === wizard.step);
    });
    document.getElementById('wzPrevBtn').style.visibility = wizard.step > 1 ? 'visible' : 'hidden';
    document.getElementById('wzNextBtn').style.display = wizard.step < 4 ? '' : 'none';
}

function wizardPrev() {
    if (wizard.step > 1) {
        wizard.step--;
        updateWizardStep();
    }
}

async function wizardNext() {
    if (wizard.step === 1 && !wizard.polygon) {
        toast('Draw an area first', 'error');
        return;
    }
    if (wizard.step === 2) {
        // Generate preview
        await wizardGeneratePreview();
    }
    wizard.step++;
    updateWizardStep();
}

function wizardStartDraw() {
    // Hide wizard temporarily, switch to draw mode
    document.getElementById('wizardModal').classList.remove('visible');
    setMapMode('wizard-draw');
    drawnItems.clearLayers();
    // Auto-trigger polygon draw
    setTimeout(() => {
        const polygonBtn = document.querySelector('.leaflet-draw-draw-polygon');
        if (polygonBtn) polygonBtn.click();
    }, 200);
}

function captureWizardArea(layer) {
    const latlngs = layer.getLatLngs()[0];
    wizard.polygon = latlngs.map(ll => [ll.lat, ll.lng]);
    document.getElementById('wizardAreaStatus').textContent =
        `Area captured · ${wizard.polygon.length} vertices`;
    document.getElementById('wizardAreaStatus').classList.add('success');
    // Reopen wizard
    setTimeout(() => {
        document.getElementById('wizardModal').classList.add('visible');
    }, 300);
}

async function wizardGeneratePreview() {
    const altitude = parseFloat(document.getElementById('wzAltitude').value);
    const overlap = parseFloat(document.getElementById('wzOverlap').value) / 100;
    const speed = parseFloat(document.getElementById('wzSpeed').value);
    const hfov = parseFloat(document.getElementById('wzHfov').value);

    try {
        const res = await fetch(`${API}/plan/scan`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                polygon: wizard.polygon,
                altitude, overlap,
                flight_speed: speed,
                camera_hfov_deg: hfov,
                save: false,
            }),
        });

        if (!res.ok) {
            const err = await res.json();
            toast('Plan failed: ' + (err.detail || res.statusText), 'error');
            return;
        }

        const data = await res.json();
        wizard.waypoints = data.waypoints;
        wizard.stats = data.stats;

        // Render preview stats
        const statsHtml = `
            <div class="preview-stat">
                <div class="preview-stat-label">Waypoints</div>
                <div class="preview-stat-value">${data.waypoint_count}</div>
            </div>
            <div class="preview-stat">
                <div class="preview-stat-label">Distance</div>
                <div class="preview-stat-value">${data.stats.total_distance_m}<span class="preview-stat-unit">m</span></div>
            </div>
            <div class="preview-stat">
                <div class="preview-stat-label">Duration</div>
                <div class="preview-stat-value">${data.stats.estimated_duration_str}</div>
            </div>
            <div class="preview-stat">
                <div class="preview-stat-label">Est. Images</div>
                <div class="preview-stat-value">${data.stats.estimated_images}</div>
            </div>
            <div class="preview-stat">
                <div class="preview-stat-label">Altitude</div>
                <div class="preview-stat-value">${data.stats.altitude}<span class="preview-stat-unit">m</span></div>
            </div>
            <div class="preview-stat">
                <div class="preview-stat-label">Footprint</div>
                <div class="preview-stat-value">${data.stats.ground_footprint_m[0]}<span class="preview-stat-unit">×${data.stats.ground_footprint_m[1]} m</span></div>
            </div>
        `;
        document.getElementById('previewStats').innerHTML = statsHtml;

        // Draw flight path on map
        renderFlightPath(data.waypoints);
    } catch(e) {
        console.error(e);
        toast('Plan request failed', 'error');
    }
}

function renderFlightPath(waypoints) {
    flightPathLayer.clearLayers();
    const navWps = waypoints.filter(wp => wp.command === 16);   // NAV_WAYPOINT
    const path = navWps.map(wp => [wp.x, wp.y]);
    if (path.length < 2) return;

    // Non-interactive so it doesn't block clicks on detections beneath
    const polyline = L.polyline(path, {
        color: '#3b82f6', weight: 3, opacity: 0.85,
        className: 'flight-path',
        interactive: false,
    });
    flightPathLayer.addLayer(polyline);

    // Start/end markers (still interactive — they have popups)
    const start = L.circleMarker(path[0], {
        radius: 6, color: '#10b981', fillColor: '#10b981', fillOpacity: 1, weight: 2,
    }).bindPopup('<strong>Start</strong>');
    const end = L.circleMarker(path[path.length - 1], {
        radius: 6, color: '#ef4444', fillColor: '#ef4444', fillOpacity: 1, weight: 2,
    }).bindPopup('<strong>End</strong>');
    flightPathLayer.addLayer(start);
    flightPathLayer.addLayer(end);

    map.fitBounds(L.latLngBounds(path), { padding: [60, 60] });
}

async function wizardLaunch(mode) {
    const altitude = parseFloat(document.getElementById('wzAltitude').value);
    const overlap = parseFloat(document.getElementById('wzOverlap').value) / 100;
    const speed = parseFloat(document.getElementById('wzSpeed').value);
    const hfov = parseFloat(document.getElementById('wzHfov').value);

    try {
        // Save the mission
        const res = await fetch(`${API}/plan/scan`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                polygon: wizard.polygon,
                altitude, overlap,
                flight_speed: speed,
                camera_hfov_deg: hfov,
                save: true,
                notes: `Created via wizard @ ${new Date().toLocaleString()}`,
            }),
        });
        const data = await res.json();
        wizard.missionId = data.mission_id;

        toast(`Mission #${data.mission_id} created`, 'success');

        if (mode === 'simulate') {
            // Trigger simulation
            await fetch(`${API}/missions/${data.mission_id}/simulate`, { method: 'POST' });
            toast('Simulation started — watch the HUD', 'success');
            followingDrone = true;
        }

        closeWizard();
        loadStats();
        loadMissionHistory();
    } catch(e) {
        console.error(e);
        toast('Launch failed', 'error');
    }
}

// ══════════════════════════════════════════
// MISSION DETAIL MODAL
// ══════════════════════════════════════════

async function openMissionDetail(missionId) {
    document.getElementById('missionModal').classList.add('visible');
    document.getElementById('mdTitle').textContent = `Mission #${missionId}`;
    document.getElementById('mdBody').innerHTML = '<div class="empty-state"><div class="spinner"></div><p>Loading...</p></div>';

    try {
        const m = await fetch(`${API}/missions/${missionId}/full`).then(r => r.json());
        currentMissionDetail = m;

        document.getElementById('mdTitle').textContent = `Mission #${m.id} · ${m.type}`;

        const sumDetections = (m.detections || []).length;
        const sumZones = (m.spray_zones || []).length;
        const sumTreats = (m.treatments || []).length;
        const wps = m.waypoints ? JSON.parse(m.waypoints) : [];

        const html = `
            <div class="detail-grid">
                <div class="detail-stat">
                    <div class="detail-stat-value">${m.status.replace('_', ' ')}</div>
                    <div class="detail-stat-label">Status</div>
                </div>
                <div class="detail-stat">
                    <div class="detail-stat-value">${wps.length}</div>
                    <div class="detail-stat-label">Waypoints</div>
                </div>
                <div class="detail-stat">
                    <div class="detail-stat-value">${sumDetections}</div>
                    <div class="detail-stat-label">Detections</div>
                </div>
                <div class="detail-stat">
                    <div class="detail-stat-value">${sumZones}</div>
                    <div class="detail-stat-label">Spray Zones</div>
                </div>
                <div class="detail-stat">
                    <div class="detail-stat-value">${sumTreats}</div>
                    <div class="detail-stat-label">Treatments</div>
                </div>
            </div>

            <div class="detail-section">
                <div class="detail-section-title">Timeline</div>
                <div class="detail-list">
                    <div class="detail-list-item"><span>Created</span><span>${formatTime(m.created_at)}</span></div>
                    ${m.started_at ? `<div class="detail-list-item"><span>Started</span><span>${formatTime(m.started_at)}</span></div>` : ''}
                    ${m.completed_at ? `<div class="detail-list-item"><span>Completed</span><span>${formatTime(m.completed_at)}</span></div>` : ''}
                </div>
            </div>

            ${m.notes ? `
            <div class="detail-section">
                <div class="detail-section-title">Notes</div>
                <p style="font-size:13px;color:var(--text-secondary);">${m.notes}</p>
            </div>` : ''}

            ${sumDetections > 0 ? `
            <div class="detail-section">
                <div class="detail-section-title">Detection Breakdown</div>
                <div class="detail-list">
                    ${detectionBreakdownHtml(m.detections)}
                </div>
            </div>` : ''}

            ${sumZones > 0 ? `
            <div class="detail-section">
                <div class="detail-section-title">Spray Zones</div>
                <div class="detail-list">
                    ${m.spray_zones.map(z => `
                        <div class="detail-list-item">
                            <span>
                                <span class="legend-dot ${z.disease_type}"></span>
                                ${DISEASE_LABELS[z.disease_type] || z.disease_type}
                            </span>
                            <span>
                                <span class="badge badge-${z.status}">${z.status}</span>
                                ${(z.severity * 100).toFixed(0)}%
                            </span>
                        </div>
                    `).join('')}
                </div>
            </div>` : ''}
        `;

        document.getElementById('mdBody').innerHTML = html;

        // Show flight path on map if waypoints exist
        if (wps.length > 0) {
            renderFlightPath(wps);
            // Activate the HUD as a control panel for this mission so the
            // user gets immediate access to Start / Pause / Stop / Restart.
            activateControlPanel(m);
        }

        // Highlight scan area
        if (m.scan_area) {
            try {
                const area = JSON.parse(m.scan_area);
                scanAreaLayer.clearLayers();
                const poly = L.polygon(area, {
                    color: '#06b6d4', fillColor: '#06b6d420', weight: 2, dashArray: '4 4',
                });
                scanAreaLayer.addLayer(poly);
            } catch(e) {}
        }

        // Toggle simulate button
        document.getElementById('mdSimBtn').style.display = wps.length > 0 ? '' : 'none';
    } catch(e) {
        console.error(e);
        document.getElementById('mdBody').innerHTML = '<div class="empty-state"><p>Failed to load mission.</p></div>';
    }
}

function detectionBreakdownHtml(detections) {
    const counts = {};
    detections.forEach(d => { counts[d.class_name] = (counts[d.class_name] || 0) + 1; });
    return Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([d, c]) => `
        <div class="detail-list-item">
            <span><span class="legend-dot ${d}"></span> ${DISEASE_LABELS[d] || d}</span>
            <span>${c}</span>
        </div>
    `).join('');
}

function closeMissionDetail() {
    document.getElementById('missionModal').classList.remove('visible');
    currentMissionDetail = null;
}

async function simulateCurrentMission() {
    if (!currentMissionDetail) return;
    try {
        await fetch(`${API}/missions/${currentMissionDetail.id}/simulate`, { method: 'POST' });
        toast(`Simulating mission #${currentMissionDetail.id}`, 'success');
        followingDrone = true;
        closeMissionDetail();
    } catch(e) {
        toast('Simulation failed', 'error');
    }
}

function exportMissionPlan() {
    if (!currentMissionDetail || !currentMissionDetail.waypoints) {
        toast('No waypoints to export', 'error');
        return;
    }
    const wps = JSON.parse(currentMissionDetail.waypoints);

    // Build a minimal QGC plan structure
    const plan = {
        fileType: 'Plan',
        version: 1,
        groundStation: 'DiseaseDrone',
        mission: {
            cruiseSpeed: 2,
            hoverSpeed: 1,
            items: wps.map((wp, i) => ({
                autoContinue: true,
                command: wp.command,
                doJumpId: i + 1,
                frame: wp.frame,
                params: [wp.param1, wp.param2, wp.param3, wp.param4, wp.x, wp.y, wp.z],
                type: 'SimpleItem',
            })),
            plannedHomePosition: { lat: wps[0].x, lon: wps[0].y, alt: 0 },
        },
    };

    const blob = new Blob([JSON.stringify(plan, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `mission_${currentMissionDetail.id}.plan`;
    a.click();
    URL.revokeObjectURL(url);
    toast('QGC plan downloaded', 'success');
}

// ══════════════════════════════════════════
// ACTIONS
// ══════════════════════════════════════════

async function planSpray() {
    try {
        const res = await fetch(`${API}/plan/spray`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ save: true, altitude: 3.0, hover_time: 5.0 }),
        });
        if (!res.ok) {
            const err = await res.json();
            toast(err.detail || 'No approved zones', 'error');
            return;
        }
        const data = await res.json();
        toast(`Spray mission #${data.mission_id} planned (${data.stats.zone_count} zones, ${data.stats.estimated_duration_str})`, 'success');
        renderFlightPath(data.waypoints);
        loadStats();
        loadMissionHistory();
    } catch(e) {
        console.error(e);
        toast('Spray planning failed', 'error');
    }
}

async function approveZone(zoneId) {
    await fetch(`${API}/spray-zones/${zoneId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status: 'approved' }),
    });
    toast(`Zone #${zoneId} approved`, 'success');
    loadSprayZones();
    loadStats();
}

async function rejectZone(zoneId) {
    await fetch(`${API}/spray-zones/${zoneId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status: 'rejected' }),
    });
    toast(`Zone #${zoneId} rejected`, 'info');
    loadSprayZones();
    loadStats();
}

async function seedDemoData() {
    toast('Running ML inference on test images...', 'info');
    try {
        const res = await fetch(`${API}/demo/detect`, { method: 'POST' });
        if (!res.ok) {
            const err = await res.json();
            toast(`Demo failed: ${err.detail || res.statusText}`, 'error');
            return;
        }
        const data = await res.json();
        toast(`ML Demo: ${data.total_detections} detections from ${data.images_processed} images, ${data.spray_zones_created} spray zones`, 'success');
        await loadAllData();
        setTimeout(() => map.flyTo([12.9716, 77.5946], 15, { duration: 1.2 }), 300);
    } catch(e) {
        console.error('seedDemoData:', e);
        toast('ML inference failed — falling back to random demo', 'error');
        // Fallback to random demo data
        await fetch(`${API}/demo/seed`, { method: 'POST' });
        toast('Random demo data loaded', 'success');
        await loadAllData();
        setTimeout(() => map.flyTo([12.9716, 77.5946], 15, { duration: 1.2 }), 300);
    }
}

async function seedRandomDemo() {
    toast('Seeding random demo data...', 'info');
    await fetch(`${API}/demo/seed`, { method: 'POST' });
    toast('Demo data loaded', 'success');
    await loadAllData();
    setTimeout(() => map.flyTo([12.9716, 77.5946], 15, { duration: 1.2 }), 300);
}

function tearDownFlightVisuals() {
    // Remove every flight-related element from the map and HUD
    if (droneMarker) { map.removeLayer(droneMarker); droneMarker = null; }
    if (droneTrail)  { map.removeLayer(droneTrail);  droneTrail  = null; }
    droneTrailPoints = [];
    _lastDroneState = { type: null, heading: 0, innerEl: null };
    _stopDroneAnimation();
    _previewMode = false;
    _previewMission = null;
    document.getElementById('telemetryHud').classList.remove('visible');
    document.getElementById('scoutDot').classList.add('offline');
    document.getElementById('treatDot').classList.add('offline');
    flightPathLayer.clearLayers();
    document.getElementById('hudMode').textContent = 'STANDBY';
}

async function clearAllData() {
    if (!confirm('Clear all data? This cannot be undone.')) return;

    // Stop simulator first so it can't push new telemetry mid-clear
    try { await fetch(`${API}/simulator/stop`, { method: 'POST' }); } catch(e) {}

    // Now clear data
    await fetch(`${API}/demo/clear`, { method: 'POST' });

    // Tear down all visuals
    detectionMarkers.clearLayers();
    sprayZoneLayer.clearLayers();
    scanAreaLayer.clearLayers();
    if (healthHeatLayer) { map.removeLayer(healthHeatLayer); healthHeatLayer = null; }
    drawnItems.clearLayers();
    tearDownFlightVisuals();
    followingDrone = false;
    document.getElementById('followBtn').classList.remove('btn-primary');

    toast('All data cleared', 'info');
    await loadAllData();
}

// ══════════════════════════════════════════
// FLIGHT SIMULATION CONTROLS
// ══════════════════════════════════════════

async function simPause() {
    try {
        await fetch(`${API}/simulator/pause`, { method: 'POST' });
        toast('Simulation paused', 'info');
    } catch(e) { toast('Pause failed', 'error'); }
}

async function simResume() {
    try {
        await fetch(`${API}/simulator/resume`, { method: 'POST' });
        toast('Simulation resumed', 'info');
    } catch(e) { toast('Resume failed', 'error'); }
}

async function simStop() {
    if (!confirm('Stop the current flight?')) return;
    try {
        await fetch(`${API}/simulator/stop`, { method: 'POST' });
        toast('Flight stopped', 'info');
    } catch(e) { toast('Stop failed', 'error'); }
}

async function simAbort() {
    if (!confirm('Abort the current mission? It will be marked as aborted.')) return;
    try {
        await fetch(`${API}/simulator/abort`, { method: 'POST' });
        toast('Mission aborted', 'error');
    } catch(e) { toast('Abort failed', 'error'); }
}

async function simSpeed(multiplier) {
    try {
        await fetch(`${API}/simulator/speed`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ multiplier }),
        });
        toast(`Speed: ${multiplier}×`, 'info');
    } catch(e) { toast('Speed change failed', 'error'); }
}

async function simRestart() {
    if (!_lastFlightMissionId) {
        toast('No mission to restart', 'error');
        return;
    }
    try {
        // Clear the trail/marker first so the restart looks clean
        if (droneTrail) { map.removeLayer(droneTrail); droneTrail = null; }
        droneTrailPoints = [];
        if (droneMarker) { map.removeLayer(droneMarker); droneMarker = null; }
        _lastDroneState = { type: null, heading: 0, innerEl: null };
        _stopDroneAnimation();

        await fetch(`${API}/missions/${_lastFlightMissionId}/simulate`, { method: 'POST' });
        toast(`Restarting mission #${_lastFlightMissionId}`, 'success');
    } catch(e) {
        toast('Restart failed', 'error');
    }
}

function dismissHud() {
    document.getElementById('telemetryHud').classList.remove('visible');
    if (droneMarker) { map.removeLayer(droneMarker); droneMarker = null; }
    if (droneTrail)  { map.removeLayer(droneTrail);  droneTrail  = null; }
    droneTrailPoints = [];
    _lastDroneState = { type: null, heading: 0, innerEl: null };
    _stopDroneAnimation();
    _previewMode = false;
    _previewMission = null;
    document.getElementById('scoutDot').classList.add('offline');
    document.getElementById('treatDot').classList.add('offline');
}

// ══════════════════════════════════════════
// CONTROL PANEL — show HUD for selected mission
// ══════════════════════════════════════════

/**
 * Activate the control panel (HUD) for a selected mission.
 *
 * If the simulator is currently running this same mission, the HUD shows the
 * live state automatically (via the next WS update). Otherwise, the HUD enters
 * "preview" mode showing the mission's static info with a Start button.
 */
function activateControlPanel(mission) {
    if (!mission) return;
    _lastFlightMissionId = mission.id;

    const wps = mission.waypoints
        ? (typeof mission.waypoints === 'string' ? JSON.parse(mission.waypoints) : mission.waypoints)
        : [];

    if (wps.length === 0) {
        toast('Mission has no waypoints', 'error');
        return;
    }

    _previewMode = true;
    _previewMission = mission;

    // Show HUD in preview state
    const hud = document.getElementById('telemetryHud');
    hud.classList.add('visible');
    document.getElementById('hudLiveControls').style.display = 'none';
    document.getElementById('hudStoppedControls').style.display = 'none';
    document.getElementById('hudPreviewControls').style.display = 'flex';

    // Title and badge
    const isSpray = mission.type === 'spray';
    document.getElementById('hudTitle').textContent =
        `MISSION #${mission.id} · ${isSpray ? 'TREATMENT' : 'SCOUT'}`;
    document.getElementById('hudDroneIcon').textContent = isSpray ? '💧' : '🛰';

    const modeEl = document.getElementById('hudMode');
    modeEl.classList.remove('stopped', 'aborted', 'complete');
    modeEl.textContent = 'READY';

    // Compute static info from waypoints
    const navWps = wps.filter(w => w.command === 16);
    let totalDist = 0;
    for (let i = 0; i < navWps.length - 1; i++) {
        totalDist += _haversineMeters(
            navWps[i].x, navWps[i].y,
            navWps[i + 1].x, navWps[i + 1].y,
        );
    }
    const altitude = navWps.length > 0 ? navWps[0].z : 0;

    // Reset HUD live values to ready state
    document.getElementById('hudAlt').textContent = altitude.toFixed(1);
    document.getElementById('hudSpeed').textContent = '0.0';
    document.getElementById('hudHeading').textContent = '0';
    document.getElementById('hudBattery').textContent = '100';
    const batFill = document.getElementById('batteryFill');
    batFill.style.width = '100%';
    batFill.classList.remove('low', 'critical');
    document.getElementById('hudElapsed').textContent = '0:00';
    document.getElementById('hudEta').textContent = formatTimeShort(totalDist / 2.0);
    document.getElementById('hudWpt').textContent = `0/${wps.length}`;
    document.getElementById('hudDistNext').textContent = '0';
    document.getElementById('hudDistTraveled').textContent = '0';
    document.getElementById('hudDistTotal').textContent = totalDist.toFixed(0);
    document.getElementById('hudProgress').textContent = '0%';
    document.getElementById('progressFill').style.width = '0%';
    document.getElementById('compassNeedle').style.transform = 'translate(-50%, -100%) rotate(0deg)';

    // Render the flight path on the map for context
    renderFlightPath(wps);
}

function _haversineMeters(lat1, lon1, lat2, lon2) {
    const R = 6371000;
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLon = (lon2 - lon1) * Math.PI / 180;
    const a = Math.sin(dLat / 2) ** 2 +
              Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
              Math.sin(dLon / 2) ** 2;
    return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

async function startSelectedMission() {
    if (!_lastFlightMissionId) {
        toast('No mission selected', 'error');
        return;
    }
    try {
        await fetch(`${API}/missions/${_lastFlightMissionId}/simulate`, { method: 'POST' });
        toast(`Starting mission #${_lastFlightMissionId}`, 'success');
        // _previewMode will flip false on the next WS frame (active=true)
        followingDrone = true;
        document.getElementById('followBtn').classList.add('btn-primary');
    } catch(e) {
        toast('Start failed', 'error');
    }
}

function centerMap() {
    if (detectionMarkers.getLayers().length > 0) {
        map.fitBounds(detectionMarkers.getBounds(), { padding: [40, 40], maxZoom: 16 });
    } else if (drawnItems.getLayers().length > 0) {
        map.fitBounds(drawnItems.getBounds(), { padding: [60, 60] });
    } else {
        map.flyTo([12.9716, 77.5946], 15, { duration: 0.6 });
    }
}

// ══════════════════════════════════════════
// UTILITIES
// ══════════════════════════════════════════

function formatTime(ts) {
    if (!ts) return '';
    const d = new Date(ts + 'Z');
    return d.toLocaleString(undefined, {
        month: 'short', day: 'numeric',
        hour: '2-digit', minute: '2-digit',
    });
}

function animateValue(elId, target) {
    const el = document.getElementById(elId);
    if (!el) return;
    const current = parseInt(el.textContent) || 0;
    if (current === target) return;
    const diff = target - current;
    const steps = 20;
    const stepVal = diff / steps;
    let frame = 0;
    const anim = setInterval(() => {
        frame++;
        el.textContent = Math.round(current + stepVal * frame);
        if (frame >= steps) {
            el.textContent = target;
            clearInterval(anim);
        }
    }, 25);
}

function toast(message, type = 'info') {
    const container = document.getElementById('toasts');
    const el = document.createElement('div');
    el.className = `toast ${type}`;
    el.innerHTML = message;
    container.appendChild(el);
    setTimeout(() => el.remove(), 3000);
}

// ══════════════════════════════════════════
// DRONE CONNECT MODAL
// ══════════════════════════════════════════

let currentDroneType = 'scout';

async function openDroneConnect(droneType) {
    currentDroneType = droneType;
    document.getElementById('dcTitle').textContent =
        `Connect ${droneType.charAt(0).toUpperCase() + droneType.slice(1)} Drone`;
    document.getElementById('droneConnectModal').classList.add('visible');
    await refreshDroneStatus();
}

function closeDroneConnect() {
    document.getElementById('droneConnectModal').classList.remove('visible');
}

async function refreshDroneStatus() {
    // Status is now handled entirely by pollTelemetry every 1 second
}

async function droneConnect() {
    const conn = document.getElementById('dcConn').value.trim();
    if (!conn) {
        toast('Enter a connection string', 'error');
        return;
    }
    const btn = document.getElementById('dcConnectBtn');
    btn.disabled = true;
    btn.textContent = 'Connecting...';
    
    try {
        await fetch(`${API}/drone/connect`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ drone_type: currentDroneType, connection: conn }),
        });
    } catch(e) {
        toast('API Error', 'error');
        btn.disabled = false;
        btn.textContent = 'Connect';
    }
}

async function droneDisconnect() {
    try {
        await fetch(`${API}/drone/disconnect/${currentDroneType}`, { method: 'POST' });
        toast(`${currentDroneType} drone disconnected`, 'info');
    } catch(e) {}
}

// Refresh drone status on page load
setTimeout(refreshDroneStatus, 500);

// Allow ESC to close modals
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeWizard();
        closeMissionDetail();
        closeDroneConnect();
        closeUploadModal();
        closeStreamModal();
    }
});

// ════════════════════════════════════════════
// IMAGE UPLOAD & DETECTION
// ════════════════════════════════════════════

function openUploadModal() {
    document.getElementById('uploadModal').classList.add('visible');
    document.getElementById('uploadResults').innerHTML = '';
    document.getElementById('uploadFileInput').value = '';
    document.getElementById('uploadPreview').innerHTML = '';
    document.getElementById('uploadBtn').disabled = true;
}

function closeUploadModal() {
    document.getElementById('uploadModal').classList.remove('visible');
}

function handleUploadPreview() {
    const input = document.getElementById('uploadFileInput');
    const preview = document.getElementById('uploadPreview');
    const btn = document.getElementById('uploadBtn');
    preview.innerHTML = '';

    if (input.files.length === 0) {
        btn.disabled = true;
        return;
    }

    btn.disabled = false;
    const count = Math.min(input.files.length, 6);
    for (let i = 0; i < count; i++) {
        const file = input.files[i];
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = document.createElement('img');
            img.src = e.target.result;
            img.className = 'upload-thumb';
            img.title = file.name;
            preview.appendChild(img);
        };
        reader.readAsDataURL(file);
    }
    if (input.files.length > 6) {
        const more = document.createElement('div');
        more.className = 'upload-thumb-more';
        more.textContent = `+${input.files.length - 6} more`;
        preview.appendChild(more);
    }
}

async function uploadAndDetect() {
    const input = document.getElementById('uploadFileInput');
    const conf = parseFloat(document.getElementById('uploadConf').value) || 0.4;
    const resultsEl = document.getElementById('uploadResults');
    const btn = document.getElementById('uploadBtn');

    if (input.files.length === 0) {
        toast('Select an image or video first', 'error');
        return;
    }

    btn.disabled = true;
    btn.textContent = 'Detecting...';
    resultsEl.innerHTML = '<div class="empty-state"><div class="spinner"></div><p>Running ML inference...</p></div>';

    let allResults = [];

    for (let i = 0; i < input.files.length; i++) {
        const file = input.files[i];
        const formData = new FormData();
        const isVideo = file.type.startsWith('video/');
        
        formData.append(isVideo ? 'video' : 'image', file);

        try {
            const endpoint = isVideo ? `/api/detect/video?conf=${conf}&frame_skip=5` : `/detect?conf=${conf}`;
            const res = await fetch(endpoint, {
                method: 'POST',
                body: formData,
            });
            if (!res.ok) {
                const err = await res.json();
                toast(`Failed: ${err.detail || res.statusText}`, 'error');
                continue;
            }
            const data = await res.json();
            // Video endpoint returns frames_processed, total_frames, etc.
            if (isVideo) {
                allResults.push({ file: file.name, count: data.detections.length, ...data });
            } else {
                allResults.push({ file: file.name, ...data });
            }
        } catch(e) {
            console.error('Upload error:', e);
            toast(`Error processing ${file.name}`, 'error');
        }
    }

    btn.disabled = false;
    btn.textContent = 'Detect Diseases';

    if (allResults.length === 0) {
        resultsEl.innerHTML = '<div class="empty-state"><p>No results.</p></div>';
        return;
    }

    // Render results
    let html = '';
    for (const result of allResults) {
        html += `<div class="upload-result-file">
            <div class="upload-result-header">
                <span class="upload-filename">${result.file}</span>
                <span class="badge badge-${result.count > 0 ? 'approved' : 'pending'}">${result.count} detection${result.count !== 1 ? 's' : ''}</span>
            </div>`;
            
        if (result.frames_processed) {
            html += `<div style="font-size: 12px; color: var(--text-muted); margin-bottom: 8px; padding-left: 12px;">Processed ${result.frames_processed} frames out of ${result.total_frames} (Skipping 5)</div>`;
        }

        if (result.detections.length > 0) {
            for (const det of result.detections) {
                const confClass = det.confidence > 0.8 ? 'high' : det.confidence > 0.6 ? 'medium' : 'low';
                html += `
                    <div class="upload-det-row">
                        <span class="legend-dot ${det.class_name}"></span>
                        <span>${DISEASE_LABELS[det.class_name] || det.class_name} ${det.frame ? `(Frame ${det.frame})` : ''}</span>
                        <span class="detection-conf ${confClass}">${(det.confidence * 100).toFixed(1)}%</span>
                        <span class="upload-bbox">[${det.bbox.map(v => v.toFixed(0)).join(', ')}]</span>
                    </div>`;
            }
        } else {
            html += '<div class="upload-det-empty">No leaf detected in frame</div>';
        }
        html += '</div>';
    }

    resultsEl.innerHTML = html;
    toast(`Processed ${allResults.length} file(s)`, 'success');
}

// ══════════════════════════════════════════
// LIVE STREAM DETECTION
// ══════════════════════════════════════════

let _streamWs = null;
let _streamActive = false;
let _streamFeedTimer = null;
let _streamTotalDetections = 0;
const _streamDetHistory = [];   // last N detection events
const STREAM_DET_HISTORY_MAX = 50;

function openStreamModal() {
    document.getElementById('streamModal').classList.add('visible');
    // If stream is active, start live feed display
    _pollStreamStatus();
}

function closeStreamModal() {
    document.getElementById('streamModal').classList.remove('visible');
    _stopFeedRefresh();
}

async function toggleStream() {
    const btn = document.getElementById('streamConnectBtn');
    if (_streamActive) {
        // Stop stream
        btn.disabled = true;
        btn.innerHTML = '<div class="spinner"></div> Stopping...';
        try {
            await fetch('/api/stream/stop', { method: 'POST' });
            toast('Stream stopped', 'info');
        } catch(e) {
            toast('Failed to stop stream', 'error');
        }
        _onStreamStopped();
        btn.disabled = false;
    } else {
        // Start stream
        const url = document.getElementById('streamUrl').value.trim();
        if (!url) {
            toast('Enter a stream URL', 'error');
            return;
        }
        const conf = parseFloat(document.getElementById('streamConf').value) || 0.6;
        const skip = parseInt(document.getElementById('streamSkip').value) || 5;

        btn.disabled = true;
        btn.innerHTML = '<div class="spinner"></div> Connecting...';
        try {
            const res = await fetch('/api/stream/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    stream_url: url,
                    frame_skip: skip,
                    conf_threshold: conf,
                }),
            });
            const data = await res.json();
            if (!res.ok) {
                toast(`Stream error: ${data.detail || res.statusText}`, 'error');
                btn.disabled = false;
                _updateConnectBtn(false);
                return;
            }
            toast('Stream connected \u2014 YOLO inference running', 'success');
            _onStreamStarted();
        } catch(e) {
            toast('Failed to connect stream', 'error');
            btn.disabled = false;
            _updateConnectBtn(false);
        }
    }
}

function _onStreamStarted() {
    _streamActive = true;
    _streamTotalDetections = 0;
    _streamDetHistory.length = 0;
    _updateConnectBtn(true);
    _updateStatusBadge(true);
    document.getElementById('streamVideoOverlay').style.display = 'none';
    document.getElementById('streamStatsOverlay').style.display = 'flex';
    document.getElementById('snapshotBtn').disabled = false;
    document.getElementById('streamDetLog').innerHTML =
        '<div class="empty-state" style="padding:16px;"><p>Waiting for detections...</p></div>';

    _startFeedRefresh();
    _connectStreamWS();
}

function _onStreamStopped() {
    _streamActive = false;
    _updateConnectBtn(false);
    _updateStatusBadge(false);
    document.getElementById('streamVideoOverlay').style.display = 'flex';
    document.getElementById('streamStatsOverlay').style.display = 'none';
    document.getElementById('snapshotBtn').disabled = true;
    _stopFeedRefresh();
    _disconnectStreamWS();
}

function _updateConnectBtn(isActive) {
    const btn = document.getElementById('streamConnectBtn');
    btn.disabled = false;
    if (isActive) {
        btn.className = 'btn btn-danger';
        btn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2"/></svg> Disconnect';
    } else {
        btn.className = 'btn btn-primary';
        btn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg> Connect';
    }
}

function _updateStatusBadge(isActive) {
    const badge = document.getElementById('streamStatusBadge');
    if (isActive) {
        badge.textContent = 'LIVE';
        badge.className = 'stream-status-badge live';
    } else {
        badge.textContent = 'OFFLINE';
        badge.className = 'stream-status-badge';
    }
}

// ── Video Feed Refresh (polling JPEG frames) ──

function _startFeedRefresh() {
    _stopFeedRefresh();
    const img = document.getElementById('streamVideoFeed');
    let frameSeq = 0;
    _streamFeedTimer = setInterval(() => {
        if (!_streamActive) return;
        // Cache-bust with timestamp to force reload
        img.src = `/api/stream/frame?annotated=true&_t=${Date.now()}_${frameSeq++}`;
    }, 200);  // 5 FPS refresh for the displayed video
}

function _stopFeedRefresh() {
    if (_streamFeedTimer) {
        clearInterval(_streamFeedTimer);
        _streamFeedTimer = null;
    }
}

// ── WebSocket Consumer ──

function _connectStreamWS() {
    _disconnectStreamWS();
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    _streamWs = new WebSocket(`${proto}//${location.host}/ws/stream-detect`);

    _streamWs.onopen = () => {
        console.log('[Stream WS] Connected');
    };

    _streamWs.onmessage = (event) => {
        try {
            const msg = JSON.parse(event.data);
            if (msg.type === 'stream_detection') {
                _handleStreamDetection(msg);
            } else if (msg.type === 'stream_status') {
                _handleStreamStatus(msg.data);
            } else if (msg.type === 'stream_stopped') {
                _onStreamStopped();
            }
        } catch(e) {
            console.error('[Stream WS] Parse error:', e);
        }
    };

    _streamWs.onclose = () => {
        console.log('[Stream WS] Disconnected');
    };

    _streamWs.onerror = (e) => {
        console.error('[Stream WS] Error:', e);
    };
}

function _disconnectStreamWS() {
    if (_streamWs) {
        _streamWs.close();
        _streamWs = null;
    }
}

function _handleStreamDetection(msg) {
    const dets = msg.detections || [];
    _streamTotalDetections += dets.length;

    // Update stats overlay
    document.getElementById('streamDetCount').textContent = _streamTotalDetections;

    // Add to history
    _streamDetHistory.unshift({
        frame_id: msg.frame_id,
        timestamp: msg.timestamp,
        detections: dets,
    });
    if (_streamDetHistory.length > STREAM_DET_HISTORY_MAX) {
        _streamDetHistory.length = STREAM_DET_HISTORY_MAX;
    }

    // Render detection log
    _renderStreamDetLog();
}

function _handleStreamStatus(data) {
    document.getElementById('streamFps').textContent = data.fps || '0';
    document.getElementById('streamFrameCount').textContent = data.frame_count || '0';
    document.getElementById('streamProcCount').textContent = data.processed_count || '0';

    if (data.active && !_streamActive) {
        _onStreamStarted();
    }
}

function _renderStreamDetLog() {
    const container = document.getElementById('streamDetLog');

    if (_streamDetHistory.length === 0) {
        container.innerHTML = '<div class="empty-state" style="padding:16px;"><p>Waiting for detections...</p></div>';
        return;
    }

    let html = '';
    // Show latest 15 detection events
    const shown = _streamDetHistory.slice(0, 15);
    for (const event of shown) {
        const timeStr = new Date(event.timestamp * 1000).toLocaleTimeString();
        html += `<div class="stream-det-event">`;
        html += `<div class="stream-det-event-header"><span class="stream-det-frame">Frame #${event.frame_id}</span><span class="stream-det-time">${timeStr}</span></div>`;

        if (event.detections.length === 0) {
            html += '<div class="upload-det-empty">No leaf detected in frame</div></div>';
            continue;
        }

        for (const det of event.detections) {
            const confClass = det.confidence > 0.8 ? 'high' : det.confidence > 0.6 ? 'medium' : 'low';
            html += `<div class="upload-det-row">
                <span class="legend-dot ${det.class_name}"></span>
                <span>${DISEASE_LABELS[det.class_name] || det.class_name}</span>
                <span class="detection-conf ${confClass}">${(det.confidence * 100).toFixed(1)}%</span>
            </div>`;
        }
        html += '</div>';
    }

    container.innerHTML = html;
}

// ── Slider Live Updates (push via WS) ──

document.addEventListener('DOMContentLoaded', () => {
    const confSlider = document.getElementById('streamConf');
    const confVal = document.getElementById('streamConfValue');
    const skipSlider = document.getElementById('streamSkip');
    const skipVal = document.getElementById('streamSkipValue');

    if (confSlider) {
        confSlider.addEventListener('input', () => {
            confVal.textContent = parseFloat(confSlider.value).toFixed(2);
        });
        confSlider.addEventListener('change', () => {
            if (_streamWs && _streamWs.readyState === WebSocket.OPEN) {
                _streamWs.send(JSON.stringify({
                    type: 'update_settings',
                    conf_threshold: parseFloat(confSlider.value),
                }));
            }
        });
    }

    if (skipSlider) {
        skipSlider.addEventListener('input', () => {
            skipVal.textContent = skipSlider.value;
        });
        skipSlider.addEventListener('change', () => {
            if (_streamWs && _streamWs.readyState === WebSocket.OPEN) {
                _streamWs.send(JSON.stringify({
                    type: 'update_settings',
                    frame_skip: parseInt(skipSlider.value),
                }));
            }
        });
    }
});

// ── Snapshot ──

async function captureSnapshot() {
    try {
        const res = await fetch('/api/stream/snapshot', { method: 'POST' });
        const data = await res.json();
        toast(`Snapshot saved: ${data.count} detection(s)`, 'success');
    } catch(e) {
        toast('Snapshot failed', 'error');
    }
}

// ── Poll stream status on modal open ──

async function _pollStreamStatus() {
    try {
        const res = await fetch('/api/stream/status');
        const data = await res.json();
        if (data.active) {
            _onStreamStarted();
            _handleStreamStatus(data);
            document.getElementById('streamUrl').value = data.stream_url || '';
            document.getElementById('streamConf').value = data.conf_threshold || 0.6;
            document.getElementById('streamConfValue').textContent = (data.conf_threshold || 0.6).toFixed(2);
            document.getElementById('streamSkip').value = data.frame_skip || 5;
            document.getElementById('streamSkipValue').textContent = data.frame_skip || 5;
        }
    } catch(e) {
        // Stream status endpoint not available, that's fine
    }
}
