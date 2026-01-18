VIEWER_HTML = r"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            width: 100%;
            height: 100vh;
            overflow: hidden;
            background: #0a0a0f;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            user-select: none;
            -webkit-user-select: none;
        }

        #container {
            width: 100%;
            height: 100%;
            position: relative;
        }

        #threejs-container {
            width: 100%;
            height: 100%;
        }

        #color-picker-container {
            position: absolute;
            top: 8px;
            left: 8px;
            right: 8px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(10, 10, 15, 0.9);
            border: 1px solid rgba(233, 61, 130, 0.3);
            border-radius: 6px;
            padding: 6px 10px;
            backdrop-filter: blur(4px);
        }

        #prompt-preview {
            flex: 1;
            font-size: 11px;
            color: #E93D82;
            font-family: 'Consolas', 'Monaco', monospace;
            word-break: break-all;
            line-height: 1.4;
            margin-right: 12px;
        }

        #color-section {
            display: flex;
            align-items: center;
            gap: 6px;
            flex-shrink: 0;
        }

        #color-picker-label {
            font-size: 10px;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        #color-picker {
            width: 24px;
            height: 24px;
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-radius: 4px;
            cursor: pointer;
            padding: 0;
            background: none;
        }

        #color-picker::-webkit-color-swatch-wrapper {
            padding: 0;
        }

        #color-picker::-webkit-color-swatch {
            border: none;
            border-radius: 2px;
        }

        #color-picker::-moz-color-swatch {
            border: none;
            border-radius: 2px;
        }

        #color-hex-display {
            font-size: 11px;
            color: #FFB800;
            font-family: 'Consolas', 'Monaco', monospace;
            font-weight: 600;
            min-width: 60px;
        }

        #info-panel {
            position: absolute;
            bottom: 8px;
            left: 8px;
            right: 8px;
            background: rgba(10, 10, 15, 0.9);
            border: 1px solid rgba(233, 61, 130, 0.3);
            border-radius: 6px;
            padding: 8px 12px;
            font-size: 11px;
            color: #e0e0e0;
            display: flex;
            justify-content: space-around;
            backdrop-filter: blur(4px);
        }

        .param-item {
            text-align: center;
        }

        .param-label {
            color: #888;
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .param-value {
            color: #E93D82;
            font-weight: 600;
            font-size: 13px;
        }

        .param-value.elevation {
            color: #00FFD0;
        }

        .param-value.zoom {
            color: #FFB800;
        }

        #view-btn {
            position: absolute;
            right: 8px;
            bottom: 100%;
            margin-bottom: 8px;
            width: 24px;
            height: 24px;
            border-radius: 4px;
            border: 1px solid rgba(233, 61, 130, 0.4);
            background: rgba(10, 10, 15, 0.8);
            color: #E93D82;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
            transition: all 0.2s ease;
        }

        #view-btn:hover {
            background: rgba(233, 61, 130, 0.2);
            border-color: #E93D82;
        }

        #view-btn:active {
            transform: scale(0.95);
        }

        #reset-btn {
            position: absolute;
            right: 8px;
            bottom: 8px;
            width: 24px;
            height: 24px;
            border-radius: 4px;
            border: 1px solid rgba(233, 61, 130, 0.4);
            background: rgba(10, 10, 15, 0.8);
            color: #E93D82;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
            transition: all 0.2s ease;
        }

        #reset-btn:hover {
            background: rgba(233, 61, 130, 0.2);
            border-color: #E93D82;
        }

        #reset-btn:active {
            transform: scale(0.95);
        }

        #zoom-slider-container {
            position: absolute;
            right: 8px;
            top: 50%;
            transform: translateY(-50%);
            width: 24px;
            height: 160px;
            background: rgba(10, 10, 15, 0.9);
            border: 1px solid rgba(255, 184, 0, 0.3);
            border-radius: 12px;
            padding: 8px 0;
            display: flex;
            justify-content: center;
            backdrop-filter: blur(4px);
            cursor: pointer;
        }

        #zoom-slider-track {
            position: absolute;
            width: 4px;
            height: calc(100% - 16px);
            top: 8px;
            background: rgba(255, 184, 0, 0.3);
            border-radius: 2px;
            cursor: pointer;
        }

        #zoom-slider-handle {
            position: absolute;
            width: 16px;
            height: 16px;
            background: #FFB800;
            border-radius: 50%;
            cursor: pointer;
            box-shadow: 0 0 10px rgba(255, 184, 0, 0.5);
            left: 50%;
            transform: translate(-50%, -50%);
        }
    </style>
</head>
<body>
    <div id="container">
        <div id="threejs-container"></div>
        <div id="color-picker-container">
            <div id="prompt-preview">cinematic relighting</div>
            <div id="color-section">
                <span id="color-picker-label">Color</span>
                <input type="color" id="color-picker" value="#FFFFFF">
                <span id="color-hex-display">#FFFFFF</span>
            </div>
        </div>
        <div id="info-panel">
            <div class="param-item">
                <div class="param-label">Azimuth</div>
                <div class="param-value" id="h-value">0°</div>
            </div>
            <div class="param-item">
                <div class="param-label">Elevation</div>
                <div class="param-value elevation" id="v-value">0°</div>
            </div>
            <div class="param-item">
                <div class="param-label">Intensity</div>
                <div class="param-value zoom" id="z-value">5.0</div>
            </div>
            <button id="view-btn" title="Toggle View">👁️</button>
            <button id="reset-btn" title="Reset to defaults">↺</button>
        </div>
        <div id="zoom-slider-container">
            <div id="zoom-slider-track"></div>
            <div id="zoom-slider-handle"></div>
        </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
        let state = {
            azimuth: 0,
            elevation: 0,
            distance: 5,
            lightColor: "#FFFFFF",
            imageUrl: null,
            cameraView: false
        };

        let threeScene = null;

        const container = document.getElementById('threejs-container');
        const hValueEl = document.getElementById('h-value');
        const vValueEl = document.getElementById('v-value');
        const zValueEl = document.getElementById('z-value');
        const promptPreviewEl = document.getElementById('prompt-preview');
        const viewBtn = document.getElementById('view-btn');
        const zoomSliderContainer = document.getElementById('zoom-slider-container');
        const zoomSliderHandle = document.getElementById('zoom-slider-handle');
        const colorPicker = document.getElementById('color-picker');
        const colorHexDisplay = document.getElementById('color-hex-display');

        function normalizeHex(hex) {
            if (!hex) return "#FFFFFF";
            const upper = String(hex).trim().toUpperCase();
            if (upper.startsWith("#") && (upper.length === 7 || upper.length === 4)) return upper;
            if (/^[0-9A-F]{6}$/.test(upper)) return "#" + upper;
            return "#FFFFFF";
        }

        colorPicker.addEventListener('input', (e) => {
            state.lightColor = normalizeHex(e.target.value);
            colorHexDisplay.textContent = state.lightColor;
            if (threeScene) threeScene.syncFromState();
            sendAngleUpdate();
        });

        colorPicker.addEventListener('change', (e) => {
            state.lightColor = normalizeHex(e.target.value);
            colorHexDisplay.textContent = state.lightColor;
            if (threeScene) threeScene.syncFromState();
            sendAngleUpdate();
        });

        const GLOBAL_CONSTRAINTS = "SCENE LOCK, FIXED VIEWPOINT, maintaining character consistency and pose. RELIGHTING ONLY: ";

        function buildLightingPrompt(azimuth, elevation, intensity, colorHex) {
            const az = ((azimuth % 360) + 360) % 360;
            let posDesc;
            if (az >= 337.5 || az < 22.5) posDesc = "light source in front";
            else if (az < 67.5) posDesc = "light source from the front-right";
            else if (az < 112.5) posDesc = "light source from the right";
            else if (az < 157.5) posDesc = "light source from the back-right";
            else if (az < 202.5) posDesc = "light source from behind";
            else if (az < 247.5) posDesc = "light source from the back-left";
            else if (az < 292.5) posDesc = "light source from the left";
            else posDesc = "light source from the front-left";

            const e = elevation;
            let elevDesc;
            if (e >= -90 && e < -30) elevDesc = "uplighting, light source positioned below the character, light shining upwards";
            else if (e >= -30 && e < -10) elevDesc = "low-angle light source from below, upward illumination";
            else if (e >= -10 && e < 20) elevDesc = "horizontal level light source";
            else if (e >= 20 && e < 60) elevDesc = "high-angle light source";
            else elevDesc = "overhead top-down light source";

            let intDesc;
            if (intensity < 3.0) intDesc = "soft";
            else if (intensity < 7.0) intDesc = "bright";
            else intDesc = "intense";

            const colorDesc = "colored light (" + normalizeHex(colorHex) + ")";
            return GLOBAL_CONSTRAINTS + posDesc + ", " + elevDesc + ", " + intDesc + " " + colorDesc + ", cinematic relighting";
        }

        function buildLightingPromptPreview(azimuth, elevation, intensity, colorHex) {
            const full = buildLightingPrompt(azimuth, elevation, intensity, colorHex);
            if (full.startsWith(GLOBAL_CONSTRAINTS)) return full.slice(GLOBAL_CONSTRAINTS.length);
            return full;
        }

        function updateZoomSlider() {
            const min = 0;
            const max = 10;
            const value = Math.max(min, Math.min(max, state.distance));
            const t = (value - min) / (max - min);
            const padding = 8;
            const handleRadius = 8;
            const containerHeight = zoomSliderContainer.getBoundingClientRect().height || zoomSliderContainer.clientHeight;
            if (!containerHeight || containerHeight <= (padding * 2 + handleRadius * 2)) {
                return;
            }
            const trackHeight = Math.max(0, containerHeight - padding * 2 - handleRadius * 2);
            const y = padding + handleRadius + (1 - t) * trackHeight;
            zoomSliderHandle.style.top = y + "px";
        }

        function scheduleZoomSliderUpdate(tries = 0) {
            updateZoomSlider();
            const containerHeight = zoomSliderContainer.getBoundingClientRect().height || zoomSliderContainer.clientHeight;
            if (containerHeight && containerHeight > 0) return;
            if (tries >= 30) return;
            requestAnimationFrame(() => scheduleZoomSliderUpdate(tries + 1));
        }

        function handleZoomSlider(clientY) {
            const rect = zoomSliderContainer.getBoundingClientRect();
            const padding = 8;
            const handleRadius = 8;
            const trackTop = rect.top + padding + handleRadius;
            const trackBottom = rect.bottom - padding - handleRadius;
            const clampedY = Math.max(trackTop, Math.min(trackBottom, clientY));
            const t = 1 - (clampedY - trackTop) / (trackBottom - trackTop);
            const min = 0;
            const max = 10;
            state.distance = Math.round((min + t * (max - min)) * 10) / 10;
            if (threeScene) threeScene.syncFromState();
            updateDisplay();
            sendAngleUpdate();
        }

        let isDraggingZoom = false;
        zoomSliderContainer.addEventListener('mousedown', (e) => {
            e.preventDefault();
            isDraggingZoom = true;
            handleZoomSlider(e.clientY);
        });

        window.addEventListener('mousemove', (e) => {
            if (isDraggingZoom) {
                e.preventDefault();
                handleZoomSlider(e.clientY);
            }
        });

        window.addEventListener('mouseup', () => {
            isDraggingZoom = false;
        });

        zoomSliderContainer.addEventListener('touchstart', (e) => {
            e.preventDefault();
            isDraggingZoom = true;
            handleZoomSlider(e.touches[0].clientY);
        }, { passive: false });

        window.addEventListener('touchmove', (e) => {
            if (isDraggingZoom) {
                e.preventDefault();
                handleZoomSlider(e.touches[0].clientY);
            }
        }, { passive: false });

        window.addEventListener('touchend', () => {
            isDraggingZoom = false;
        });

        function updateDisplay() {
            hValueEl.textContent = Math.round(state.azimuth) + '°';
            vValueEl.textContent = Math.round(state.elevation) + '°';
            zValueEl.textContent = state.distance.toFixed(1);
            updateZoomSlider();
            promptPreviewEl.textContent = buildLightingPromptPreview(state.azimuth, state.elevation, state.distance, state.lightColor);
        }

        function sendAngleUpdate() {
            window.parent.postMessage({
                type: 'ANGLE_UPDATE',
                horizontal: Math.round(state.azimuth),
                vertical: Math.round(state.elevation),
                zoom: Math.round(state.distance * 10) / 10,
                lightColor: state.lightColor || "#FFFFFF"
            }, '*');
        }

        function resetToDefaults() {
            state.azimuth = 0;
            state.elevation = 0;
            state.distance = 5.0;
            state.lightColor = "#FFFFFF";
            colorPicker.value = state.lightColor;
            colorHexDisplay.textContent = state.lightColor;
            if (threeScene) threeScene.syncFromState();
            updateDisplay();
            sendAngleUpdate();
        }

        document.getElementById('reset-btn').addEventListener('click', resetToDefaults);

        function initThreeJS() {
            const width = container.clientWidth;
            const height = container.clientHeight;

            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x0a0a0f);

            const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
            camera.position.set(4, 3.5, 4);
            camera.lookAt(0, 0.3, 0);

            const previewCamera = new THREE.PerspectiveCamera(50, width / height, 0.1, 100);

            let useCameraView = false;
            let activeCamera = camera;

            const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
            renderer.setSize(width, height);
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            renderer.outputEncoding = THREE.sRGBEncoding;
            container.appendChild(renderer.domElement);

            const ambientLight = new THREE.AmbientLight(0xffffff, 0.25);
            scene.add(ambientLight);

            const controlledLight = new THREE.DirectionalLight(0xffffff, 1.0);
            controlledLight.position.set(2, 2, 2);
            const controlledTarget = new THREE.Object3D();
            controlledTarget.position.set(0, 0.5, 0);
            scene.add(controlledTarget);
            controlledLight.target = controlledTarget;
            scene.add(controlledLight);

            const fillLight = new THREE.DirectionalLight(0xE93D82, 0.2);
            fillLight.position.set(-5, 5, -5);
            scene.add(fillLight);

            const gridHelper = new THREE.GridHelper(5, 20, 0x1a1a2e, 0x12121a);
            gridHelper.position.y = -0.01;
            scene.add(gridHelper);

            const CENTER = new THREE.Vector3(0, 0.5, 0);
            const AZIMUTH_RADIUS = 1.8;
            const ELEVATION_RADIUS = 1.4;
            const ELEV_ARC_X = -0.8;

            let liveAzimuth = state.azimuth;
            let liveElevation = state.elevation;
            let liveDistance = state.distance;

            const cardThickness = 0.45;
            const cardGeo = new THREE.BoxGeometry(1.2, 1.2, cardThickness);

            function createGridTexture() {
                const canvas = document.createElement('canvas');
                const size = 256;
                canvas.width = size;
                canvas.height = size;
                const ctx = canvas.getContext('2d');

                ctx.fillStyle = '#1a1a2a';
                ctx.fillRect(0, 0, size, size);

                ctx.strokeStyle = '#2a2a3a';
                ctx.lineWidth = 1;
                const gridSize = 16;
                for (let i = 0; i <= size; i += gridSize) {
                    ctx.beginPath();
                    ctx.moveTo(i, 0);
                    ctx.lineTo(i, size);
                    ctx.stroke();
                    ctx.beginPath();
                    ctx.moveTo(0, i);
                    ctx.lineTo(size, i);
                    ctx.stroke();
                }

                const texture = new THREE.CanvasTexture(canvas);
                texture.wrapS = THREE.RepeatWrapping;
                texture.wrapT = THREE.RepeatWrapping;
                texture.repeat.set(4, 4);
                return texture;
            }

            const frontMat = new THREE.MeshStandardMaterial({
                color: 0x3a3a4a,
                transparent: false,
                opacity: 1.0,
                metalness: 0.0,
                roughness: 0.95
            });
            const backMat = new THREE.MeshStandardMaterial({
                map: createGridTexture(),
                transparent: true,
                opacity: 0.5,
                metalness: 0.0,
                roughness: 1.0
            });
            const edgeMat = new THREE.MeshStandardMaterial({
                color: 0x1a1a2a,
                transparent: true,
                opacity: 0.28,
                metalness: 0.0,
                roughness: 1.0
            });

            function applyCardMaterialMode(cameraViewEnabled) {
                if (cameraViewEnabled) {
                    backMat.opacity = 0.25;
                    edgeMat.opacity = 0.12;
                } else {
                    backMat.opacity = 0.5;
                    edgeMat.opacity = 0.28;
                }
                frontMat.side = THREE.FrontSide;
                backMat.side = THREE.DoubleSide;
                edgeMat.side = THREE.DoubleSide;
                frontMat.depthWrite = true;
                backMat.depthWrite = false;
                edgeMat.depthWrite = false;
                frontMat.needsUpdate = true;
                backMat.needsUpdate = true;
                edgeMat.needsUpdate = true;
            }

            const cardMaterials = [edgeMat, edgeMat, edgeMat, edgeMat, frontMat, backMat];
            const imagePlane = new THREE.Mesh(cardGeo, cardMaterials);
            imagePlane.position.copy(CENTER);
            scene.add(imagePlane);

            const planeMat = frontMat;
            applyCardMaterialMode(useCameraView);

            const frameGeo = new THREE.EdgesGeometry(cardGeo);
            const frameMat = new THREE.LineBasicMaterial({ color: 0xE93D82 });
            const imageFrame = new THREE.LineSegments(frameGeo, frameMat);
            imageFrame.position.copy(CENTER);
            scene.add(imageFrame);

            const glowRingGeo = new THREE.RingGeometry(0.55, 0.58, 64);
            const glowRingMat = new THREE.MeshBasicMaterial({
                color: 0xE93D82,
                transparent: true,
                opacity: 0.4,
                side: THREE.DoubleSide
            });
            const glowRing = new THREE.Mesh(glowRingGeo, glowRingMat);
            glowRing.position.set(0, 0.01, 0);
            glowRing.rotation.x = -Math.PI / 2;
            scene.add(glowRing);

            const camGeo = new THREE.ConeGeometry(0.15, 0.4, 4);
            const camMat = new THREE.MeshStandardMaterial({
                color: 0xE93D82,
                emissive: 0xE93D82,
                emissiveIntensity: 0.5,
                metalness: 0.8,
                roughness: 0.2
            });
            const cameraIndicator = new THREE.Mesh(camGeo, camMat);
            scene.add(cameraIndicator);

            const camGlowGeo = new THREE.SphereGeometry(0.08, 16, 16);
            const camGlowMat = new THREE.MeshBasicMaterial({
                color: 0xff6ba8,
                transparent: true,
                opacity: 0.8
            });
            const camGlow = new THREE.Mesh(camGlowGeo, camGlowMat);
            scene.add(camGlow);

            const azRingGeo = new THREE.TorusGeometry(AZIMUTH_RADIUS, 0.04, 16, 100);
            const azRingMat = new THREE.MeshBasicMaterial({
                color: 0xE93D82,
                transparent: true,
                opacity: 0.7
            });
            const azimuthRing = new THREE.Mesh(azRingGeo, azRingMat);
            azimuthRing.rotation.x = Math.PI / 2;
            azimuthRing.position.y = 0.02;
            scene.add(azimuthRing);

            const azHandleGeo = new THREE.SphereGeometry(0.16, 32, 32);
            const azHandleMat = new THREE.MeshStandardMaterial({
                color: 0xE93D82,
                emissive: 0xE93D82,
                emissiveIntensity: 0.6,
                metalness: 0.3,
                roughness: 0.4
            });
            const azimuthHandle = new THREE.Mesh(azHandleGeo, azHandleMat);
            scene.add(azimuthHandle);

            const azGlowGeo = new THREE.SphereGeometry(0.22, 16, 16);
            const azGlowMat = new THREE.MeshBasicMaterial({
                color: 0xE93D82,
                transparent: true,
                opacity: 0.2
            });
            const azGlow = new THREE.Mesh(azGlowGeo, azGlowMat);
            scene.add(azGlow);

            const arcPoints = [];
            for (let i = 0; i <= 32; i++) {
                const angle = (-90 + (180 * i / 32)) * Math.PI / 180;
                arcPoints.push(new THREE.Vector3(
                    ELEV_ARC_X,
                    ELEVATION_RADIUS * Math.sin(angle) + CENTER.y,
                    ELEVATION_RADIUS * Math.cos(angle)
                ));
            }
            const arcCurve = new THREE.CatmullRomCurve3(arcPoints);
            const elArcGeo = new THREE.TubeGeometry(arcCurve, 32, 0.04, 8, false);
            const elArcMat = new THREE.MeshBasicMaterial({
                color: 0x00FFD0,
                transparent: true,
                opacity: 0.8
            });
            const elevationArc = new THREE.Mesh(elArcGeo, elArcMat);
            scene.add(elevationArc);

            const elHandleGeo = new THREE.SphereGeometry(0.16, 32, 32);
            const elHandleMat = new THREE.MeshStandardMaterial({
                color: 0x00FFD0,
                emissive: 0x00FFD0,
                emissiveIntensity: 0.6,
                metalness: 0.3,
                roughness: 0.4
            });
            const elevationHandle = new THREE.Mesh(elHandleGeo, elHandleMat);
            scene.add(elevationHandle);

            const elGlowGeo = new THREE.SphereGeometry(0.22, 16, 16);
            const elGlowMat = new THREE.MeshBasicMaterial({
                color: 0x00FFD0,
                transparent: true,
                opacity: 0.2
            });
            const elGlow = new THREE.Mesh(elGlowGeo, elGlowMat);
            scene.add(elGlow);

            const distHandleGeo = new THREE.SphereGeometry(0.15, 32, 32);
            const distHandleMat = new THREE.MeshStandardMaterial({
                color: 0xFFB800,
                emissive: 0xFFB800,
                emissiveIntensity: 0.7,
                metalness: 0.5,
                roughness: 0.3
            });
            const distanceHandle = new THREE.Mesh(distHandleGeo, distHandleMat);
            scene.add(distanceHandle);

            const distGlowGeo = new THREE.SphereGeometry(0.21, 16, 16);
            const distGlowMat = new THREE.MeshBasicMaterial({
                color: 0xFFB800,
                transparent: true,
                opacity: 0.25
            });
            const distGlow = new THREE.Mesh(distGlowGeo, distGlowMat);
            scene.add(distGlow);

            const initialLightColor = normalizeHex(state.lightColor || "#FFFFFF");
            distHandleMat.color.set(initialLightColor);
            distHandleMat.emissive.set(initialLightColor);
            distGlowMat.color.set(initialLightColor);

            let distanceTube = null;
            function updateDistanceLine(start, end) {
                if (distanceTube) {
                    scene.remove(distanceTube);
                    distanceTube.geometry.dispose();
                    distanceTube.material.dispose();
                }
                const path = new THREE.LineCurve3(start, end);
                const tubeGeo = new THREE.TubeGeometry(path, 1, 0.025, 8, false);
                const tubeMat = new THREE.MeshBasicMaterial({
                    color: normalizeHex(state.lightColor || "#FFB800"),
                    transparent: true,
                    opacity: 0.8
                });
                distanceTube = new THREE.Mesh(tubeGeo, tubeMat);
                scene.add(distanceTube);
            }

            function updateVisuals() {
                const azRad = (liveAzimuth * Math.PI) / 180;
                const elRad = (liveElevation * Math.PI) / 180;
                const visualDist = 2.6 - (liveDistance / 10) * 2.0;

                const camX = visualDist * Math.sin(azRad) * Math.cos(elRad);
                const camY = CENTER.y + visualDist * Math.sin(elRad);
                const camZ = visualDist * Math.cos(azRad) * Math.cos(elRad);

                cameraIndicator.position.set(camX, camY, camZ);
                cameraIndicator.lookAt(CENTER);
                cameraIndicator.rotateX(Math.PI / 2);
                camGlow.position.copy(cameraIndicator.position);

                const azX = AZIMUTH_RADIUS * Math.sin(azRad);
                const azZ = AZIMUTH_RADIUS * Math.cos(azRad);
                azimuthHandle.position.set(azX, 0.16, azZ);
                azGlow.position.copy(azimuthHandle.position);

                const elY = CENTER.y + ELEVATION_RADIUS * Math.sin(elRad);
                const elZ = ELEVATION_RADIUS * Math.cos(elRad);
                elevationHandle.position.set(ELEV_ARC_X, elY, elZ);
                elGlow.position.copy(elevationHandle.position);

                const distT = 0.15 + ((10 - liveDistance) / 10) * 0.7;
                distanceHandle.position.lerpVectors(CENTER, cameraIndicator.position, distT);
                distGlow.position.copy(distanceHandle.position);

                updateDistanceLine(CENTER.clone(), cameraIndicator.position.clone());

                previewCamera.position.copy(cameraIndicator.position);
                previewCamera.lookAt(CENTER);

                controlledLight.position.copy(cameraIndicator.position);
                controlledTarget.position.copy(CENTER);
                controlledLight.color.set(normalizeHex(state.lightColor));
                controlledLight.intensity = Math.max(0.0, liveDistance / 5.0);

                glowRing.rotation.z += 0.005;
            }

            updateVisuals();

            const raycaster = new THREE.Raycaster();
            const mouse = new THREE.Vector2();
            let isDragging = false;
            let dragTarget = null;
            let hoveredHandle = null;
            let dragStartMouseY = 0;
            let dragStartDistance = 0;
            let isOrbiting = false;
            let lastMouseX = 0;
            let lastMouseY = 0;

            function getMousePos(event) {
                const rect = renderer.domElement.getBoundingClientRect();
                mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
                mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
            }

            function setHandleScale(handle, glow, scale) {
                handle.scale.setScalar(scale);
                if (glow) glow.scale.setScalar(scale);
            }

            function onPointerDown(event) {
                if (event.preventDefault) {
                    event.preventDefault();
                }
                if (useCameraView) {
                    isOrbiting = true;
                    lastMouseX = event.clientX;
                    lastMouseY = event.clientY;
                    renderer.domElement.style.cursor = 'grabbing';
                    return;
                }
                getMousePos(event);
                raycaster.setFromCamera(mouse, activeCamera);

                const intersects = raycaster.intersectObjects([azimuthHandle, elevationHandle, distanceHandle], true);
                if (intersects.length > 0) {
                    isDragging = true;
                    dragTarget = intersects[0].object;
                    renderer.domElement.style.cursor = 'grabbing';
                    if (dragTarget === distanceHandle) {
                        dragStartMouseY = event.clientY;
                        dragStartDistance = liveDistance;
                    }
                }
            }

            function onPointerMove(event) {
                if (useCameraView && isOrbiting) {
                    const deltaX = event.clientX - lastMouseX;
                    const deltaY = event.clientY - lastMouseY;
                    lastMouseX = event.clientX;
                    lastMouseY = event.clientY;
                    state.azimuth -= deltaX * 0.5;
                    state.elevation += deltaY * 0.5;
                    state.elevation = Math.max(-90, Math.min(90, state.elevation));
                    if (state.azimuth < 0) state.azimuth += 360;
                    if (state.azimuth >= 360) state.azimuth -= 360;
                    liveAzimuth = state.azimuth;
                    liveElevation = state.elevation;
                    updateVisuals();
                    updateDisplay();
                    sendAngleUpdate();
                    return;
                }
                if (useCameraView) {
                    renderer.domElement.style.cursor = 'grab';
                    return;
                }
                if (!isDragging) {
                    getMousePos(event);
                    raycaster.setFromCamera(mouse, activeCamera);
                    const intersects = raycaster.intersectObjects([azimuthHandle, elevationHandle, distanceHandle], true);
                    if (intersects.length > 0) {
                        hoveredHandle = intersects[0].object;
                        renderer.domElement.style.cursor = 'grab';
                    } else {
                        hoveredHandle = null;
                        renderer.domElement.style.cursor = 'default';
                    }
                    return;
                }

                if (dragTarget === azimuthHandle) {
                    getMousePos(event);
                    raycaster.setFromCamera(mouse, activeCamera);
                    const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -0.16);
                    const hitPoint = new THREE.Vector3();
                    raycaster.ray.intersectPlane(plane, hitPoint);
                    const angle = Math.atan2(hitPoint.x, hitPoint.z);
                    liveAzimuth = (angle * 180 / Math.PI + 360) % 360;
                    state.azimuth = liveAzimuth;
                    updateVisuals();
                    updateDisplay();
                    sendAngleUpdate();
                } else if (dragTarget === elevationHandle) {
                    getMousePos(event);
                    raycaster.setFromCamera(mouse, activeCamera);
                    const plane = new THREE.Plane(new THREE.Vector3(1, 0, 0), -ELEV_ARC_X);
                    const hitPoint = new THREE.Vector3();
                    raycaster.ray.intersectPlane(plane, hitPoint);
                    const rel = hitPoint.clone().sub(CENTER);
                    const angle = Math.atan2(rel.y, rel.z);
                    liveElevation = Math.max(-90, Math.min(90, angle * 180 / Math.PI));
                    state.elevation = liveElevation;
                    updateVisuals();
                    updateDisplay();
                    sendAngleUpdate();
                } else if (dragTarget === distanceHandle) {
                    const deltaY = event.clientY - dragStartMouseY;
                    liveDistance = Math.max(0, Math.min(10, dragStartDistance + (-deltaY / 50)));
                    state.distance = liveDistance;
                    updateVisuals();
                    updateDisplay();
                    sendAngleUpdate();
                }
            }

            function onPointerUp() {
                isDragging = false;
                dragTarget = null;
                isOrbiting = false;
                renderer.domElement.style.cursor = useCameraView ? 'grab' : 'default';
            }

            renderer.domElement.addEventListener('mousedown', onPointerDown);
            renderer.domElement.addEventListener('mousemove', onPointerMove);
            renderer.domElement.addEventListener('mouseup', onPointerUp);
            renderer.domElement.addEventListener('mouseleave', onPointerUp);

            renderer.domElement.addEventListener('touchstart', (e) => {
                e.preventDefault();
                onPointerDown({ clientX: e.touches[0].clientX, clientY: e.touches[0].clientY });
            }, { passive: false });

            renderer.domElement.addEventListener('touchmove', (e) => {
                e.preventDefault();
                onPointerMove({ clientX: e.touches[0].clientX, clientY: e.touches[0].clientY });
            }, { passive: false });

            renderer.domElement.addEventListener('touchend', onPointerUp);

            function setCameraView(enabled) {
                useCameraView = enabled;
                applyCardMaterialMode(useCameraView);
                isDragging = false;
                dragTarget = null;
                isOrbiting = false;
                if (useCameraView) {
                    activeCamera = previewCamera;
                    azimuthRing.visible = false;
                    azimuthHandle.visible = false;
                    azGlow.visible = false;
                    elevationArc.visible = false;
                    elevationHandle.visible = false;
                    elGlow.visible = false;
                    distanceHandle.visible = false;
                    distGlow.visible = false;
                    if (distanceTube) distanceTube.visible = false;
                    cameraIndicator.visible = false;
                    camGlow.visible = false;
                    glowRing.visible = false;
                    gridHelper.visible = false;
                    imageFrame.visible = false;
                    renderer.domElement.style.cursor = 'grab';
                } else {
                    activeCamera = camera;
                    azimuthRing.visible = true;
                    azimuthHandle.visible = true;
                    azGlow.visible = true;
                    elevationArc.visible = true;
                    elevationHandle.visible = true;
                    elGlow.visible = true;
                    distanceHandle.visible = true;
                    distGlow.visible = true;
                    if (distanceTube) distanceTube.visible = true;
                    cameraIndicator.visible = true;
                    camGlow.visible = true;
                    glowRing.visible = true;
                    gridHelper.visible = true;
                    imageFrame.visible = true;
                    renderer.domElement.style.cursor = 'default';
                }
            }

            viewBtn.addEventListener('click', () => {
                state.cameraView = !state.cameraView;
                setCameraView(state.cameraView);
                window.parent.postMessage({ type: 'SET_CAMERA_VIEW', cameraView: state.cameraView }, '*');
            });

            function onResize() {
                const w = container.clientWidth;
                const h = container.clientHeight;
                camera.aspect = w / h;
                camera.updateProjectionMatrix();
                previewCamera.aspect = w / h;
                previewCamera.updateProjectionMatrix();
                renderer.setSize(w, h);
            }
            window.addEventListener('resize', onResize);

            threeScene = {
                syncFromState: () => {
                    liveAzimuth = state.azimuth;
                    liveElevation = state.elevation;
                    liveDistance = state.distance;
                    updateVisuals();
                    updateDisplay();
                    if (state.lightColor) {
                        distHandleMat.color.set(state.lightColor);
                        distHandleMat.emissive.set(state.lightColor);
                        distGlowMat.color.set(state.lightColor);
                        if (distanceTube && distanceTube.material) {
                            distanceTube.material.color.set(state.lightColor);
                        }
                    }
                },
                setCameraView: setCameraView,
                updateImage: (url) => {
                    if (url) {
                        const img = new Image();
                        if (!url.startsWith('data:')) {
                            img.crossOrigin = 'anonymous';
                        }

                        img.onload = () => {
                            const tex = new THREE.Texture(img);
                            tex.needsUpdate = true;
                            tex.encoding = THREE.sRGBEncoding;
                            planeMat.map = tex;
                            planeMat.color.set(0xffffff);
                            planeMat.needsUpdate = true;

                            const ar = img.width / img.height;
                            const maxSize = 1.5;
                            let scaleX, scaleY;
                            if (ar > 1) {
                                scaleX = maxSize;
                                scaleY = maxSize / ar;
                            } else {
                                scaleY = maxSize;
                                scaleX = maxSize * ar;
                            }
                            imagePlane.scale.set(scaleX, scaleY, 1);
                            imageFrame.scale.set(scaleX, scaleY, 1);
                        };

                        img.onerror = () => {
                            planeMat.map = null;
                            planeMat.color.set(0xE93D82);
                            planeMat.needsUpdate = true;
                        };

                        img.src = url;
                    } else {
                        planeMat.map = null;
                        planeMat.color.set(0x3a3a4a);
                        planeMat.needsUpdate = true;
                        imagePlane.scale.set(1, 1, 1);
                        imageFrame.scale.set(1, 1, 1);
                    }
                }
            };

            let time = 0;
            let isVisible = true;
            document.addEventListener('visibilitychange', () => {
                isVisible = !document.hidden;
            });

            function animate() {
                requestAnimationFrame(animate);
                if (!isVisible) return;
                time += 0.01;
                const pulse = 1 + Math.sin(time * 2) * 0.03;
                camGlow.scale.setScalar(pulse);
                renderer.render(scene, activeCamera);
            }
            animate();
        }

        window.addEventListener('message', (event) => {
            const data = event.data;

            if (data.type === 'INIT') {
                state.azimuth = data.horizontal || 0;
                state.elevation = data.vertical || 0;
                state.distance = data.zoom || 5;
                state.lightColor = normalizeHex(data.lightColor || "#FFFFFF");
                colorPicker.value = state.lightColor;
                colorHexDisplay.textContent = state.lightColor;
                if (threeScene) {
                    threeScene.syncFromState();
                    threeScene.setCameraView(data.cameraView || false);
                }
            } else if (data.type === 'SYNC_ANGLES') {
                state.azimuth = data.horizontal || 0;
                state.elevation = data.vertical || 0;
                state.distance = data.zoom || 5;
                state.lightColor = normalizeHex(data.lightColor || state.lightColor || "#FFFFFF");
                colorPicker.value = state.lightColor;
                colorHexDisplay.textContent = state.lightColor;
                if (threeScene) {
                    threeScene.syncFromState();
                    threeScene.setCameraView(data.cameraView || false);
                }
                updateDisplay();
            } else if (data.type === 'UPDATE_IMAGE') {
                state.imageUrl = data.imageUrl;
                if (threeScene) {
                    threeScene.updateImage(data.imageUrl);
                }
            } else if (data.type === 'SET_CAMERA_VIEW') {
                if (threeScene) {
                    threeScene.setCameraView(data.cameraView || false);
                }
            }
        });

        initThreeJS();
        window.parent.postMessage({ type: 'VIEWER_READY' }, '*');
        requestAnimationFrame(() => {
            updateDisplay();
            scheduleZoomSliderUpdate();
        });
        window.addEventListener('resize', scheduleZoomSliderUpdate);
    </script>
</body>
</html>
"""


def get_viewer_html():
    import html

    escaped_html = html.escape(VIEWER_HTML)
    iframe_code = f'<iframe id="qwen_anglelight_iframe" srcdoc="{escaped_html}" style="width: 100%; height: 400px; border: none; border-radius: 8px; background-color: #0a0a0f;"></iframe>'

    glue_js = """
    <img src="x" style="display:none" onerror='(function(){
        if (window.qwenAnglelightInitialized) {
            return;
        }
        window.qwenAnglelightInitialized = true;

        function updateGradioInput(elemId, value) {
            const container = document.getElementById(elemId);
            if (!container) return;
            const input = container.querySelector("input, textarea");
            if (!input) return;
            if (input.value === String(value)) return;
            input.value = value;
            input.dispatchEvent(new Event("input", { bubbles: true }));
            input.dispatchEvent(new Event("change", { bubbles: true }));
        }

        function normalizeHex(hex) {
            if (!hex) return "#FFFFFF";
            const upper = String(hex).trim().toUpperCase();
            if (upper.startsWith("#") && (upper.length === 7 || upper.length === 4)) return upper;
            if (/^[0-9A-F]{6}$/.test(upper)) return "#" + upper;
            return "#FFFFFF";
        }

        function generateLightingPrompt(horizontal_angle, vertical_angle, intensity, lightColor) {
            const az = ((horizontal_angle % 360) + 360) % 360;
            let pos_desc;
            if (az >= 337.5 || az < 22.5) pos_desc = "light source in front";
            else if (az < 67.5) pos_desc = "light source from the front-right";
            else if (az < 112.5) pos_desc = "light source from the right";
            else if (az < 157.5) pos_desc = "light source from the back-right";
            else if (az < 202.5) pos_desc = "light source from behind";
            else if (az < 247.5) pos_desc = "light source from the back-left";
            else if (az < 292.5) pos_desc = "light source from the left";
            else pos_desc = "light source from the front-left";

            const e = vertical_angle;
            let elev_desc;
            if (e >= -90 && e < -30) elev_desc = "uplighting, light source positioned below the character, light shining upwards";
            else if (e >= -30 && e < -10) elev_desc = "low-angle light source from below, upward illumination";
            else if (e >= -10 && e < 20) elev_desc = "horizontal level light source";
            else if (e >= 20 && e < 60) elev_desc = "high-angle light source";
            else elev_desc = "overhead top-down light source";

            let int_desc;
            if (intensity < 3.0) int_desc = "soft";
            else if (intensity < 7.0) int_desc = "bright";
            else int_desc = "intense";

            const global_constraints = "SCENE LOCK, FIXED VIEWPOINT, maintaining character consistency and pose. RELIGHTING ONLY: ";
            const color_desc = "colored light (" + normalizeHex(lightColor) + ")";
            return global_constraints + pos_desc + ", " + elev_desc + ", " + int_desc + " " + color_desc + ", cinematic relighting";
        }

        window.addEventListener("message", function(event) {
            const iframe = document.getElementById("qwen_anglelight_iframe");
            if (!iframe || event.source !== iframe.contentWindow) return;
            if (event.data && event.data.type === "ANGLE_UPDATE") {
                const prompt = generateLightingPrompt(
                    event.data.horizontal,
                    event.data.vertical,
                    event.data.zoom,
                    event.data.lightColor
                );
                updateGradioInput("scene_additional_prompt_2", prompt);
            }
        });
    })()'>
    """

    return iframe_code + glue_js

