(function() {
    // ==================== 配置常量 ====================
    const CHECK_INTERVAL = 2000;         // 检测间隔毫秒
    const MAX_RETRY_COUNT = 3;

    // ==================== 状态管理 ====================
    let state = {
        initialTimestamp: null,
        currentQueueSize: 0,
        retryCount: 0,
        isConnected: false,
        currentTheme: 'light',
        isDragging: false,
        offsetX: 0,
        offsetY: 0,
        hasAdminAPI: false,
        initialPositionMoved: false, // 新增初始位置标记
        isReconnectVisible: false    // 是否显示重连按钮
    };

    // ==================== 移动设备检测 ====================
    function isMobileDevice() {
        const userAgent = navigator.userAgent || navigator.vendor || window.opera;
        // 检测常见的移动设备标识符
        return /android|iphone|ipad|ipod|blackberry|iemobile|opera mini/i.test(userAgent.toLowerCase());
    }

    // 如果是移动设备，则直接退出，不执行后续逻辑
    if (isMobileDevice()) {
        console.log("当前设备为移动设备，状态监控组件不显示。");
        // return;
    }

    // ==================== DOM 元素创建 ====================
    const statusContainer = document.createElement('div');
    statusContainer.id = 'gradio-status-monitor';

    const statusIndicator = document.createElement('div');
    statusIndicator.className = 'status-indicator';

    const backToAdminBtn = document.createElement('button');
    backToAdminBtn.className = 'back-to-admin-btn';
    backToAdminBtn.textContent = '返回管理窗口';
    backToAdminBtn.onclick = () => {
        if (window.pywebview && window.pywebview.api && window.pywebview.api.switchToAdmin) {
            window.pywebview.api.switchToAdmin();
        }
    };

    const reconnectBtn = document.createElement('button');
    reconnectBtn.className = 'reconnect-btn';
    reconnectBtn.textContent = ' 重连';
    reconnectBtn.onclick = () => window.location.reload();

    // VRAM 占用百分比
    const vramUsage = document.createElement('div');
    vramUsage.className = 'vram-usage';

    // RAM 占用百分比
    const ramUsage = document.createElement('div');
    ramUsage.className = 'ram-usage';

    // 同时在线用户数
    const onlineUsersBadge = document.createElement('div');
    onlineUsersBadge.className = 'online-users-badge';

    // 同时在线节点数
    const onlineNodesBadge = document.createElement('div');
    onlineNodesBadge.className = 'online-nodes-badge';

    const statusContent = document.createElement('div');
    statusContent.className = 'status-content';

    const transferToggleBtn = document.createElement('button');
    transferToggleBtn.className = 'transfer-toggle';
    transferToggleBtn.type = 'button';
    transferToggleBtn.textContent = '图片中转站 ▾';

    const transferPanel = document.createElement('div');
    transferPanel.className = 'transfer-panel';

    const transferPanelActions = document.createElement('div');
    transferPanelActions.className = 'transfer-panel-actions';

    const transferPasteBtn = document.createElement('button');
    transferPasteBtn.className = 'transfer-panel-btn transfer-paste-btn';
    transferPasteBtn.type = 'button';
    transferPasteBtn.textContent = '粘贴';

    const transferCopyBtn = document.createElement('button');
    transferCopyBtn.className = 'transfer-panel-btn transfer-copy-btn';
    transferCopyBtn.type = 'button';
    transferCopyBtn.textContent = '复制';

    const transferClearBtn = document.createElement('button');
    transferClearBtn.className = 'transfer-panel-btn transfer-clear-btn';
    transferClearBtn.type = 'button';
    transferClearBtn.textContent = '清空';

    const transferHint = document.createElement('div');
    transferHint.className = 'transfer-hint';
    transferHint.textContent = '拖放图片到这里';

    const transferList = document.createElement('div');
    transferList.className = 'transfer-list';

    const transferState = {
        items: [],
        selectedId: null,
        expanded: false,
        nextId: 1,
        hintOverride: null,
        hintOverrideUntil: 0,
        hintTimer: null,
        pasteOverlayEl: null,
        pasteBoxEl: null,
        pasteCloseBtnEl: null,
        pasteHandler: null,
        keydownHandler: null,
        directPasteInited: false,
        directPasteHandler: null
    };

    transferState.nextId = Math.floor(Date.now() * 1000 + Math.random() * 1000);

    const transferSync = {
        tabId: (typeof crypto !== 'undefined' && crypto.randomUUID) ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(16).slice(2)}`,
        channel: null,
        supported: false,
        inited: false,
        suppress: false
    };

    function postTransferSyncMessage(payload) {
        if (!transferSync.supported || !transferSync.channel || transferSync.suppress) return;
        try {
            transferSync.channel.postMessage(Object.assign({ senderId: transferSync.tabId }, payload || {}));
        } catch (e) {
        }
    }

    function setTransferExpanded(expanded, sync) {
        transferState.expanded = !!expanded;
        statusIndicator.classList.toggle('transfer-expanded', transferState.expanded);
        transferToggleBtn.textContent = transferState.expanded ? '图片中转站 ▴' : '图片中转站 ▾';
        if (!transferState.expanded) closeTransferPasteOverlay();
        requestAnimationFrame(updateTransferPanelLayout);
        if (sync) postTransferSyncMessage({ kind: 'transfer_expand', expanded: transferState.expanded });
    }

    async function createTransferPreviewUrl(blob) {
        const thumbBlob = await createThumbnailBlobFromBlob(blob, 160);
        const previewBlob = thumbBlob || blob;
        return URL.createObjectURL(previewBlob);
    }

    async function applyRemoteTransferState(snapshot) {
        if (!snapshot || typeof snapshot !== 'object') return;
        const remoteItems = Array.isArray(snapshot.items) ? snapshot.items : [];
        const remoteSelectedId = snapshot.selectedId ?? null;
        const remoteExpanded = !!snapshot.expanded;

        transferSync.suppress = true;
        try {
            const wasEmpty = !transferState.items.length;
            if (wasEmpty) {
                setTransferExpanded(remoteExpanded, false);
            }

            for (const remote of remoteItems) {
                if (!remote || !remote.blob) continue;
                const id = typeof remote.id === 'number' ? remote.id : Number(remote.id);
                if (!Number.isFinite(id)) continue;
                if (transferState.items.some(x => x.id === id)) continue;
                const blob = remote.blob;
                const type = remote.type || blob.type || 'image/png';
                const name = remote.name || `image_${Date.now()}.png`;
                const previewUrl = await createTransferPreviewUrl(blob);
                transferState.items.unshift({ id, blob, type, name, previewUrl });
                if (!transferState.selectedId) transferState.selectedId = id;
                transferState.nextId = Math.max(transferState.nextId, id + 1);
            }

            if (wasEmpty && remoteSelectedId !== null) {
                const id = typeof remoteSelectedId === 'number' ? remoteSelectedId : Number(remoteSelectedId);
                if (Number.isFinite(id) && transferState.items.some(x => x.id === id)) {
                    transferState.selectedId = id;
                }
            }
            if (!transferState.selectedId && transferState.items.length) {
                transferState.selectedId = transferState.items[0].id;
            }
            if (!transferState.items.length) transferState.selectedId = null;

            renderTransferGrid();
        } finally {
            transferSync.suppress = false;
        }
    }

    async function applyRemoteTransferAdd(remote) {
        if (!remote || !remote.blob) return;
        const id = typeof remote.id === 'number' ? remote.id : Number(remote.id);
        if (!Number.isFinite(id)) return;
        if (transferState.items.some(x => x.id === id)) return;

        transferSync.suppress = true;
        try {
            const blob = remote.blob;
            const type = remote.type || blob.type || 'image/png';
            const name = remote.name || `image_${Date.now()}.png`;
            const previewUrl = await createTransferPreviewUrl(blob);
            transferState.items.unshift({ id, blob, type, name, previewUrl });
            if (!transferState.selectedId) transferState.selectedId = id;
            transferState.nextId = Math.max(transferState.nextId, id + 1);
            renderTransferGrid();
        } finally {
            transferSync.suppress = false;
        }
    }

    function applyRemoteTransferRemove(idRaw) {
        const id = typeof idRaw === 'number' ? idRaw : Number(idRaw);
        if (!Number.isFinite(id)) return;
        if (!transferState.items.some(x => x.id === id)) return;

        transferSync.suppress = true;
        try {
            removeTransferItem(id);
        } finally {
            transferSync.suppress = false;
        }
    }

    function applyRemoteTransferClear() {
        transferSync.suppress = true;
        try {
            clearTransferItems();
        } finally {
            transferSync.suppress = false;
        }
    }

    function applyRemoteTransferSelect(idRaw) {
        const id = typeof idRaw === 'number' ? idRaw : Number(idRaw);
        if (!Number.isFinite(id)) return;
        if (!transferState.items.some(x => x.id === id)) return;

        transferSync.suppress = true;
        try {
            transferState.selectedId = id;
            renderTransferGrid();
        } finally {
            transferSync.suppress = false;
        }
    }

    function applyRemoteTransferExpanded(expandedRaw) {
        const expanded = !!expandedRaw;
        transferSync.suppress = true;
        try {
            setTransferExpanded(expanded, false);
        } finally {
            transferSync.suppress = false;
        }
    }

    function initTransferCrossTabSync() {
        if (transferSync.inited) return;
        transferSync.inited = true;

        if (!('BroadcastChannel' in window)) return;
        try {
            transferSync.channel = new BroadcastChannel('simpleai-transfer-station-v1');
            transferSync.supported = true;
        } catch (e) {
            transferSync.supported = false;
            transferSync.channel = null;
            return;
        }

        transferSync.channel.addEventListener('message', (evt) => {
            const data = evt ? evt.data : null;
            if (!data || typeof data !== 'object') return;
            if (data.senderId && data.senderId === transferSync.tabId) return;

            const kind = data.kind;
            if (kind === 'transfer_state_request') {
                const requestId = data.requestId;
                if (!requestId) return;
                postTransferSyncMessage({
                    kind: 'transfer_state',
                    requestId,
                    snapshot: {
                        items: transferState.items.filter(x => x && x.blob).map(x => ({ id: x.id, blob: x.blob, type: x.type, name: x.name })),
                        selectedId: transferState.selectedId,
                        expanded: transferState.expanded
                    }
                });
                return;
            }
            if (kind === 'transfer_state') {
                const snapshot = data.snapshot;
                applyRemoteTransferState(snapshot);
                return;
            }
            if (kind === 'transfer_add') {
                applyRemoteTransferAdd(data.item);
                return;
            }
            if (kind === 'transfer_remove') {
                applyRemoteTransferRemove(data.id);
                return;
            }
            if (kind === 'transfer_clear') {
                applyRemoteTransferClear();
                return;
            }
            if (kind === 'transfer_select') {
                applyRemoteTransferSelect(data.id);
                return;
            }
            if (kind === 'transfer_expand') {
                applyRemoteTransferExpanded(data.expanded);
                return;
            }
        });

        const requestState = () => {
            const requestId = `${transferSync.tabId}:${Date.now()}:${Math.random().toString(16).slice(2)}`;
            postTransferSyncMessage({ kind: 'transfer_state_request', requestId });
        };

        requestState();
        document.addEventListener('visibilitychange', () => {
            if (document.visibilityState === 'visible') requestState();
        });
    }

    // ==================== 样式配置 ====================
    const style = document.createElement('style');
    style.textContent = `
        #gradio-status-monitor {
            position: fixed;
            top: 8px;
            right: 3px;
            z-index: 2147483647;
            font-family: Arial, sans-serif;
            background: transparent; /* 设置背景为透明 */
            pointer-events: auto; /* 修改为auto以支持拖拽 */
            cursor: grab; /* 显示可拖拽的手型光标 */
            transition: all 0.2s ease;
        }
        
        #gradio-status-monitor.dragging {
            cursor: grabbing; /* 拖拽时显示抓取状态的光标 */
            opacity: 0.8;
            transition: none !important; /* 拖拽时禁用所有过渡 */
        }

        /* 亮色主题 */
        .status-indicator.light {
            padding: 4px 8px;
            border-radius: 3px;
            display: flex;
            flex-direction: column;
            align-items: flex-end; /* 右对齐 */
            font-size: 12px;
            position: relative;
	    background: transparent; /* 设置背景为透明 */
            border: none;
            color: #333;
	    box-shadow: none; /* 移除阴影 */
        }
        .light .status-connected { color: #2c7a2c; }
        .light .status-disconnected { color: #c53030; }
        .light .status-exception { color: #d97706; }

        /* 暗色主题 */
        .status-indicator.dark {
            padding: 4px 8px;
            border-radius: 3px;
            display: flex;
            flex-direction: column;
            align-items: flex-end; /* 右对齐 */
            font-size: 12px;
            position: relative;
	    background: transparent; /* 设置背景为透明 */
            border-color: #4a5568;
            color: #fff;
	    border: none;
	    box-shadow: none; /* 移除阴影 */
        }
        .dark .status-connected { color: #48bb78; }
        .dark .status-disconnected { color: #f56565; }
        .dark .status-exception { color: #ecc94b; }

        /* 第一行样式 */
        .queue-badge {
            margin-left: 6px;
            padding: 1px 6px;
            border-radius: 8px;
            font-size: 11px;
        }
        .light .queue-badge { background: #f0f0f0; }
        .dark .queue-badge { background: #2d3748; }

        /* 重连按钮样式 */
        .reconnect-btn {
	    margin-left: 8px;
            padding: 2px 8px;
            border-radius: 3px;
            cursor: pointer;
            font-size: 11px;
	    pointer-events: auto;
        }
        .light .reconnect-btn {
            border: 1px solid #c53030;
            background: #fff0f0;
            color: #c53030;
        }
        .dark .reconnect-btn {
            border: 1px solid #f56565;
            background: #2d1a1a;
            color: #f56565;
        }

        .vram-usage, .ram-usage, .online-users-badge, .online-nodes-badge {
            margin-top: 4px;
            font-size: 11px;
        }
	    .light .vram-usage {
            background: #f0f0f0;
            padding: 1px 6px;
            border-radius: 8px;
            color: var(--neutral-700);
        }
        .light .ram-usage, .light .online-users-badge, .light .online-nodes-badge {
            background: #f0f0f0;
            padding: 1px 6px;
            border-radius: 8px;
	    color: var(--neutral-400);
        }
	    .dark .vram-usage {
            background: #2d3748;
            padding: 1px 6px;
            border-radius: 8px;
            color: var(--neutral-300);
        }
        .dark .ram-usage, .dark .online-users-badge, .dark .online-nodes-badge {
            background: #2d3748;
            padding: 1px 6px;
            border-radius: 8px;
	    color: var(--neutral-500);
        }
        .back-to-admin-btn {
            margin-left: 8px;
            padding: 2px 8px;
            border-radius: 3px;
            cursor: pointer;
            font-size: 11px;
            pointer-events: auto;
        }
        .light .back-to-admin-btn {
            border: 1px solid #4a90e2;
            background: #f0f8ff;
            color: #4a90e2;
        }
        .dark .back-to-admin-btn {
            border: 1px solid aqua;
            background: #1a2a3a;
            color: aqua;
        }

        .status-indicator .status-content {
            display: flex;
            flex-direction: column;
            align-items: flex-end;
        }

        .status-indicator .transfer-toggle {
            margin-top: 6px;
            font-size: 11px;
            padding: 2px 8px;
            border-radius: 10px;
            cursor: pointer;
            border: 1px solid transparent;
            background: transparent;
            color: inherit;
            align-self: flex-end;
            pointer-events: auto;
        }

        .status-indicator.light .transfer-toggle {
            border-color: rgba(0, 0, 0, 0.12);
            background: rgba(0, 0, 0, 0.04);
        }

        .status-indicator.dark .transfer-toggle {
            border-color: rgba(255, 255, 255, 0.18);
            background: rgba(81, 125, 255, 0.5);
        }

        .status-indicator .transfer-panel {
            display: none;
            position: absolute;
            right: 0;
            top: calc(100% + 6px);
            width: max(112px, 100%);
            max-width: min(320px, calc(100vw - 16px));
            max-height: calc(100vh - 16px);
            border-radius: 8px;
            overflow: hidden;
            user-select: none;
            pointer-events: auto;
            box-sizing: border-box;
            flex-direction: column;
            cursor: default;
        }

        .status-indicator.transfer-expanded .transfer-panel {
            display: flex;
        }

        .status-indicator.light .transfer-panel {
            background: rgba(255, 255, 255, 0.995);
            border: 1px solid rgba(0, 0, 0, 0.16);
            color: #333;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.18);
        }

        .status-indicator.dark .transfer-panel {
            background: rgba(12, 14, 18, 0.995);
            border: 1px solid rgba(255, 255, 255, 0.18);
            color: #fff;
            box-shadow: 0 12px 34px rgba(0, 0, 0, 0.62);
        }

        @supports ((-webkit-backdrop-filter: blur(6px)) or (backdrop-filter: blur(6px))) {
            .status-indicator.light .transfer-panel {
                background: rgba(255, 255, 255, 0.92);
                -webkit-backdrop-filter: blur(8px);
                backdrop-filter: blur(8px);
            }

            .status-indicator.dark .transfer-panel {
                background: rgba(12, 14, 18, 0.92);
                -webkit-backdrop-filter: blur(8px);
                backdrop-filter: blur(8px);
            }
        }

        .status-indicator .transfer-panel-actions {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            padding: 6px 8px;
            justify-content: flex-start;
            width: 100%;
            box-sizing: border-box;
            cursor: default;
        }

        .status-indicator .transfer-panel-btn {
            font-size: 11px;
            padding: 2px 8px;
            border-radius: 6px;
            cursor: pointer;
            border: 1px solid transparent;
            background: transparent;
            color: inherit;
        }

        .status-indicator.light .transfer-panel-btn {
            border-color: rgba(0, 0, 0, 0.12);
        }

        .status-indicator.dark .transfer-panel-btn {
            border-color: rgba(255, 255, 255, 0.18);
        }

        .status-indicator .transfer-hint {
            font-size: 11px;
            opacity: 0.75;
            padding: 0 8px 8px 8px;
            text-align: center;
            cursor: default;
        }

        .status-indicator .transfer-list {
            display: flex;
            flex-direction: column;
            gap: 6px;
            overflow: auto;
            padding: 0 8px 8px 8px;
            align-items: stretch;
            flex: 1 1 auto;
            min-height: 0;
            cursor: default;
        }

        .status-indicator .transfer-panel.two-col .transfer-list {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            align-items: stretch;
        }

        .status-indicator .transfer-item {
            position: relative;
            width: 100%;
            aspect-ratio: 1 / 1;
            border-radius: 8px;
            overflow: hidden;
            border: 2px solid transparent;
            background: rgba(0,0,0,0.06);
            cursor: pointer;
        }

        .status-indicator.dark .transfer-item {
            background: rgba(255,255,255,0.06);
        }

        .status-indicator .transfer-item.selected {
            border-color: rgba(76, 139, 245, 0.95);
        }

        .status-indicator .transfer-item img {
            width: 100%;
            height: 100%;
            object-fit: contain;
            display: block;
            user-drag: none;
            -webkit-user-drag: none;
        }

        @supports not (aspect-ratio: 1 / 1) {
            .status-indicator .transfer-item {
                height: 0;
                padding-bottom: 100%;
            }

            .status-indicator .transfer-item img {
                position: absolute;
                top: 0;
                left: 0;
            }
        }

        .status-indicator .transfer-item .transfer-remove {
            position: absolute;
            top: 4px;
            right: 4px;
            width: 18px;
            height: 18px;
            border-radius: 9px;
            border: 1px solid rgba(255,255,255,0.25);
            background: rgba(0,0,0,0.55);
            color: #fff;
            font-size: 12px;
            line-height: 16px;
            cursor: pointer;
            display: none;
            align-items: center;
            justify-content: center;
            padding: 0;
        }

        .status-indicator .transfer-item:hover .transfer-remove {
            display: flex;
        }

        .status-indicator .transfer-panel.dragover {
            outline: 2px dashed rgba(76, 139, 245, 0.9);
            outline-offset: -2px;
        }

        .status-indicator .transfer-paste-overlay {
            position: absolute;
            inset: 0;
            background: rgba(0, 0, 0, 0.55);
            display: none;
            align-items: center;
            justify-content: center;
            padding: 10px;
            box-sizing: border-box;
            z-index: 2;
        }

        .status-indicator .transfer-paste-overlay.open {
            display: flex;
        }

        .status-indicator .transfer-paste-box {
            width: 100%;
            border-radius: 10px;
            padding: 10px;
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
            gap: 8px;
            outline: none;
        }

        .status-indicator.light .transfer-paste-box {
            background: rgba(255, 255, 255, 0.96);
            color: #333;
            border: 1px solid rgba(0, 0, 0, 0.12);
        }

        .status-indicator.dark .transfer-paste-box {
            background: rgba(20, 24, 28, 0.96);
            color: #fff;
            border: 1px solid rgba(255, 255, 255, 0.16);
        }

        .status-indicator .transfer-paste-box-title {
            font-size: 12px;
            font-weight: 600;
        }

        .status-indicator .transfer-paste-box-subtitle {
            font-size: 11px;
            opacity: 0.8;
            line-height: 1.3;
        }

        .status-indicator .transfer-paste-box-actions {
            display: flex;
            justify-content: flex-end;
            gap: 6px;
        }

        .status-indicator .transfer-paste-close {
            font-size: 11px;
            padding: 2px 8px;
            border-radius: 6px;
            cursor: pointer;
            border: 1px solid transparent;
            background: transparent;
            color: inherit;
        }

        .status-indicator.light .transfer-paste-close {
            border-color: rgba(0, 0, 0, 0.12);
        }

        .status-indicator.dark .transfer-paste-close {
            border-color: rgba(255, 255, 255, 0.18);
        }
    `;

    // ==================== 主题管理 ====================
    function detectTheme() {
        const params = new URLSearchParams(window.location.search);
        return params.get('__theme') || 'light';
    }

    function applyTheme() {
        statusIndicator.classList.remove('light', 'dark');
        statusIndicator.classList.add(state.currentTheme);
    }

    // ==================== 核心功能 ====================
    async function fetchAppStatus() {
        try {
            const response = await fetch('/run/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    fn_index: 0,
                    data: []
                })
            });

            if (!response.ok) throw new Error('请求失败');

            const result = await response.json();
            const [timestampStr, queueSizeStr, vramTotalStr, ramTotalStr, vramUsedStr, ramUsedStr, onlineUsersStr, onlineDomainUsersStr, onlineNodesStr] = result.data[0].split(',');

            return {
                timestamp: parseFloat(timestampStr),
                queueSize: parseInt(queueSizeStr),
                ramUsed: parseInt(ramUsedStr),
                ramTotal: parseInt(ramTotalStr),
                vramUsed: parseInt(vramUsedStr),
                vramTotal: parseInt(vramTotalStr),
                onlineUsers: parseInt(onlineUsersStr),
		        onlineDomainUsers: parseInt(onlineDomainUsersStr),
		        onlineNodes: parseInt(onlineNodesStr),
            };
        } catch (error) {
            return null;
        }
    }

    function updateStatusUI(statusType, queueSize, ramUsed, ramTotal, vramUsed, vramTotal, onlineUsers, onlineDomainUsers, onlineNodes) {
        statusContent.innerHTML = '';
        state.isReconnectVisible = (statusType === 'exception' || statusType === 'disconnected');
        
        if (state.hasAdminAPI) {
            statusContent.appendChild(backToAdminBtn);
        }

        const statusMap = {
            connected: { text: '连接', class: 'status-connected' },
            disconnected: { text: '断开', class: 'status-disconnected' },
            exception: { text: '异常', class: 'status-exception' }
        };
        const { text, class: statusClass } = statusMap[statusType];

        // 构建状态指示
        const statusEl = document.createElement('span');
        statusEl.className = statusClass;
        statusEl.innerHTML = `● ${text}`;

        // 队列数与状态显示在同一行
        const firstRow = document.createElement('div');
        firstRow.style.display = 'flex';
        firstRow.style.alignItems = 'center';
        firstRow.appendChild(statusEl);

        if (statusType === 'connected') {
            const queueBadge = document.createElement('span');
            queueBadge.className = 'queue-badge';
            queueBadge.textContent = `队列: ${queueSize}`;
            firstRow.appendChild(queueBadge);
        } else if (statusType === 'exception' || statusType === 'disconnected') {
            firstRow.appendChild(reconnectBtn);
            if (statusType === 'disconnected') {
                const retryText = document.createElement('span');
                retryText.textContent = ` (${state.retryCount * CHECK_INTERVAL / 1000}s)`;
                firstRow.appendChild(retryText);
            }
        }
	    
        statusContent.appendChild(firstRow);

        // 添加附加信息
        if (statusType === 'connected' && !isMobileDevice()) {
	    // 显示 VRAM 使用情况
            const vramPercent = ((vramUsed / vramTotal) * 100).toFixed(1);
            vramUsage.textContent = `显存: ${vramPercent}%`;
            statusContent.appendChild(vramUsage);

            // 显示 RAM 使用情况
            const ramPercent = ((ramUsed / ramTotal) * 100).toFixed(1);
            ramUsage.textContent = `内存: ${ramPercent}%`;
            statusContent.appendChild(ramUsage);

            // 显示在线用户数
            if (onlineDomainUsers===0) {
		onlineUsersBadge.textContent = `用户: ${onlineUsers}`;
	    } else {
		onlineUsersBadge.textContent = `用户: ${onlineUsers}/${onlineDomainUsers}`;
	    }
            statusContent.appendChild(onlineUsersBadge);

	    // 显示在线节点数
	    if (onlineNodes!=0) {
                onlineNodesBadge.textContent = `节点: ${onlineNodes}`;
                statusContent.appendChild(onlineNodesBadge);
	    }
        }
    }

    async function performHealthCheck() {
        checkAdminAPIAvailability();
        const statusData = await fetchAppStatus();

        if (!statusData) {
            state.retryCount++;
            if (state.retryCount >= MAX_RETRY_COUNT) {
                updateStatusUI('disconnected');
            }
            return;
        }

        state.retryCount = 0;

        if (!state.initialTimestamp) {
            state.initialTimestamp = statusData.timestamp;
            state.isConnected = true;
        }

        if (statusData.timestamp === state.initialTimestamp) {
            updateStatusUI(
                'connected',
                statusData.queueSize,
                statusData.ramUsed,
                statusData.ramTotal,
                statusData.vramUsed,
                statusData.vramTotal,
                statusData.onlineUsers,
		        statusData.onlineDomainUsers,
		        statusData.onlineNodes,
            );
        } else {
            updateStatusUI('exception');
        }
    }

    // ==================== 浏览器检测 ====================
    function detectBrowser() {
        const userAgent = navigator.userAgent.toLowerCase();
        if (userAgent.indexOf('chrome') > -1) return 'chrome';
        if (userAgent.indexOf('safari') > -1 && userAgent.indexOf('chrome') === -1) return 'safari';
        if (userAgent.indexOf('firefox') > -1) return 'firefox';
        if (userAgent.indexOf('edge') > -1) return 'edge';
        return 'unknown';
    }

    // ==================== 拖拽功能 ====================
    function initDragFeature() {
        const browser = detectBrowser();
        const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
        
        // 标准鼠标拖拽事件
        statusContainer.addEventListener('mousedown', startDrag);
        
        // 触摸设备支持
        statusContainer.addEventListener('touchstart', startTouchDrag, { passive: false });
        
        // 为 Chrome 和 Edge 在 Mac 上添加 Pointer Events 支持
        if ((browser === 'chrome' || browser === 'edge') && isMac) {
            statusContainer.addEventListener('pointerdown', startPointerDrag);
        }
        
        function startDrag(e) {
            // 只响应左键 (button === 0)
            if (e.button === 0) {
                if (e.target && e.target.closest && (e.target.closest('.transfer-panel') || e.target.closest('.transfer-toggle') || e.target.closest('.transfer-item') || e.target.closest('button'))) return;
                e.preventDefault();
                state.isDragging = true;
                
                // 获取当前位置
                const rect = statusContainer.getBoundingClientRect();
                state.offsetX = e.clientX - rect.left;
                state.offsetY = e.clientY - rect.top;
                
                statusContainer.classList.add('dragging');
                
                // 添加临时事件监听器
                document.addEventListener('mousemove', doDrag);
                document.addEventListener('mouseup', stopDrag);
            }
        }
        
        function startPointerDrag(e) {
            // 只响应主指针（通常是左键或触控板点击）
            if (e.isPrimary && (e.pointerType === 'mouse' || e.pointerType === 'touch')) {
                if (e.target && e.target.closest && (e.target.closest('.transfer-panel') || e.target.closest('.transfer-toggle') || e.target.closest('.transfer-item') || e.target.closest('button'))) return;
                e.preventDefault();
                state.isDragging = true;
                
                // 获取当前位置
                const rect = statusContainer.getBoundingClientRect();
                state.offsetX = e.clientX - rect.left;
                state.offsetY = e.clientY - rect.top;
                
                statusContainer.classList.add('dragging');
                
                // 添加临时事件监听器
                document.addEventListener('pointermove', doPointerDrag);
                document.addEventListener('pointerup', stopPointerDrag);
                document.addEventListener('pointercancel', stopPointerDrag);
            }
        }
        
        function startTouchDrag(e) {
            if (e.touches && e.touches.length === 1) {
                if (e.target && e.target.closest && (e.target.closest('.transfer-panel') || e.target.closest('.transfer-toggle') || e.target.closest('.transfer-item') || e.target.closest('button'))) return;
                e.preventDefault();
                state.isDragging = true;
                
                // 获取当前位置
                const rect = statusContainer.getBoundingClientRect();
                state.offsetX = e.touches[0].clientX - rect.left;
                state.offsetY = e.touches[0].clientY - rect.top;
                
                statusContainer.classList.add('dragging');
                
                // 添加临时事件监听器
                document.addEventListener('touchmove', doTouchDrag, { passive: false });
                document.addEventListener('touchend', stopTouchDrag);
                document.addEventListener('touchcancel', stopTouchDrag);
            }
        }
        
        function doDrag(e) {
            if (state.isDragging) {
                e.preventDefault();
                moveElement(e.clientX, e.clientY);
            }
        }
        
        function doPointerDrag(e) {
            if (state.isDragging) {
                e.preventDefault();
                moveElement(e.clientX, e.clientY);
            }
        }
        
        function doTouchDrag(e) {
            if (state.isDragging && e.touches && e.touches.length === 1) {
                e.preventDefault(); // 阻止页面滚动
                moveElement(e.touches[0].clientX, e.touches[0].clientY);
            }
        }

        function stopDrag() {
            if (state.isDragging) {
                state.isDragging = false;
                statusContainer.classList.remove('dragging');
                
                // 移除临时事件监听器
                document.removeEventListener('mousemove', doDrag);
                document.removeEventListener('mouseup', stopDrag);
                if (transferState && transferState.expanded) requestAnimationFrame(updateTransferPanelLayout);
            }
        }
        
        function stopPointerDrag() {
            if (state.isDragging) {
                state.isDragging = false;
                statusContainer.classList.remove('dragging');
                
                // 移除临时事件监听器
                document.removeEventListener('pointermove', doPointerDrag);
                document.removeEventListener('pointerup', stopPointerDrag);
                document.removeEventListener('pointercancel', stopPointerDrag);
                if (transferState && transferState.expanded) requestAnimationFrame(updateTransferPanelLayout);
            }
        }
        
        function stopTouchDrag() {
            if (state.isDragging) {
                state.isDragging = false;
                statusContainer.classList.remove('dragging');
                
                // 移除临时事件监听器
                document.removeEventListener('touchmove', doTouchDrag);
                document.removeEventListener('touchend', stopTouchDrag);
                document.removeEventListener('touchcancel', stopTouchDrag);
                if (transferState && transferState.expanded) requestAnimationFrame(updateTransferPanelLayout);
            }
        }
    }
    // 统一移动元素的函数，增加边界保护
    function moveElement(clientX, clientY) {
        // 计算新位置
        const newLeft = clientX - state.offsetX;
        const newTop = clientY - state.offsetY;

        // 获取元素实际尺寸
        const rect = statusContainer.getBoundingClientRect();
        const elementWidth = rect.width;
        const elementHeight = rect.height;

        // 确保不超出视口边界，并留出余量防止变形
        const safeMargin = 3; // 安全边距，防止元素变形
        const maxX = window.innerWidth - elementWidth - safeMargin;
        const maxY = window.innerHeight - elementHeight - safeMargin;

        statusContainer.style.left = `${Math.max(safeMargin, Math.min(maxX, newLeft))}px`;
        statusContainer.style.top = `${Math.max(safeMargin, Math.min(maxY, newTop))}px`;
        statusContainer.style.right = 'auto'; // 取消右侧定位
        statusContainer.style.bottom = 'auto'; // 取消底部定位
    }

    function clamp(n, min, max) {
        return Math.max(min, Math.min(max, n));
    }

    function readFileAsDataUrl(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onerror = () => reject(new Error('read_failed'));
            reader.onload = () => resolve(reader.result);
            reader.readAsDataURL(file);
        });
    }

    function fileFromBlob(blob, filename, fallbackType) {
        const type = blob && blob.type ? blob.type : (fallbackType || 'image/png');
        const ext = type === 'image/jpeg' ? 'jpg' : (String(type).split('/')[1] || 'png');
        const name = filename || `transfer_${Date.now()}.${ext}`;
        return new File([blob], name, { type });
    }

    async function urlToBlob(url) {
        const res = await fetch(url, { mode: 'cors' });
        return await res.blob();
    }

    async function createThumbnailBlobFromBlob(blob, maxSide) {
        const target = typeof maxSide === 'number' ? maxSide : 160;
        try {
            const bitmap = await createImageBitmap(blob);
            const w = bitmap.width || 1;
            const h = bitmap.height || 1;
            const scale = Math.min(1, target / Math.max(w, h));
            const tw = Math.max(1, Math.round(w * scale));
            const th = Math.max(1, Math.round(h * scale));
            const canvas = document.createElement('canvas');
            canvas.width = tw;
            canvas.height = th;
            const ctx = canvas.getContext('2d');
            if (!ctx) return null;
            ctx.drawImage(bitmap, 0, 0, tw, th);
            if (bitmap && bitmap.close) bitmap.close();
            const thumb = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.86));
            return thumb || null;
        } catch (e) {
            return null;
        }
    }

    async function convertBlobToPng(blob) {
        try {
            const bitmap = await createImageBitmap(blob);
            const canvas = document.createElement('canvas');
            canvas.width = Math.max(1, bitmap.width || 1);
            canvas.height = Math.max(1, bitmap.height || 1);
            const ctx = canvas.getContext('2d');
            if (!ctx) return null;
            ctx.drawImage(bitmap, 0, 0);
            if (bitmap && bitmap.close) bitmap.close();
            const out = await new Promise((resolve) => canvas.toBlob(resolve, 'image/png'));
            return out || null;
        } catch (e) {
            return null;
        }
    }

    async function blobToDataUrl(blob) {
        try {
            return await new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onerror = () => reject(new Error('read failed'));
                reader.onload = () => resolve(String(reader.result || ''));
                reader.readAsDataURL(blob);
            });
        } catch (e) {
            return '';
        }
    }

    function setTransferHintMessage(message, durationMs) {
        const text = String(message || '').trim();
        const ms = typeof durationMs === 'number' ? durationMs : 1600;
        transferState.hintOverride = text || null;
        transferState.hintOverrideUntil = text ? (Date.now() + Math.max(250, ms)) : 0;
        if (transferState.hintTimer) {
            clearTimeout(transferState.hintTimer);
            transferState.hintTimer = null;
        }
        if (text) {
            transferState.hintTimer = setTimeout(() => {
                transferState.hintOverride = null;
                transferState.hintOverrideUntil = 0;
                transferState.hintTimer = null;
                renderTransferGrid();
            }, Math.max(250, ms));
        }
        renderTransferGrid();
    }

    function ensureTransferPasteOverlay() {
        if (transferState.pasteOverlayEl && transferState.pasteBoxEl && transferState.pasteCloseBtnEl) return;

        const overlay = document.createElement('div');
        overlay.className = 'transfer-paste-overlay';

        const box = document.createElement('div');
        box.className = 'transfer-paste-box';
        box.tabIndex = 0;

        const title = document.createElement('div');
        title.className = 'transfer-paste-box-title';
        title.textContent = '粘贴图片';

        const subtitle = document.createElement('div');
        subtitle.className = 'transfer-paste-box-subtitle';
        subtitle.textContent = '请按 Ctrl+V';

        const actions = document.createElement('div');
        actions.className = 'transfer-paste-box-actions';

        const closeBtn = document.createElement('button');
        closeBtn.type = 'button';
        closeBtn.className = 'transfer-paste-close';
        closeBtn.textContent = '关闭';

        actions.appendChild(closeBtn);
        box.appendChild(title);
        box.appendChild(subtitle);
        box.appendChild(actions);
        overlay.appendChild(box);
        transferPanel.appendChild(overlay);

        transferState.pasteOverlayEl = overlay;
        transferState.pasteBoxEl = box;
        transferState.pasteCloseBtnEl = closeBtn;

        closeBtn.addEventListener('click', (e) => {
            e.preventDefault();
            closeTransferPasteOverlay();
        });

        overlay.addEventListener('mousedown', (e) => {
            if (e.target === overlay) closeTransferPasteOverlay();
        });
    }

    function closeTransferPasteOverlay() {
        if (!transferState.pasteOverlayEl) return;
        transferState.pasteOverlayEl.classList.remove('open');

        if (transferState.pasteHandler) {
            document.removeEventListener('paste', transferState.pasteHandler, true);
            transferState.pasteHandler = null;
        }
        if (transferState.keydownHandler) {
            document.removeEventListener('keydown', transferState.keydownHandler, true);
            transferState.keydownHandler = null;
        }
    }

    function openTransferPasteOverlay() {
        ensureTransferPasteOverlay();
        if (!transferState.pasteOverlayEl || !transferState.pasteBoxEl) return;
        transferState.pasteOverlayEl.classList.add('open');
        try { transferState.pasteBoxEl.focus(); } catch (e) {}

        if (!transferState.pasteHandler) {
            transferState.pasteHandler = async (evt) => {
                try {
                    const data = evt && evt.clipboardData ? evt.clipboardData : null;
                    const items = data && data.items ? Array.from(data.items) : [];
                    let added = 0;
                    for (const it of items) {
                        if (!it || !it.type || !String(it.type).startsWith('image/')) continue;
                        const file = it.getAsFile();
                        if (!file) continue;
                        await addTransferFile(file);
                        added++;
                    }
                    if (added > 0) {
                        evt.preventDefault();
                        evt.stopPropagation();
                        if (evt.stopImmediatePropagation) evt.stopImmediatePropagation();
                        setTransferHintMessage(`已粘贴 ${added} 张`);
                        closeTransferPasteOverlay();
                        return;
                    }

                    const text = data && typeof data.getData === 'function' ? String(data.getData('text/plain') || '') : '';
                    if (text && text.trim()) {
                        evt.preventDefault();
                        evt.stopPropagation();
                        if (evt.stopImmediatePropagation) evt.stopImmediatePropagation();
                        await addTransferUrl(text);
                        setTransferHintMessage('已粘贴');
                        closeTransferPasteOverlay();
                        return;
                    }

                    setTransferHintMessage('剪贴板无图片');
                } catch (e) {
                    console.error('Paste capture failed:', e);
                    setTransferHintMessage('粘贴失败');
                }
            };
            document.addEventListener('paste', transferState.pasteHandler, true);
        }

        if (!transferState.keydownHandler) {
            transferState.keydownHandler = (evt) => {
                if (evt && evt.key === 'Escape') closeTransferPasteOverlay();
            };
            document.addEventListener('keydown', transferState.keydownHandler, true);
        }
    }

    function initTransferDirectPaste() {
        if (transferState.directPasteInited) return;
        transferState.directPasteInited = true;

        transferState.directPasteHandler = async (evt) => {
            try {
                if (!transferState.expanded) return;
                if (!evt || evt.defaultPrevented) return;
                if (transferState.pasteOverlayEl && transferState.pasteOverlayEl.classList && transferState.pasteOverlayEl.classList.contains('open')) return;

                const data = evt.clipboardData || null;
                const items = data && data.items ? Array.from(data.items) : [];
                let added = 0;

                for (const it of items) {
                    if (!it || !it.type || !String(it.type).startsWith('image/')) continue;
                    const file = it.getAsFile();
                    if (!file) continue;
                    await addTransferFile(file);
                    added++;
                }

                if (added > 0) {
                    evt.preventDefault();
                    evt.stopPropagation();
                    if (evt.stopImmediatePropagation) evt.stopImmediatePropagation();
                    setTransferHintMessage(`已粘贴 ${added} 张`);
                    closeTransferPasteOverlay();
                }
            } catch (e) {
                console.error('Direct paste failed:', e);
            }
        };

        document.addEventListener('paste', transferState.directPasteHandler, true);
    }

    function renderTransferGrid() {
        transferList.innerHTML = '';
        const now = Date.now();
        const overrideActive = !!(transferState.hintOverride && transferState.hintOverrideUntil && now < transferState.hintOverrideUntil);
        if (overrideActive) {
            transferHint.textContent = transferState.hintOverride;
            transferHint.style.display = 'block';
        } else {
            transferHint.textContent = '拖放图片到这里';
            transferHint.style.display = transferState.items.length ? 'none' : 'block';
        }

        transferPanel.classList.toggle('two-col', transferState.items.length > 12);

        for (const item of transferState.items) {
            const wrap = document.createElement('div');
            wrap.className = 'transfer-item' + (item.id === transferState.selectedId ? ' selected' : '');
            wrap.draggable = true;
            wrap.dataset.transferId = String(item.id);

            const img = document.createElement('img');
            img.src = item.previewUrl;
            img.alt = 'image';

            const removeBtn = document.createElement('button');
            removeBtn.className = 'transfer-remove';
            removeBtn.type = 'button';
            removeBtn.textContent = '×';

            removeBtn.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                removeTransferItem(item.id);
            });

            wrap.addEventListener('click', () => {
                transferState.selectedId = item.id;
                renderTransferGrid();
                postTransferSyncMessage({ kind: 'transfer_select', id: item.id });
            });

            wrap.addEventListener('dragstart', (e) => {
                try {
                    e.dataTransfer.effectAllowed = 'copy';
                    e.dataTransfer.setData('application/x-simpleai-transfer-id', String(item.id));
                    try {
                        if (item.blob && !item.dragUrl) {
                            item.dragUrl = URL.createObjectURL(item.blob);
                        }
                        if (item.dragUrl) {
                            e.dataTransfer.setData('text/uri-list', String(item.dragUrl));
                            e.dataTransfer.setData('text/plain', String(item.dragUrl));
                        } else {
                            e.dataTransfer.setData('text/plain', 'image');
                        }
                    } catch (e3) {
                        e.dataTransfer.setData('text/plain', 'image');
                    }
                    try {
                        if (e.dataTransfer.items && item.blob) {
                            const file = fileFromBlob(item.blob, item.name, item.type);
                            e.dataTransfer.items.add(file);
                        }
                    } catch (e2) {
                    }
                } catch (err) {
                }
            });

            wrap.appendChild(img);
            wrap.appendChild(removeBtn);
            transferList.appendChild(wrap);
        }
    }

    function removeTransferItem(id) {
        for (const item of transferState.items) {
            if (item.id === id) {
                try {
                    if (item.previewUrl) URL.revokeObjectURL(item.previewUrl);
                } catch (e) {
                }
                try {
                    if (item.dragUrl) URL.revokeObjectURL(item.dragUrl);
                } catch (e) {
                }
            }
        }
        transferState.items = transferState.items.filter(x => x.id !== id);
        if (transferState.selectedId === id) {
            transferState.selectedId = transferState.items.length ? transferState.items[0].id : null;
        }
        renderTransferGrid();
        postTransferSyncMessage({ kind: 'transfer_remove', id });
    }

    function clearTransferItems() {
        for (const item of transferState.items) {
            try {
                if (item.previewUrl) URL.revokeObjectURL(item.previewUrl);
            } catch (e) {
            }
            try {
                if (item.dragUrl) URL.revokeObjectURL(item.dragUrl);
            } catch (e) {
            }
        }
        transferState.items = [];
        transferState.selectedId = null;
        renderTransferGrid();
        postTransferSyncMessage({ kind: 'transfer_clear' });
    }

    async function addTransferBlob(blob, filename) {
        if (!blob) return;
        const type = blob.type || 'image/png';
        if (!String(type).startsWith('image/')) return;
        const thumbBlob = await createThumbnailBlobFromBlob(blob, 160);
        const previewBlob = thumbBlob || blob;
        const previewUrl = URL.createObjectURL(previewBlob);

        const item = {
            id: transferState.nextId++,
            blob,
            type,
            name: filename || `image_${Date.now()}.png`,
            previewUrl
        };
        transferState.items.unshift(item);
        if (!transferState.selectedId) transferState.selectedId = transferState.items[0].id;
        renderTransferGrid();
        postTransferSyncMessage({ kind: 'transfer_add', item: { id: item.id, blob: item.blob, type: item.type, name: item.name } });
    }

    async function addTransferFile(file) {
        if (!file || !file.type || !file.type.startsWith('image/')) return;
        await addTransferBlob(file, file.name);
    }

    async function addTransferDataUrl(dataUrl) {
        if (!dataUrl || typeof dataUrl !== 'string') return;
        if (!dataUrl.startsWith('data:image/')) return;
        try {
            const blob = await (await fetch(dataUrl)).blob();
            await addTransferBlob(blob, `image_${Date.now()}.png`);
        } catch (e) {
        }
    }

    async function addTransferUrl(url) {
        if (!url || typeof url !== 'string') return;
        const normalized = url.trim();
        if (!normalized) return;
        try {
            if (normalized.startsWith('data:image/')) {
                await addTransferDataUrl(normalized);
                return;
            }
            const blob = await urlToBlob(normalized);
            await addTransferBlob(blob, `image_${Date.now()}.png`);
        } catch (e) {
        }
    }

    function setFileInputFromFile(fileInput, file) {
        if (!fileInput || !file) return false;
        try {
            const dt = new DataTransfer();
            dt.items.add(file);
            fileInput.files = dt.files;
            fileInput.dispatchEvent(new Event('change', { bubbles: true }));
            fileInput.dispatchEvent(new Event('input', { bubbles: true }));
            return true;
        } catch (e) {
            return false;
        }
    }

    function findFileInputForDropEvent(evt) {
        try {
            const root = gradioApp && gradioApp();
            if (!root) return null;
            let el = evt && evt.target ? evt.target : null;
            for (let i = 0; i < 10 && el; i++) {
                if (el.querySelector) {
                    const input = el.querySelector('input[type="file"]');
                    if (input) return input;
                }
                el = el.parentElement;
            }

            const x = typeof evt.clientX === 'number' ? evt.clientX : null;
            const y = typeof evt.clientY === 'number' ? evt.clientY : null;
            if (x === null || y === null) return null;

            const inputs = Array.from(root.querySelectorAll('input[type="file"]'));
            if (!inputs.length) return null;

            let best = null;
            let bestDist = Infinity;
            for (const input of inputs) {
                const rect = input.getBoundingClientRect();
                const cx = clamp(x, rect.left, rect.right);
                const cy = clamp(y, rect.top, rect.bottom);
                const dx = x - cx;
                const dy = y - cy;
                const dist = (dx * dx) + (dy * dy);
                if (dist < bestDist) {
                    bestDist = dist;
                    best = input;
                }
            }
            return best;
        } catch (e) {
            return null;
        }
    }

    function copyTextToClipboardLegacy(text) {
        try {
            const value = String(text || '');
            if (!value) return false;
            const ta = document.createElement('textarea');
            ta.value = value;
            ta.setAttribute('readonly', '');
            ta.style.position = 'fixed';
            ta.style.left = '-9999px';
            ta.style.top = '0';
            document.body.appendChild(ta);
            ta.focus();
            ta.select();
            const ok = document.execCommand && document.execCommand('copy');
            document.body.removeChild(ta);
            return !!ok;
        } catch (e) {
            return false;
        }
    }

    async function copySelectedToClipboard() {
        const item = transferState.items.find(x => x.id === transferState.selectedId);
        if (!item || !item.blob) {
            setTransferHintMessage('未选择图片');
            return;
        }
        try {
            try { window.focus(); } catch (e0) {}
            if (navigator.clipboard && navigator.clipboard.write && window.ClipboardItem) {
                const sourceBlob = item.blob;
                await navigator.clipboard.write([
                    new ClipboardItem({
                        'image/png': (async () => (await convertBlobToPng(sourceBlob)) || sourceBlob)()
                    })
                ]);
                setTransferHintMessage('已复制');
                return;
            }

            const dataUrl = await blobToDataUrl(item.blob);
            if (copyTextToClipboardLegacy(dataUrl)) {
                setTransferHintMessage('已复制为文本');
                return;
            }
        } catch (e) {
            console.error('Clipboard copy failed:', e);
            try {
                const dataUrl = await blobToDataUrl(item.blob);
                if (copyTextToClipboardLegacy(dataUrl)) {
                    setTransferHintMessage('已复制为文本');
                    return;
                }
            } catch (e2) {
            }
            setTransferHintMessage(window.isSecureContext ? '复制失败' : '复制失败：建议用 https/localhost');
        }
    }

    function hasTransferExternalDropPayload(dataTransfer) {
        try {
            if (!dataTransfer) return false;
            const types = dataTransfer.types ? Array.from(dataTransfer.types) : [];
            if (types.includes('application/x-simpleai-transfer-id') || types.includes('application/x-simpleai-image-dataurl')) return false;
            if (types.includes('Files')) return true;
            if (types.includes('text/uri-list') || types.includes('text/plain')) return true;
            const files = dataTransfer.files ? Array.from(dataTransfer.files) : [];
            return files.length > 0;
        } catch (e) {
            return false;
        }
    }

    async function handleTransferDropDataTransfer(dataTransfer) {
        const transferId = dataTransfer ? (dataTransfer.getData('application/x-simpleai-transfer-id') || '') : '';
        if (transferId) return;

        const files = (dataTransfer && dataTransfer.files) ? Array.from(dataTransfer.files) : [];
        if (files.length) {
            for (const f of files) {
                await addTransferFile(f);
            }
            return;
        }

        const uri = dataTransfer ? (dataTransfer.getData('text/uri-list') || '') : '';
        const text = dataTransfer ? (dataTransfer.getData('text/plain') || '') : '';
        const payload = (uri || text).trim();
        if (payload) {
            try {
                await addTransferUrl(payload);
            } catch (err) {
            }
        }
    }

    function initTransferDropZone() {
        const prevent = (e) => {
            e.preventDefault();
            e.stopPropagation();
        };

        transferPanel.addEventListener('dragover', (e) => {
            prevent(e);
            transferPanel.classList.add('dragover');
        });

        transferPanel.addEventListener('dragleave', (e) => {
            prevent(e);
            transferPanel.classList.remove('dragover');
        });

        transferPanel.addEventListener('drop', async (e) => {
            prevent(e);
            transferPanel.classList.remove('dragover');
            await handleTransferDropDataTransfer(e.dataTransfer);
        });
    }

    function updateTransferPanelLayout() {
        try {
            if (!transferPanel || !statusIndicator) return;
            if (!transferState.expanded) {
                transferPanel.style.maxHeight = '';
                transferPanel.style.top = 'calc(100% + 6px)';
                transferPanel.style.bottom = '';
                return;
            }

            const statusRect = statusIndicator.getBoundingClientRect();
            const margin = 8;
            const downAvail = window.innerHeight - (statusRect.bottom + 6) - margin;
            const upAvail = (statusRect.top - 6) - margin;
            const openUp = downAvail < 220 && upAvail > downAvail;

            if (openUp) {
                transferPanel.style.top = 'auto';
                transferPanel.style.bottom = 'calc(100% + 6px)';
                transferPanel.style.maxHeight = `${Math.max(140, Math.floor(upAvail))}px`;
            } else {
                transferPanel.style.bottom = 'auto';
                transferPanel.style.top = 'calc(100% + 6px)';
                transferPanel.style.maxHeight = `${Math.max(140, Math.floor(downAvail))}px`;
            }
        } catch (e) {
        }
    }

    function initTransferActions() {
        transferToggleBtn.addEventListener('click', (e) => {
            e.preventDefault();
            setTransferExpanded(!transferState.expanded, true);
        });

        const prevent = (e) => {
            e.preventDefault();
            e.stopPropagation();
        };

        transferToggleBtn.addEventListener('dragenter', (e) => {
            try {
                const types = e.dataTransfer && e.dataTransfer.types ? Array.from(e.dataTransfer.types) : [];
                const hasAny = types.includes('Files') || types.includes('text/uri-list') || types.includes('text/plain') || types.includes('application/x-simpleai-transfer-id') || types.includes('application/x-simpleai-image-dataurl');
                if (!hasAny) return;
            } catch (e0) {
            }
            prevent(e);
            if (hasTransferExternalDropPayload(e.dataTransfer)) {
                if (!transferState.expanded) setTransferExpanded(true, true);
                transferPanel.classList.add('dragover');
            }
        });

        transferToggleBtn.addEventListener('dragover', (e) => {
            try {
                const types = e.dataTransfer && e.dataTransfer.types ? Array.from(e.dataTransfer.types) : [];
                const hasAny = types.includes('Files') || types.includes('text/uri-list') || types.includes('text/plain') || types.includes('application/x-simpleai-transfer-id') || types.includes('application/x-simpleai-image-dataurl');
                if (!hasAny) return;
            } catch (e0) {
            }
            prevent(e);
            if (hasTransferExternalDropPayload(e.dataTransfer)) {
                if (!transferState.expanded) setTransferExpanded(true, true);
                transferPanel.classList.add('dragover');
            }
        });

        transferToggleBtn.addEventListener('drop', async (e) => {
            try {
                const types = e.dataTransfer && e.dataTransfer.types ? Array.from(e.dataTransfer.types) : [];
                const hasAny = types.includes('Files') || types.includes('text/uri-list') || types.includes('text/plain') || types.includes('application/x-simpleai-transfer-id') || types.includes('application/x-simpleai-image-dataurl');
                if (!hasAny) return;
            } catch (e0) {
            }
            prevent(e);
            if (!hasTransferExternalDropPayload(e.dataTransfer)) return;
            if (!transferState.expanded) setTransferExpanded(true, true);
            transferPanel.classList.remove('dragover');
            await handleTransferDropDataTransfer(e.dataTransfer);
        });

        transferClearBtn.addEventListener('click', (e) => {
            e.preventDefault();
            clearTransferItems();
        });

        transferPasteBtn.addEventListener('click', async (e) => {
            e.preventDefault();
            try {
                try { window.focus(); } catch (e0) {}
                let added = 0;
                let usedApi = false;

                if (navigator.clipboard && navigator.clipboard.read) {
                    usedApi = true;
                    try {
                        const items = await navigator.clipboard.read();
                        for (const clipItem of items) {
                            const type = (clipItem.types || []).find(t => String(t).startsWith('image/'));
                            if (!type) continue;
                            const blob = await clipItem.getType(type);
                            const file = new File([blob], `clipboard_${Date.now()}.png`, { type });
                            await addTransferFile(file);
                            added++;
                        }
                    } catch (e1) {
                    }
                }
                if (added > 0) {
                    setTransferHintMessage(`已粘贴 ${added} 张`);
                    return;
                }

                if (navigator.clipboard && navigator.clipboard.readText) {
                    usedApi = true;
                    const text = await navigator.clipboard.readText();
                    if (text && text.trim()) {
                        await addTransferUrl(text);
                        setTransferHintMessage('已粘贴');
                        return;
                    }
                }

                if (usedApi) {
                    setTransferHintMessage('剪贴板无图片');
                    return;
                }
                openTransferPasteOverlay();
            } catch (err) {
                console.error('Clipboard paste failed:', err);
                openTransferPasteOverlay();
            }
        });
    }

    function initTransferDropToGradio() {
        const isLayerForgeDragTarget = (evt) => {
            try {
                const t = evt && evt.target ? evt.target : null;
                const iframe = t && t.closest ? (t.tagName === 'IFRAME' ? t : t.closest('iframe')) : (t && t.tagName === 'IFRAME' ? t : null);
                if (!iframe) return false;
                const src = String(iframe.getAttribute('src') || iframe.src || '');
                return /file=javascript\/layerforge\/app\.html/i.test(src) || /javascript\/layerforge\/app\.html/i.test(src);
            } catch (e) {
                return false;
            }
        };

        const isLayerForgePoint = (evt) => {
            try {
                if (isLayerForgeDragTarget(evt)) return true;
                const x = typeof evt?.clientX === 'number' ? evt.clientX : null;
                const y = typeof evt?.clientY === 'number' ? evt.clientY : null;
                if (x === null || y === null) return false;
                const iframes = Array.from(document.querySelectorAll('iframe'));
                for (const iframe of iframes) {
                    const src = String(iframe.getAttribute('src') || iframe.src || '');
                    const isLayerForge = /file=javascript\/layerforge\/app\.html/i.test(src) || /javascript\/layerforge\/app\.html/i.test(src);
                    if (!isLayerForge) continue;
                    const r = iframe.getBoundingClientRect();
                    if (x >= r.left && x <= r.right && y >= r.top && y <= r.bottom) return true;
                }
                return false;
            } catch (e) {
                return false;
            }
        };

        document.addEventListener('dragover', (e) => {
            try {
                if (isLayerForgePoint(e)) return;
                const types = e.dataTransfer && e.dataTransfer.types ? Array.from(e.dataTransfer.types) : [];
                const hasPayload = types.includes('application/x-simpleai-transfer-id') || types.includes('application/x-simpleai-image-dataurl');
                if (!hasPayload) return;
                e.preventDefault();
            } catch (err) {
            }
        }, true);

        document.addEventListener('drop', (e) => {
            try {
                if (isLayerForgePoint(e)) return;
                const input = findFileInputForDropEvent(e);
                if (!input) return;
                const transferId = e.dataTransfer ? (e.dataTransfer.getData('application/x-simpleai-transfer-id') || '') : '';
                if (transferId) {
                    const idNum = Number(transferId);
                    const item = transferState.items.find(x => x.id === idNum);
                    if (item && item.blob) {
                        const file = fileFromBlob(item.blob, item.name, item.type);
                        if (setFileInputFromFile(input, file)) {
                            e.preventDefault();
                            return;
                        }
                    }
                }
                const dataUrl = e.dataTransfer ? (e.dataTransfer.getData('application/x-simpleai-image-dataurl') || '') : '';
                if (dataUrl && typeof setFileInputFromDataUrl === 'function') {
                    if (setFileInputFromDataUrl(input, dataUrl, `transfer_${Date.now()}.png`)) {
                        e.preventDefault();
                        return;
                    }
                }
            } catch (err) {
            }
        }, true);
    }

    function initImageTransferStation() {
        transferPanelActions.appendChild(transferPasteBtn);
        transferPanelActions.appendChild(transferClearBtn);
        transferPanel.appendChild(transferPanelActions);
        transferPanel.appendChild(transferHint);
        transferPanel.appendChild(transferList);
        initTransferActions();
        initTransferDropZone();
        initTransferDropToGradio();
        initTransferDirectPaste();
        renderTransferGrid();
        initTransferCrossTabSync();
    }

    function checkAdminAPIAvailability() {
        state.hasAdminAPI = !!(window.pywebview && 
                             window.pywebview.api && 
                             typeof window.pywebview.api.switchToAdmin === 'function');
    }

    // ==================== 初始化 ====================
    function initializeMonitor() {
        checkAdminAPIAvailability();
        // 检测并应用主题
        state.currentTheme = detectTheme();
        applyTheme();
        const enableTransferStation = !isMobileDevice();

        // 注入样式
        document.head.appendChild(style);

        // 组装 DOM
        statusContainer.appendChild(statusIndicator);
        statusIndicator.appendChild(statusContent);
        if (enableTransferStation) {
            statusIndicator.appendChild(transferToggleBtn);
            statusIndicator.appendChild(transferPanel);
        } else {
            transferState.expanded = false;
            statusIndicator.classList.remove('transfer-expanded');
        }
        statusIndicator.classList.toggle('transfer-expanded', !!transferState.expanded);
        // 新增resize事件监听
        window.addEventListener('resize', () => {
            // 复位到初始位置
            statusContainer.style.left = 'auto';
            statusContainer.style.top = '18px';
            statusContainer.style.right = '3px';
            statusContainer.style.bottom = 'auto';
            state.initialPositionMoved = false; // 重置位置标记
            if (transferState && transferState.expanded) requestAnimationFrame(updateTransferPanelLayout);
        });
        // 集成到 Gradio
        const gradioContainer = gradioApp();
        if (gradioContainer) {
            const host = document.body || gradioContainer;
            host.appendChild(statusContainer);
            
            // 初始化拖拽功能
            initDragFeature();
            if (enableTransferStation) {
                initImageTransferStation();
            }
        }
        // 启动检测
        setInterval(performHealthCheck, CHECK_INTERVAL);
        performHealthCheck();
    }

    // 启动监控
    if (document.readyState === 'complete') {
        initializeMonitor();
    } else {
        window.addEventListener('load', initializeMonitor);
    }
})();


