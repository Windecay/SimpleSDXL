import { createModuleLogger } from "/file=javascript/layerforge/js/utils/LoggerUtils.js?v=patch26";
const log = createModuleLogger('db');
const DB_NAME = 'CanvasNodeDB_v2';
const STATE_STORE_NAME = 'CanvasState';
const IMAGE_STORE_NAME = 'CanvasImages';
const DB_VERSION = 12;
let db = null;
/**
 * Funkcja pomocnicza do tworzenia żądań IndexedDB z ujednoliconą obsługą błędów
 * @param {IDBObjectStore} store - Store IndexedDB
 * @param {DBRequestOperation} operation - Nazwa operacji (get, put, delete, clear)
 * @param {any} data - Dane dla operacji (opcjonalne)
 * @param {string} errorMessage - Wiadomość błędu
 * @returns {Promise<any>} Promise z wynikiem operacji
 */
function createDBRequest(store, operation, data, errorMessage) {
    return new Promise((resolve, reject) => {
        let request;
        switch (operation) {
            case 'get':
                request = store.get(data);
                break;
            case 'put':
                request = store.put(data);
                break;
            case 'delete':
                request = store.delete(data);
                break;
            case 'clear':
                request = store.clear();
                break;
            default:
                reject(new Error(`Unknown operation: ${operation}`));
                return;
        }
        request.onerror = (event) => {
            log.error(errorMessage, event.target.error);
            reject(errorMessage);
        };
        request.onsuccess = (event) => {
            resolve(event.target.result);
        };
    });
}
function openDB() {
    return new Promise((resolve, reject) => {
        if (db) {
            resolve(db);
            return;
        }
        const request = indexedDB.open(DB_NAME, DB_VERSION);
        request.onerror = (event) => {
            log.error("IndexedDB error:", event.target.error);
            reject("Error opening IndexedDB.");
        };
        request.onsuccess = (event) => {
            db = event.target.result;
            db.onversionchange = () => {
                db.close();
                db = null;
            };
            resolve(db);
        };
        request.onupgradeneeded = (event) => {
            const dbInstance = event.target.result;
            if (!dbInstance.objectStoreNames.contains(STATE_STORE_NAME)) {
                dbInstance.createObjectStore(STATE_STORE_NAME, { keyPath: 'id' });
            }
            if (!dbInstance.objectStoreNames.contains(IMAGE_STORE_NAME)) {
                dbInstance.createObjectStore(IMAGE_STORE_NAME, { keyPath: 'imageId' });
            }
        };
    });
}
export async function getCanvasState(id) {
    const db = await openDB();
    const transaction = db.transaction([STATE_STORE_NAME], 'readonly');
    const store = transaction.objectStore(STATE_STORE_NAME);
    const result = await createDBRequest(store, 'get', id, "Error getting canvas state");
    return result ? result.state : null;
}
export async function setCanvasState(id, state) {
    const db = await openDB();
    const transaction = db.transaction([STATE_STORE_NAME], 'readwrite');
    const store = transaction.objectStore(STATE_STORE_NAME);
    await createDBRequest(store, 'put', { id, state }, "Error setting canvas state");
}
export async function saveImage(imageId, imageBitmap) {
    const db = await openDB();
    const transaction = db.transaction([IMAGE_STORE_NAME], 'readwrite');
    const store = transaction.objectStore(IMAGE_STORE_NAME);
    await createDBRequest(store, 'put', { imageId, data: imageBitmap }, "Error saving image");
}

export async function getImage(imageId) {
    const db = await openDB();
    const transaction = db.transaction([IMAGE_STORE_NAME], 'readonly');
    const store = transaction.objectStore(IMAGE_STORE_NAME);
    const result = await createDBRequest(store, 'get', imageId, "Error getting image");
    
    if (result) {
        if (result.data) {
             return result.data;
        } else if (result.imageSrc) { // Fallback for legacy data
             return result.imageSrc;
        }
    }
    return null;
}
export async function removeImage(imageId) {
    const db = await openDB();
    const transaction = db.transaction([IMAGE_STORE_NAME], 'readwrite');
    const store = transaction.objectStore(IMAGE_STORE_NAME);
    await createDBRequest(store, 'delete', imageId, "Error removing image");
}
export async function getAllImageIds() {
    const db = await openDB();
    const transaction = db.transaction([IMAGE_STORE_NAME], 'readonly');
    const store = transaction.objectStore(IMAGE_STORE_NAME);
    return new Promise((resolve, reject) => {
        const request = store.getAllKeys();
        request.onerror = (event) => {
            log.error("Error getting all image IDs:", event.target.error);
            reject("Error getting all image IDs");
        };
        request.onsuccess = (event) => {
            const imageIds = event.target.result;
            resolve(imageIds);
        };
    });
}
export async function clearAllCanvasStates() {
    const db = await openDB();
    const transaction = db.transaction([STATE_STORE_NAME], 'readwrite');
    const store = transaction.objectStore(STATE_STORE_NAME);
    await createDBRequest(store, 'clear', null, "Error clearing canvas states");
}
