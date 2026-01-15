
export const api = {
    apiURL: (path) => {
        const baseUrl = window.comfy_api_url || window.location.origin;
        if (path.startsWith("/")) {
            path = path.substring(1);
        }
        return `${baseUrl}/${path}`;
    },
    fetchApi: async (path, options) => {
        const url = api.apiURL(path);
        return fetch(url, options);
    },
    addEventListener: () => { },
    registerFunc: () => { }
};
