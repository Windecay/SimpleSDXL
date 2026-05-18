import { createApp } from 'vue'
import App from './App.vue'
import './style.css'
import '@/styles/theme.css'
import VueClipboard from 'vue-clipboard3'
import { createPinia } from 'pinia' // 导入 Pinia
import i18n, { initI18n } from './i18n'
import { version } from './utils/version.js'
const { toClipboard } = VueClipboard()
import piniaPluginPersistedstate from 'pinia-plugin-persistedstate'
import '@/styles/vue3-json-viewer.css'

const div = document.createElement('div')
div.id = 'weilin_comfyui_tools_prompt_ui_div'
const body = document.querySelector('body')
body.appendChild(div)

const pinia = createPinia()
pinia.use(piniaPluginPersistedstate)

const app = createApp(App)
app.use(pinia)
app.use(i18n)
// 在 pinia 初始化后调用
initI18n()
app.config.globalProperties.$copyText = toClipboard
app.mount(div)

console.log(`WeiLin Prompt UI is running - version ${version}`)
console.log(`WeiLin 节点插件已运行 - 版本 ${version}`)


const observer = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
        // 检查新增的节点中是否有目标 class
        const elements = document.querySelectorAll('#weilin-hidden-weight');
        if (elements.length > 0) {
            for (const element of elements) {
               const parentEl = element.closest('.lg-node-widget');
                parentEl.style.display = 'none';
                // console.log('目标元素已隐藏', parentEl);
            }
            // observer.disconnect(); // 如果只需要检测一次，可以断开观察
        }
    });
});

// 配置观察选项
const config = {
    childList: true, // 监听子节点的变化
    subtree: true    // 监听整个子树
};

// 开始观察 document.body（或指定其他父元素）
observer.observe(document.body, config);