import { fileURLToPath, URL } from 'node:url'
import { resolve } from 'path'
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

/** @type {import('vite').UserConfig} */
// https://vitejs.dev/config/
export default defineConfig({
  define: {
    'process.env.NODE_ENV': '"production"'
  },
  plugins: [vue()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('.', import.meta.url))
    }
  },
  build: {
    minify: true,
    sourcemap: true,
    lib: {
      entry: resolve(__dirname, 'main.js'),
      name: 'weilin-prompt-ui',
      formats: ['umd']
    },
    rollupOptions: {
      plugins: [],
      output: {
        globals: {},
        name: 'WeiLinPromptUI',
        dir: '../dist/', // 明确输出到项目根目录的dist
        format: 'umd',
        chunkFileNames: 'javascript/[name].chunk.js',
        entryFileNames: 'javascript/[name].entry.js',
        assetFileNames: (assetInfo) => {
          // 固定CSS文件名为style.css，确保前端能正确加载
          if (assetInfo.name.endsWith('.css')) {
            return 'style.css'
          }
          return '[name].[ext]'
        }
      }
    }
  }
})
