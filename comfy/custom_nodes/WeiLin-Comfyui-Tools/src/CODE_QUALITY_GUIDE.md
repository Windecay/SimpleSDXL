# 代码质量检查指南

## 📋 概述

本项目配置了完整的代码质量检查体系,包括:
- ✅ **ESLint** - JavaScript/Vue代码检查
- ✅ **Prettier** - 代码格式化
- ✅ **Stylelint** - CSS/Vue样式检查
- ✅ **Husky** - Git Hooks自动化
- ✅ **Lint-staged** - 暂存文件检查

## 🚀 快速开始

### 安装依赖

```bash
cd src
yarn install
```

### 运行代码质量检查

```bash
# 完整的代码质量检查
yarn quality

# 自动修复代码问题
yarn quality:fix
```

## 🔧 可用命令

### ESLint (JavaScript/Vue检查)

```bash
# 检查代码质量
yarn lint

# 自动修复可修复的问题
yarn lint:fix
```

### Stylelint (CSS/Vue样式检查)

```bash
# 检查样式质量
yarn lint:style

# 自动修复样式问题
yarn lint:style:fix
```

### Prettier (代码格式化)

```bash
# 格式化所有文件
yarn format

# 检查文件格式
yarn format:check
```

### 综合检查

```bash
# 运行所有检查
yarn quality

# 自动修复所有问题
yarn quality:fix
```

## 📁 配置文件

### ESLint配置 (`.eslintrc.json`)

**检查内容**:
- ✅ Vue 3最佳实践
- ✅ JavaScript代码质量
- ✅ 代码风格一致性
- ✅ 潜在错误检测

**主要规则**:
- 单引号字符串
- 无分号
- 2空格缩进
- 箭头函数优先
- const/let优先于var
- Vue组件命名规范

### Prettier配置 (`.prettierrc.json`)

**格式化规则**:
- ✅ 单引号
- ✅ 无分号
- ✅ 2空格缩进
- ✅ 100字符换行
- ✅ LF换行符

### Stylelint配置 (`.stylelintrc.json`)

**样式检查**:
- ✅ CSS规范
- ✅ Vue单文件组件样式
- ✅ 选择器规范
- ✅ 属性顺序
- ✅ 颜色格式

## 🔍 Git Hooks

### Pre-commit Hook

每次提交代码前自动运行:
```bash
yarn quality:fix
```

这会:
1. 自动修复代码问题
2. 自动格式化代码
3. 检查样式问题
4. 确保代码质量

### 配置文件

- `.husky/pre-commit` - Pre-commit hook脚本
- `.lintstagedrc.json` - Lint-staged配置

## 🎯 最佳实践

### 开发流程

1. **编写代码**
   ```bash
   # 正常开发
   yarn dev
   ```

2. **提交代码**
   ```bash
   git add .
   git commit -m "feat: 新功能"
   # Pre-commit hook会自动运行质量检查
   ```

3. **手动检查** (可选)
   ```bash
   # 提交前手动检查
   yarn quality
   ```

### 代码规范

#### JavaScript/Vue

```javascript
// ✅ 推荐
const message = 'Hello'
const handleClick = () => {
  console.log(message)
}

// ❌ 避免
var message = 'Hello'
function handleClick() {
  console.log(message)
}
```

#### Vue组件

```vue
<!-- ✅ 推荐 -->
<script setup>
import { ref } from 'vue'

const count = ref(0)
</script>

<template>
  <div class="container">
    <button @click="count++">
      {{ count }}
    </button>
  </div>
</template>

<style scoped>
.container {
  padding: 16px;
}
</style>
```

#### CSS样式

```css
/* ✅ 推荐 */
.container {
  padding: 16px;
  margin: 0 auto;
}

.button {
  background-color: #007bff;
  color: white;
}

/* ❌ 避免 */
.container {
  padding: 16px;
  MARGIN: 0 auto;  /* 大写属性 */
}
```

## 🐛 常见问题

### Q: Pre-commit hook失败怎么办?

A: 运行自动修复:
```bash
yarn quality:fix
git add .
git commit -m "fix: 修复代码质量问题"
```

### Q: 如何跳过Git Hooks?

A: 不建议跳过,但可以:
```bash
git commit --no-verify -m "跳过检查"
```

### Q: ESLint报错太多怎么办?

A: 1. 先运行自动修复
   ```bash
   yarn lint:fix
   ```

   2. 如果还有错误,可以临时禁用某些规则
   ```javascript
   // eslint-disable-next-line no-console
   console.log('临时禁用规则')
   ```

### Q: Prettier格式化不符合预期?

A: 检查配置文件:
```bash
# 查看配置
cat .prettierrc.json

# 检查忽略文件
cat .prettierignore
```

### Q: Stylelint报错?

A: 运行自动修复:
```bash
yarn lint:style:fix
```

## 📊 代码质量指标

### 目标指标

- ✅ ESLint错误: 0
- ✅ Stylelint错误: 0
- ✅ Prettier格式一致性: 100%
- ✅ 代码覆盖率: >80%

### 检查报告

运行检查后会显示:
```
✅ ESLint: 通过
✅ Stylelint: 通过
✅ Prettier: 通过
```

## 🔗 相关资源

- [ESLint文档](https://eslint.org/)
- [Prettier文档](https://prettier.io/)
- [Stylelint文档](https://stylelint.io/)
- [Vue风格指南](https://vuejs.org/style-guide/)
- [Husky文档](https://typicode.github.io/husky/)

## 📝 配置说明

### ESLint规则分类

#### 错误级别 (Error)
- 必须修复的问题
- 会导致代码运行错误
- 示例: 未定义变量、语法错误

#### 警告级别 (Warning)
- 建议修复的问题
- 可能导致潜在问题
- 示例: console.log、未使用的变量

### Prettier格式化

#### 自动格式化
- 引号: 单引号
- 分号: 无
- 缩进: 2空格
- 换行: LF
- 行宽: 100字符

#### 忽略文件
- node_modules/
- dist/
- *.min.js
- *.min.css

### Stylelint规则

#### 强制规则
- 选择器命名规范
- 属性顺序
- 颜色格式
- 单位规范

#### Vue特定规则
- Scoped样式
- 组件样式隔离
- 模板样式规范

## 🎉 总结

通过这套代码质量检查体系,我们可以:
- ✅ 保持代码风格一致
- ✅ 提前发现潜在问题
- ✅ 自动化代码格式化
- ✅ 提高代码质量
- ✅ 减少代码审查时间

**建议**: 每次提交代码前运行 `yarn quality` 检查代码质量!

---

**最后更新**: 2024-03-15
**工具版本**:
- ESLint: 10.0.3
- Prettier: 3.8.1
- Stylelint: 17.4.0
- Husky: 9.1.7
