# Yarn 依赖管理指南

## 📋 概述

本项目使用 **Yarn** 作为包管理器,提供更快的安装速度、更好的依赖管理和安全检查机制。

## 🚀 快速开始

### 1. 首次安装依赖

```bash
# 进入src目录
cd src

# 安装依赖
yarn install
```

### 2. 开发模式运行

```bash
yarn dev
```

### 3. 编译生产版本

```bash
yarn build
```

## 🔒 安全检查

### 自动安全检查

在安装依赖后,建议运行安全检查:

```bash
# 运行完整的安全检查
yarn run install:check
```

这个脚本会检查:
- ✅ Yarn 是否正确安装
- ✅ yarn.lock 文件是否存在
- ✅ 依赖包安全漏洞
- ✅ 依赖包版本一致性
- ✅ 过时的依赖包

### 手动安全检查

#### 检查安全漏洞

```bash
# 检查中等及以上级别的漏洞
yarn audit

# 检查所有漏洞
yarn audit --level low
```

#### 自动修复漏洞

```bash
# 自动修复可修复的漏洞
yarn audit fix
```

#### 检查过时的依赖

```bash
# 查看过时的依赖包
yarn outdated
```

#### 更新依赖包

```bash
# 更新所有依赖包到最新版本
yarn upgrade

# 更新特定依赖包
yarn upgrade <package-name>
```

## 📦 依赖管理

### 添加新依赖

```bash
# 添加生产依赖
yarn add <package-name>

# 添加开发依赖
yarn add -D <package-name>

# 添加特定版本
yarn add <package-name>@<version>
```

### 移除依赖

```bash
# 移除依赖
yarn remove <package-name>
```

### 清理缓存

```bash
# 清理yarn缓存
yarn cache clean
```

## 🔧 配置文件

### .yarnrc.yml

Yarn配置文件,包含:
- ✅ 严格SSL模式
- ✅ 离线镜像缓存
- ✅ 国内镜像源(淘宝)
- ✅ 缓存目录配置

### .npmrc

npm配置文件(兼容yarn),包含:
- ✅ 镜像源配置
- ✅ SSL验证
- ✅ 缓存位置
- ✅ 超时设置

### yarn.lock

依赖包版本锁定文件:
- ✅ 确保跨平台一致性
- ✅ 锁定精确版本
- ✅ 防止意外更新

## ⚠️ 重要提示

### 不要混用包管理器

**❌ 错误做法:**
```bash
npm install  # 不要使用npm
```

**✅ 正确做法:**
```bash
yarn install  # 只使用yarn
```

### 版本锁定

- `yarn.lock` 文件必须提交到版本控制
- 不要手动修改 `yarn.lock`
- 不同环境应该使用相同的 `yarn.lock`

### 安全建议

1. **定期运行安全检查**
   ```bash
   # 每次更新依赖后
   yarn run install:check
   ```

2. **及时修复漏洞**
   ```bash
   # 发现漏洞后立即修复
   yarn audit fix
   ```

3. **关注依赖更新**
   ```bash
   # 定期检查过时的依赖
   yarn outdated
   ```

## 📊 可用脚本

| 脚本 | 命令 | 说明 |
|------|------|------|
| 安装检查 | `yarn run install:check` | 完整的安全检查 |
| 安全审计 | `yarn audit` | 检查安全漏洞 |
| 修复漏洞 | `yarn audit:fix` | 自动修复漏洞 |
| 检查更新 | `yarn outdated` | 查看过时依赖 |
| 更新依赖 | `yarn upgrade` | 更新到最新版本 |
| 清理缓存 | `yarn clean` | 清理yarn缓存 |

## 🐛 常见问题

### Q: 为什么使用yarn而不是npm?

A: Yarn提供:
- 更快的安装速度
- 更好的依赖管理
- 更安全的安全检查
- 更可靠的版本锁定

### Q: 如何切换从npm到yarn?

A:
```bash
# 1. 删除npm相关文件
rm package-lock.json
rm -rf node_modules

# 2. 使用yarn安装
yarn install
```

### Q: yarn.lock文件应该提交吗?

A: ✅ **必须提交!**
- 确保团队使用相同的依赖版本
- 避免跨平台差异
- 提高构建可重现性

### Q: 如何解决依赖冲突?

A:
```bash
# 1. 清理缓存
yarn cache clean

# 2. 删除node_modules和yarn.lock
rm -rf node_modules yarn.lock

# 3. 重新安装
yarn install
```

## 📞 技术支持

如果遇到问题,请:
1. 检查Yarn版本: `yarn --version`
2. 运行安全检查: `yarn run install:check`
3. 查看Yarn文档: https://yarnpkg.com/getting-started

## 🔗 相关资源

- [Yarn官方文档](https://yarnpkg.com/)
- [Yarn安全审计](https://yarnpkg.com/cli/audit)
- [Yarn配置](https://yarnpkg.com/configuration/yarnrc)

---

**最后更新**: 2024-03-15
**Yarn版本**: 1.22.22
