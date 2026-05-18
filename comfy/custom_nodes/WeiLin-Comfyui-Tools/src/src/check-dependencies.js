#!/usr/bin/env node

/**
 * 依赖包安全检查脚本
 * 使用yarn audit检查已知漏洞
 */

import { execSync } from 'child_process'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

console.log('🔍 开始依赖包安全检查...\n')

// 检查yarn是否安装
try {
  execSync('yarn --version', { stdio: 'pipe' })
  console.log('✅ Yarn 已安装\n')
} catch (error) {
  console.error('❌ Yarn 未安装,请先安装 Yarn')
  console.log('   安装命令: npm install -g yarn\n')
  process.exit(1)
}

// 检查yarn.lock是否存在
const yarnLockPath = path.join(__dirname, 'yarn.lock')
if (!fs.existsSync(yarnLockPath)) {
  console.error('❌ yarn.lock 文件不存在')
  console.log('   请先运行: yarn install\n')
  process.exit(1)
}
console.log('✅ yarn.lock 文件存在\n')

// 检查package.json是否存在
const packageJsonPath = path.join(__dirname, 'package.json')
if (!fs.existsSync(packageJsonPath)) {
  console.error('❌ package.json 文件不存在\n')
  process.exit(1)
}
console.log('✅ package.json 文件存在\n')

// 运行yarn audit检查漏洞
console.log('🔒 检查依赖包安全漏洞...\n')
try {
  const auditOutput = execSync('yarn audit --level moderate', {
    encoding: 'utf8',
    stdio: 'pipe'
  })

  console.log(auditOutput)
  console.log('✅ 未发现中等及以上级别的安全漏洞\n')
} catch (error) {
  const auditOutput = error.stdout.toString()
  console.log(auditOutput)

  // 检查是否有严重漏洞
  if (auditOutput.includes('Severity: CRITICAL') || auditOutput.includes('Severity: HIGH')) {
    console.error('❌ 发现严重安全漏洞!')
    console.log('   请运行以下命令修复:')
    console.log('   yarn audit fix\n')
    process.exit(1)
  } else if (auditOutput.includes('Severity: MODERATE')) {
    console.warn('⚠️  发现中等安全漏洞')
    console.log('   建议运行: yarn audit fix\n')
    process.exit(0)
  } else {
    console.log('✅ 未发现严重安全漏洞\n')
  }
}

// 检查依赖包版本一致性
console.log('📦 检查依赖包版本一致性...\n')
try {
  execSync('yarn check --integrity', {
    encoding: 'utf8',
    stdio: 'pipe'
  })
  console.log('✅ 依赖包版本一致\n')
} catch (error) {
  console.warn('⚠️  依赖包版本可能不一致')
  console.log('   建议运行: yarn install --force\n')
}

// 检查过时的依赖包
console.log('📋 检查过时的依赖包...\n')
try {
  const outdatedOutput = execSync('yarn outdated', {
    encoding: 'utf8',
    stdio: 'pipe'
  })

  if (outdatedOutput.trim()) {
    console.log('⚠️  发现过时的依赖包:\n')
    console.log(outdatedOutput)
    console.log('   建议运行: yarn upgrade\n')
  } else {
    console.log('✅ 所有依赖包都是最新版本\n')
  }
} catch (error) {
  console.log('✅ 无需检查过时的依赖包\n')
}

console.log('🎉 安全检查完成!\n')
