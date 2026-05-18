import request from './request'

export const cloudApi = {
  // 获取云仓库列表
  getTreeFromCloud: (path) => {
    return request({
      url: '/cloud/get/tree',
      method: 'post',
      data: { path: path }
    })
  },
  getLocalInstallPackage: () => {
    return request({
      url: '/cloud/get/local/package',
      method: 'post',
      data: {}
    })
  },
  installSelectPackage: (path, paths) => {
    return request({
      url: '/cloud/download/file',
      method: 'post',
      timeout: 0,
      data: { paths: paths, path: path }
    })
  }
}
