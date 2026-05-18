import request from './request'

// Lora相关接口
export const loraApi = {
  // 获取Lora列表
  getLoraList: (params) => {
    return request({
      url: '/get_lora_list',
      method: 'get',
      params
    })
  },

  // 获取Lora全部列表
  getAllLoraList: (params) => {
    return request({
      url: '/get_lora_load_all',
      method: 'get',
      params
    })
  },

  // 获取执行进度
  getAllLoraStatus: (params) => {
    return request({
      url: '/get_lora_load_status',
      method: 'get',
      params
    })
  },

  // 获取Lora详情
  getLoraDetail: (params) => {
    return request({
      url: '/lorainfo/api/loras/info',
      method: 'get',
      params
    })
  },

  // 获取Lora信息C站获取
  getLoraRefresh: (params) => {
    return request({
      url: '/lorainfo/api/loras/info/refresh',
      method: 'get',
      params
    })
  },

  // 上传Lora图片
  postUplaodImg: (image, path, fileName) => {
    const body = new FormData()
    body.append('image', image)
    body.append('path', path)
    body.append('fileName', fileName)
    return request({
      url: '/lorainfo/api/loras/set/img',
      method: 'post',
      data: body,
      headers: { 'Content-Type': 'multipart/form-data' }
    })
  },

  // 保存Lora信息
  postLoraSave: (file, json) => {
    const body = new FormData()
    body.append('json', JSON.stringify(json))
    return request({
      url: `/lorainfo/api/loras/info?file=${file}`,
      method: 'post',
      data: body,
      headers: { 'Content-Type': 'multipart/form-data' }
    })
  },

  // 删除Lora信息字段
  postLoraDelet: (file, json) => {
    return request({
      url: `/lorainfo/api/delete/loras/info/filed?file=${file}`,
      method: 'post',
      data: {
        json: json
      }
    })
  },

  // 获取Lora文件夹列表
  getLoraFolderList: () => {
    return request({
      url: '/get_lora_folder_list',
      method: 'post'
    })
  },
  // 根据数组获取Lora具体信息
  getLoraRangeList: (range) => {
    return request({
      url: '/get_lora_list_by_range',
      method: 'post',
      data: { range }
    })
  },
  // 查询Lora返回相关信息
  searchLoraGetFolderList: (search) => {
    return request({
      url: '/get_lora_list_by_search',
      method: 'post',
      data: { search }
    })
  }
}
