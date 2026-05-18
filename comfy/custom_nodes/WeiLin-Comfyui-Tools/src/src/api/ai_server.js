import request from './request'

export const aiServerApi = {
  getAiServerSetting: () => {
    return request({
      url: '/ai_server/get_settings',
      method: 'post',
      data: {}
    })
  },

  updateAiServerSetting: (data) => {
    return request({
      url: '/ai_server/update_settings',
      method: 'post',
      data
    })
  },

  getAiModels: () => {
    return request({
      url: '/ai_server/get_ai_models',
      method: 'post',
      data: {}
    })
  }
}
