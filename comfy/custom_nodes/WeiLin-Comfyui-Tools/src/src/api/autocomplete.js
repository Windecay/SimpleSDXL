import request from './request'
// 自动补全相关接口
export const autocompleteApi = {
  // 自动补全获取
  getAutocomplete: (text) => {
    return request({
      url: '/prompt/fast/autocomplete',
      method: 'post',
      data: { query: text }
    })
  },

  getAutocompleteLimit: () => {
    return request({
      url: '/get/setting/get_auto_limit_setting',
      method: 'post'
    })
  },

  updateAutocompleteLimit: (limit) => {
    return request({
      url: '/update/setting/update_auto_limit_setting',
      method: 'post',
      data: {
        limit: limit
      }
    })
  }
}
