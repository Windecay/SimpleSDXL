import request from './request'
// 翻译相关接口
export const randomTagApi = {
  // 获取模板列表
  getTemplateList: () => {
    return request({
      url: '/random_template/get_template_list',
      method: 'post'
    })
  },

  getTemplateData: (name) => {
    return request({
      url: '/random_template/get_template_data',
      method: 'post',
      data: { name }
    })
  },

  saveTemplateData: (data) => {
    return request({
      url: '/random_template/save_template',
      method: 'post',
      data: { data }
    })
  },

  updateTemplateData: (name, data) => {
    return request({
      url: '/random_template/update_template',
      method: 'post',
      data: { name, data }
    })
  },

  deleteTemplateData: (name) => {
    return request({
      url: '/random_template/delete_template',
      method: 'post',
      data: { name }
    })
  },

  getRandomTemplateApple: () => {
    return request({
      url: '/get/setting/get_random_template_setting',
      method: 'post'
    })
  },

  updateRandomTemplateApple: (path) => {
    return request({
      url: '/update/setting/update_random_template_setting',
      method: 'post',
      data: {
        path
      }
    })
  },

  goRandomTemplate: () => {
    return request({
      url: '/random_template/go_random_template',
      method: 'post'
    })
  },

  goRandomTemplatePath: (name) => {
    return request({
      url: '/random_template/go_random_template_path',
      method: 'post',
      data: { name }
    })
  }
}
