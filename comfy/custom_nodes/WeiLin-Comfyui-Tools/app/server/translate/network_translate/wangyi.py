import requests


def youdao_translate(q, from_lang='Auto', to_lang='Auto'):
    if not q or len(q.strip()) == 0:
        raise ValueError('欲翻译文本不可以为空！')
    if len(q.strip()) > 1000:
        raise ValueError('欲翻译文本长度不可以超过1000！')
    data = {
        'q': q,
        'from': from_lang,
        'to': to_lang
    }
    information = requests.post('https://aidemo.youdao.com/trans', data)
    json_data = information.json()
    errorCode = json_data['errorCode']
    if errorCode != '0':
        raise RuntimeError(f'出现错误！返回的状态码为：{errorCode}')
    query = json_data['query']
    translation = json_data['translation']
    return query, translation


if __name__ == "__main__":
    while True:
        Q = ''
        while Q.strip() == '' or len(Q.strip()) > 1000:
            Q = input('输入欲翻译文本：')
            if Q.strip() == '':
                print('欲翻译文本不可以为空！')
            if len(Q.strip()) > 1000:
                print('欲翻译文本长度不可以超过1000！')
        print('欲翻译文本 => %s' % (Q))
        From = input('请输入原文本语种(为空或没有输入采用自动识别)：')
        if From.strip() == '':
            From = 'Auto'
        print('原文本语种 => %s' % ({'Auto': 'Auto(自动识别)'}.get(From, From)))
        To = input('请输入翻译文本语种(为空或没有输入采用自动识别)：')
        if To.strip() == '':
            To = 'Auto'
        print('翻译文本语种 => %s' % ({'Auto': 'Auto(自动识别)'}.get(To, To)))
        try:
            query, translation = youdao_translate(Q, From, To)
            print('原文本:' + query)
            for x in range(len(translation)):
                print('翻译结果' + str(x + 1) + " : " + translation[x])
        except Exception as e:
            print(e)
            break