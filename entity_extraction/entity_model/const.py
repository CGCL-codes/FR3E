task_ner_labels = {
    'finance':['公司企业', '时间', '人员', '地域', '产品', '业务', '行业'],
    'baidu':['网站', '歌曲', '企业', '影视作品', '音乐专辑', '学校', '人物', '机构', '文本', '国家', '日期', '图书作品', '网络小说', '地点', '出版社']
}

task_rel_labels = {
    'finance':['开展', "拥有", "位于", "合作", "投资", "属于", "应用于", "任职于"],
    'baidu':["出品公司","国籍","出生地", "民族", "出生日期", "毕业院校", "歌手", "所属专辑",
   "作词", "作曲", "连载网站", "作者", "出版社", "主演", "导演", "编剧", "上映时间", "成立日期"]
}

def get_labelmap(label_list):
    label2id = {}
    id2label = {}
    for i, label in enumerate(label_list):
        label2id[label] = i + 1
        id2label[i + 1] = label
    return label2id, id2label
