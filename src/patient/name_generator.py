"""
患者姓名生成器
为模拟患者生成符合中国命名习惯的随机姓名
"""
import random


# 常见姓氏（按使用频率排序）
COMMON_SURNAMES = [
    # 超级常见姓氏（前20）
    '王', '李', '张', '刘', '陈', '杨', '黄', '赵', '吴', '周',
    '徐', '孙', '马', '朱', '胡', '郭', '何', '林', '高', '罗',
    # 常见姓氏（21-50）
    '郑', '梁', '谢', '宋', '唐', '许', '韩', '冯', '邓', '曹',
    '彭', '曾', '肖', '田', '董', '袁', '潘', '于', '蒋', '蔡',
    '余', '杜', '叶', '程', '苏', '魏', '吕', '丁', '任', '沈',
    # 较常见姓氏（51-100）
    '姚', '卢', '姜', '崔', '钟', '谭', '陆', '汪', '范', '金',
    '石', '廖', '贾', '夏', '韦', '付', '方', '白', '邹', '孟',
    '熊', '秦', '邱', '江', '尹', '薛', '闫', '段', '雷', '侯',
    '龙', '史', '陶', '黎', '贺', '顾', '毛', '郝', '龚', '邵',
    '万', '钱', '严', '覃', '武', '戴', '莫', '孔', '向', '汤',
]

# 男性常用名字（单字和双字）
MALE_NAMES_SINGLE = [
    '伟', '强', '磊', '军', '勇', '涛', '明', '超', '刚', '平',
    '辉', '鹏', '华', '飞', '杰', '波', '斌', '凯', '浩', '亮',
    '健', '峰', '龙', '鑫', '宇', '洋', '帆', '旭', '阳', '昊',
]

MALE_NAMES_DOUBLE = [
    '建国', '建军', '志强', '志明', '俊杰', '伟强', '国强', '建华',
    '子轩', '浩然', '宇航', '梓豪', '博文', '天宇', '俊熙', '皓轩',
    '嘉豪', '子涵', '晨阳', '宇轩', '睿泽', '明轩', '思源', '锦程',
    '文博', '瑞霖', '泽宇', '煜祺', '智宇', '昊天', '铭轩', '展鹏',
]

# 女性常用名字（单字和双字）
FEMALE_NAMES_SINGLE = [
    '芳', '娟', '静', '丽', '敏', '秀', '英', '华', '玲', '红',
    '霞', '艳', '萍', '梅', '莉', '兰', '婷', '慧', '琳', '颖',
    '洁', '雪', '倩', '欣', '怡', '悦', '璐', '瑶', '晴', '蕾',
]

FEMALE_NAMES_DOUBLE = [
    '秀英', '淑芬', '美玲', '雅婷', '晓红', '丽华', '秀兰', '春梅',
    '雨婷', '欣怡', '思琪', '梓涵', '诗涵', '雨萱', '佳怡', '可欣',
    '梦琪', '雅琪', '诗雨', '语嫣', '婉婷', '雨薇', '梦瑶', '紫涵',
    '思妍', '诗琪', '雅馨', '梦洁', '静怡', '心怡', '欣妍', '晨曦',
]


def generate_patient_name(gender: str, age: int = None, seed: int = None) -> str:
    """
    生成符合中国命名习惯的患者姓名
    
    Args:
        gender: 性别，'男' 或 '女'
        age: 年龄（可选），用于选择更符合年龄段的名字
        seed: 随机种子（可选），用于生成可重复的姓名
        
    Returns:
        str: 生成的姓名，格式为"姓+名"
        
    Examples:
        >>> generate_patient_name('男', 30)
        '张伟强'
        >>> generate_patient_name('女', 25)
        '李雨婷'
    """
    # 如果提供了种子，设置随机种子
    if seed is not None:
        random.seed(seed)
    
    # 选择姓氏（权重：前20个姓氏出现概率更高）
    if random.random() < 0.6:  # 60%概率选择前20个常见姓
        surname = random.choice(COMMON_SURNAMES[:20])
    elif random.random() < 0.8:  # 32%概率选择21-50的姓
        surname = random.choice(COMMON_SURNAMES[20:50])
    else:  # 8%概率选择其他姓
        surname = random.choice(COMMON_SURNAMES[50:])
    
    
    if isinstance(age, str) and age.strip().isdigit():
        age = int(age.strip())
    
    # 根据性别和年龄选择名字
    if gender == '男':
        # 年龄影响名字风格
        if age is not None:
            if age >= 50:  # 老年人更可能有传统名字
                if random.random() < 0.7:
                    given_name = random.choice(MALE_NAMES_DOUBLE[:8])  # 传统双字名
                else:
                    given_name = random.choice(MALE_NAMES_SINGLE[:15])  # 传统单字名
            elif age >= 30:  # 中年人
                if random.random() < 0.6:
                    given_name = random.choice(MALE_NAMES_DOUBLE)
                else:
                    given_name = random.choice(MALE_NAMES_SINGLE)
            else:  # 年轻人更可能有现代名字
                if random.random() < 0.7:
                    given_name = random.choice(MALE_NAMES_DOUBLE[8:])  # 现代双字名
                else:
                    given_name = random.choice(MALE_NAMES_SINGLE[15:])  # 现代单字名
        else:
            # 没有年龄信息，随机选择
            if random.random() < 0.65:  # 65%双字名
                given_name = random.choice(MALE_NAMES_DOUBLE)
            else:  # 35%单字名
                given_name = random.choice(MALE_NAMES_SINGLE)
    
    elif gender == '女':
        # 年龄影响名字风格
        if age is not None:
            if age >= 50:  # 老年人更可能有传统名字
                if random.random() < 0.7:
                    given_name = random.choice(FEMALE_NAMES_DOUBLE[:8])  # 传统双字名
                else:
                    given_name = random.choice(FEMALE_NAMES_SINGLE[:15])  # 传统单字名
            elif age >= 30:  # 中年人
                if random.random() < 0.6:
                    given_name = random.choice(FEMALE_NAMES_DOUBLE)
                else:
                    given_name = random.choice(FEMALE_NAMES_SINGLE)
            else:  # 年轻人更可能有现代名字
                if random.random() < 0.7:
                    given_name = random.choice(FEMALE_NAMES_DOUBLE[8:])  # 现代双字名
                else:
                    given_name = random.choice(FEMALE_NAMES_SINGLE[15:])  # 现代单字名
        else:
            # 没有年龄信息，随机选择
            if random.random() < 0.65:  # 65%双字名
                given_name = random.choice(FEMALE_NAMES_DOUBLE)
            else:  # 35%单字名
                given_name = random.choice(FEMALE_NAMES_SINGLE)
    else:
        # 性别未知或其他情况，使用中性名字
        if random.random() < 0.5:
            given_name = random.choice(['文', '瑞', '宇', '欣', '悦', '晨', '睿', '嘉'])
        else:
            given_name = random.choice(['文轩', '瑞霖', '嘉欣', '晨曦', '悦宁', '睿智'])
    
    return surname + given_name


def generate_patient_name_with_id(patient_id: int, gender: str, age: int = None) -> str:
    """
    使用患者ID作为种子生成姓名，确保同一患者ID总是生成相同的姓名
    
    Args:
        patient_id: 患者ID
        gender: 性别，'男' 或 '女'
        age: 年龄（可选）
        
    Returns:
        str: 生成的姓名
        
    Examples:
        >>> generate_patient_name_with_id(1, '男', 30)
        '王建国'  # 每次调用都会返回相同的名字
    """
    return generate_patient_name(gender, age, seed=patient_id)


if __name__ == "__main__":
    # 测试代码
    print("=== 姓名生成器测试 ===\n")
    
    # 测试1：生成不同性别和年龄的姓名
    print("测试1：不同性别和年龄")
    test_cases = [
        ('男', 25, "年轻男性"),
        ('男', 45, "中年男性"),
        ('男', 65, "老年男性"),
        ('女', 22, "年轻女性"),
        ('女', 40, "中年女性"),
        ('女', 60, "老年女性"),
    ]
    
    for gender, age, desc in test_cases:
        names = [generate_patient_name(gender, age) for _ in range(3)]
        print(f"{desc}（{age}岁）: {', '.join(names)}")
    
    print("\n测试2：使用患者ID生成（可重复）")
    for patient_id in [1, 7, 42, 100]:
        name1 = generate_patient_name_with_id(patient_id, '男', 30)
        name2 = generate_patient_name_with_id(patient_id, '男', 30)
        print(f"患者#{patient_id}: {name1} (重复生成: {name2}) - 一致性: {name1 == name2}")
    
    print("\n测试3：批量生成样例")
    print("男性样例:")
    for _ in range(10):
        print(f"  {generate_patient_name('男', random.randint(20, 60))}")
    
    print("\n女性样例:")
    for _ in range(10):
        print(f"  {generate_patient_name('女', random.randint(20, 60))}")

