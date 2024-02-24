Dict = {
    "我们": 1,
    "在": 1,
    "学习": 1,
    "自然": 1,
    "语言": 1,
    "处理": 1,
    "自然语言": 1,
    "语言处理": 1,
    "自然语言处理": 1,
}


def build_dag(sentence, Dict):
    N = len(sentence)
    dag = {i: [] for i in range(N)}
    for i in range(N):
        for j in range(i, N):
            if sentence[i : j + 1] in Dict:
                dag[i].append(j)
    return dag


def dfs(dag, start, path, result, sentence):
    if start == len(sentence):
        result.append(path)
        return
    for end in dag[start]:
        dfs(dag, end + 1, path + [sentence[start : end + 1]], result, sentence)


def find_all_paths(sentence, dag):
    result = []
    dfs(dag, 0, [], result, sentence)
    return result


# 示例句子
sentence = "我们在学习自然语言处理"
dag = build_dag(sentence, Dict)
all_paths = find_all_paths(sentence, dag)

# 打印所有可能的切分方式
for path in all_paths:
    print(" / ".join(path))
