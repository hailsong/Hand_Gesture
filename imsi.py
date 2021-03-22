# 각 응답에 따라 후보들이 획득할 점수, 지금은 5지선다에 문제 3개인 상황
POINT_SET = [
    {
        # Question 1
        'category' : 0,
        'points' : [[0, 5, 2, 1, 4],
                    [2, 3, 1, 4, 0]]
    },
    {
        # Question 2
        'category': 0,
        'points': [[0, 5, 2, 1, 4],
                   [2, 3, 1, 4, 0]]
    },
    {
        # Question 3
        'category': 0,
        'points': [[0, 5, 2, 1, 4],
                   [2, 3, 1, 4, 0]]
    },
    {
        # Question 4
        'category': 0,
        'points': [[0, 5, 2, 1, 4],
                   [2, 3, 1, 4, 0]]
    },
    {
        # Question 5
        'category': 0,
        'points': [[0, 5, 2, 1, 4],
                   [2, 3, 1, 4, 0]]
    },
    {
        # Question 6
        'category': 0,
        'points': [[0, 5, 2, 1, 4],
                   [2, 3, 1, 4, 0]]
    },
    {
        # Question 7
        'category': 0,
        'points': [[0, 5, 2, 1, 4],
                   [2, 3, 1, 4, 0]]
    },
    {
        # Question 8
        'category': 0,
        'points': [[0, 5, 2, 1, 4],
                   [2, 3, 1, 4, 0]]
    },
    {
        # Question 9
        'category': 0,
        'points': [[0, 5, 2, 1, 4],
                   [2, 3, 1, 4, 0]]
    },
    {
        # Question 10
        'category': 0,
        'points': [[0, 5, 2, 1, 4],
                   [2, 3, 1, 4, 0]]
    },
    {
        # Question 11
        'category': 0,
        'points': [[0, 5, 2, 1, 4],
                   [2, 3, 1, 4, 0]]
    },
    {
        # Question 12
        'category': 0,
        'points': [[0, 5, 2, 1, 4],
                   [2, 3, 1, 4, 0]]
    },
    {
        # Question 13
        'category': 0,
        'points': [[0, 5, 2, 1, 4],
                   [2, 3, 1, 4, 0]]
    },
    {
        # Question 14
        'category': 0,
        'points': [[0, 5, 2, 1, 4],
                   [2, 3, 1, 4, 0]]
    },
    {
        # Category
        # TODO : points는 초기 가중치, set_category_weight에서 전역변수로 호출 후 수정됨
        'category': -1,
        'points': [1, 1, 1.5, 3, 2, 4]
    },
]
CATEGORY_NUM = 6

# 함수의 input인 response_list는 (질문의 수) 크기의 1D list. candidate_num은 후보의 수
def calculate(response_list):
    set_category_weight(response_list[-1])
    score_set = get_score(response_list)
    score_sum = sum_score(score_set)
    #print(score_set)
    #print(score_sum)
    return score_set, score_sum

def set_category_weight(target):
    global POINT_SET
    if POINT_SET[-1]['category'] == -1:
        POINT_SET[-1]['points'][target - 1] = POINT_SET[-1]['points'][target - 1] * 1.5
    else:
        print('Category wight error detected')

def get_score(response_list):
    output = [[0., 0.] for _ in range(CATEGORY_NUM)] #카테고리별로 얻은 점수
    for index in range(len(response_list) - 1):
        response = response_list[index]
        for cand in range(2):
            output[POINT_SET[index]['category']][cand] = output[POINT_SET[index]['category']][cand] +\
                                                         POINT_SET[index]['points'][cand][response - 1]
    #print(output) #카테고리별 얻은 점수
    return output

def sum_score(score_set):
    category_weight = POINT_SET[-1]['points']
    for i in range(len(score_set)):
        score_set[i] = [s*category_weight[i] for s in score_set[i]]
    return score_set


if __name__ == "__main__":
    response_list = [4, 1, 3, 1, 2, 3, 5, 2, 1, 5, 2, 4, 1, 1, 2]
    result = calculate(response_list)
    print(result)

