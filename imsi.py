import numpy as np

# 각 응답에 따라 후보들이 획득할 점수, 지금은 5지선다에 문제 3개인 상황
point_set = [
    # q1
    [[5, 0, 3],
     [0, 4, 3],
     [5, 0, 0],
     [0, 4, 0],
     [2, 0, 3]],
    # q2
    [[0, 2, 0],
     [3, 0, 0],
     [0, 0, 8],
     [0, 0, -3],
     [0, 1, 0]],
    # q3
    [[3, 0, 0],
     [0, 2, 0],
     [0, 0, 2],
     [0, 1, 0],
     [0, 0, 2]]
]

# TODO 3자대결에 정책적으로 겹쳐서 한 문항에 대해 여러 후보한테 점수를 줘야한다든지 할 수도...
# 함수의 input인 response_list는 (질문의 수) 크기의 1D list. candidate_num은 후보의 수
def calculate(response_list, candidate_num = 3):
    idx = 0
    result = np.zeros((candidate_num))
    for response in response_list:
        response_arr = np.zeros((5))
        response_arr[response - 1] = 1
        local_result = np.dot(response_arr, point_set[idx])
        result = result + local_result
        idx += 1
    print(result)
    return result

if __name__ == "__main__":
    response_list = [4, 1, 3]
    calculate(response_list)

