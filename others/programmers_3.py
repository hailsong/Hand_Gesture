def solution(arr1, arr2):
    answer = [[0 for _ in range(len(arr2[0]))] for _ in range(len(arr1))]

    for row_num in range(len(arr1)):
        for col in range(len(arr1[0])):
            for col2 in range(len(arr2[0])):
                #print(row_num, col, col2)
                answer[row_num][col2] += arr1[row_num][col] * arr2[col][col2]

    return answer

# def solution(arr1, arr2):
#     answer = [[]]
#     return answer

print(solution([[2, 3, 2], [4, 2, 4]], [[5, 4, 3], [2, 4, 1], [3, 1, 1]]))
