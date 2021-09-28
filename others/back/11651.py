# #좌표를 y좌표가 증가하는 순으로, y좌표가 같으면 x좌표가 증가하는 순서로 정렬한 다음 출력하는 프로그램을 작성하시오.
#

import sys
N=int(sys.stdin.readline())
target = []

for i in range(N):
    target.append(list(map(int, sys.stdin.readline().split())))
    target[i][0], target[i][1] = target[i][1], target[i][0]

target = sorted(target)

for i in range(N):
    print(target[i][1], target[i][0])