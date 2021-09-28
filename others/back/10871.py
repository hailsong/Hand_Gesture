'''
10 5
1 10 4 9 2 3 8 5 7 6
'''

N, M = map(int, input().split(' '))
_li = list(map(int, input().split(' ')))
for i in _li:
    if i < M:
        print(i, end=' ')