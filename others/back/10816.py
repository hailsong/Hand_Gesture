'''
10
6 3 2 10 10 10 -10 -10 7 3
8
10 9 -5 2 3 4 5 -10
'''

N = int(input())
li = list(map(int, input().split(' ')))
M = int(input())
target = list(map(int, input().split(' ')))

def find(li, start, end, now):
    out = 0
    if target[start] == now:
        out += 1
    elif target[end] == now:
        out += 1
    find(start + 1, end - 1, now)


for i in target:
