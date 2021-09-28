import sys
N = int(sys.stdin.readline())

pic = [[1 for _ in range(N)] for _ in range(N)]

def solve(r, c, size):
    if size == 3:
        pic[r+1][c+1] = 0
    else:
        now = size//3
        for i in range(r + now, r + 2 * now):
            for j in range(c + now, c + 2 * now):
                pic[i][j] = 0
        for i in range(3):
            for j in range(3):
                solve(r + i*now, c + j*now, now)
solve(0, 0, N)

for r in pic:
    for i in r:
        if i == 1:
            print('*', end='')
        else:
            print(' ', end='')
    print('')