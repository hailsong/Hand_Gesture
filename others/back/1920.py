from sys import stdin, stdout
n = stdin.readline()
N = sorted(map(int,stdin.readline().split()))
m = stdin.readline()
M = map(int, stdin.readline().split())

def binary(l, N, start, end):
    mid = N[len(N)//2]
    if mid > l:
        return binary(l, N, start + 1, end)
    elif mid < l:
        return binary(l, N, start, end-1)
    elif N[start] == l or N[end] ==l:
        return 1
    if start > end:
        return 0

for l in M:
    start = 0
    end = len(N)-1
    print(binary(l,N,start,end))