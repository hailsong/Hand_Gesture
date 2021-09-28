import sys
N = int(sys.stdin.readline())
_li = []
for i in range(N):
    now = int(sys.stdin.readline())
    print(now)
    if now != 0:
        _li.append(now)
        _li.sort()
        _li = _li[::-1]
    elif len(_li) == 0:
        print(0)
    else:
        print(_li.pop(0))