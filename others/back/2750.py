N = int(input())

t = []
for i in range(N):
    t.append(int(input()))

out = sorted(t)
for i in out:
    print(i)