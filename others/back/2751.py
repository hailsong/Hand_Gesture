N = int(input())

target = []
for _ in range(N):
    tmp = int(input())
    target.append(tmp)

while len(target) != 0:
    for i in range(len(target)):
        if target[i] == min(target):
            print(target.pop(i))