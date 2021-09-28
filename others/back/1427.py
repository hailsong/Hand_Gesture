target = str(input())
li = list(target)
for i in li:
    i = int(i)
li = sorted(li)[::-1]
for i in li:
    print(i, end='')