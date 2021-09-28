def sn(n):
    strn = str(n)
    for s in strn:
        n += int(s)
    return n

out = []
for i in range(10000):
    out.append(sn(i))

res = set(range(1, 10001))
aa = sorted(res - set(out))
for i in aa:
    print(i)