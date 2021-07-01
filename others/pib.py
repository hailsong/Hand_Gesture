import time

def pib_1(n):
    if n == 1:
        return 1
    elif n == 2:
        return 1
    else:
        return pib_1(n-1) + pib_1(n-2)

ans = []
def pib_2(n):
    if n == 1:
        ans.append(1)
        return 1
    elif n == 2:
        ans.append(1)
        return 1
    else:
        now = ans[n-2] + ans[n-3]
        ans.append(now)
        return now

start_time = time.time()

for i in range(1, 30):
    print(pib_1(i))

print('Calculation time is {}s'.format(time.time() - start_time))