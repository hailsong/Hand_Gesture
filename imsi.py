a = 10
b = 30

total = 0

for i in range(a, b+1):
    if i % 4 == 0:
        total += i

print(total)