price = [0, 1, 5, 8, 9, 10, 17, 17, 20, 24, 30]

def max_price(n):
    maxi = 0
    if n == 1:
        return 1
    for i in range(1, n+1):
        local = price[i] + max_price(n-i)
        maxi = max(maxi, local)
    print(n, maxi)
    return maxi

print(max_price(6))