n=int(input())
x=list(map(int,input().split()))
xt=list(sorted(set(x)))
print(xt)
xt={xt[i]:i for i in range(len(xt))}
print(*[xt[i] for i in x])