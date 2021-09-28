word = []
sort = []
for _ in range(int(input())):
    word.append(input())

word = set(word)

for w in word:
    sort.append((len(w), w))

sort.sort(key = lambda word: (word[0], word[1]))

for l, w in sort:
    print(w)