import time
def egg():
    f = open('poem.txt', encoding='UTF8')

    lines = f.readlines()
    for i in range(len(lines)):
        print(lines[i])
        time.sleep(1)

if __name__ == "__main__":
    egg()