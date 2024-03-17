f = open('result.txt', 'r')
dat = f.read()
f.close()
arr = dat.split()
score = 0
for i in range(100):
    if int(arr[i]) == i//10:
        score = score+1
print(score)