x = [[1,2,3],[4,5,6],[7,8,9]]
y = [[9,8,7],[6,5,4],[3,2,1]]
res = [[0]*3 for _ in range(3)]
for i in range(len(x)):
    for j in range(len(x[0])):
        res[i][j] = x[i][j] + y[i][j]

for row in res:
    print(row)