matrix1=[[0,0],[0,0]]
matrix2=[[0,0],[0,0]]

nrows=0
ncols=0

nrows=int(input("Enter the no.of rows:"))
ncols=int(input("enter the no.of cols:"))

for rows in range(nrows):
    for cols in range(ncols):
        v=int(input("enter the element:"))
        matrix1[rows][cols] = v

for rows in range(nrows):
    for cols in range(ncols):
        v = int(input("enter the element:"))
        matrix2[rows][cols]=v

for rows in range(len(matrix1)):
    for cols in range(len(matrix1[rows])):
        print(matrix1[rows][cols]+matrix2[rows][cols],end=" ")
    print()