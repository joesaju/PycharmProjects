rows = int(input("Enter number of rows: "))
cols = int(input("Enter number of columns: "))
matrix = []
print("entries row-wise")
for i in range(rows):#enter row values
    row=[]
    for j in range(cols):
        row.append(int(input()))
    matrix.append(row)

print("\nThe matrix is:")#display matrix
for i in range(rows):
    for j in range(cols):
        print(matrix[i][j], end=" ")
    print() 
