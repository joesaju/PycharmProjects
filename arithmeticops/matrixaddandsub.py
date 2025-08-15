# Define two matrices
x = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

y = [
    [9, 8, 7],
    [6, 5, 4],
    [3, 2, 1]
]

# Create empty result matrices for addition and subtraction
add_result = []
sub_result = []

# Loop through rows
for i in range(len(x)):
    # Temporary lists for each row
    add_row = []
    sub_row = []

    # Loop through columns
    for j in range(len(x[0])):
        # Addition
        add_row.append(x[i][j] + y[i][j])
        # Subtraction
        sub_row.append(x[i][j] - y[i][j])

    # Append row to results
    add_result.append(add_row)
    sub_result.append(sub_row)

# Print results
print("Matrix Addition:")
for row in add_result:
    print(row)

print("\nMatrix Subtraction:")
for row in sub_result:
    print(row)
