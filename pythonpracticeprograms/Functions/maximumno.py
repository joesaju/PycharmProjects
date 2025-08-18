def find_max(numbers):
    max_value = numbers[0]

    for num in numbers:
        if num > max_value:
            max_value = num
    return max_value

nums =[10,34,20,69,44,50]
print("Maximum value: ",find_max(nums))