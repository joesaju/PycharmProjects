import random
print("welcome to the number guessing game. \n You have to guess correct number in a given 7 chance.")

low=int(input("enter the lower bound:"))
high = int(input("enter the high bound:"))

print(f"your have 7 chances to guess the number between {low} and {high} and the game begins!!")

num = random.randint(low,high)
gc=0
ch=7

while gc<ch:
    gc+=1
    guess = int(input("enter your correct guess: "))
    if(guess==num):
        print("correct guess!!!")
        break
    elif gc>=ch and guess!=num:
        print(f"sorry the num was {num} better luck next time")
    elif guess<num:
        print("too low, try high value")
    elif guess>num:
        print("too high, try low value")



