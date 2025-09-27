import wikipedia
import pyttsx3

# import pygame
# import speach_recognization

# initialize text-to-speech
engine = pyttsx3.init()

# ask user
query = input("Ask Wikipedia: ")

# get summary (first 2 sentences)
result = wikipedia.summary(query, sentences=2)

print("Wikipedia says:", result)
engine.say(result)
engine.runAndWait()