import keyboard
import pygame

# Initialize pygame mixer
pygame.mixer.init()
pygame.mixer.music.load("Audio/voicebooking-speech (2).mp3")

print("Press UP arrow to play sound (ESC to quit).")

while True:
    if keyboard.is_pressed("up"):
        print("Up arrow pressed → Playing sound...")
        pygame.mixer.music.play()
    elif keyboard.is_pressed("esc"):
        print("Exiting...")
        break
