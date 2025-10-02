import keyboard
import pygame

# Initialize pygame mixer
pygame.mixer.init()

# Map toetsen naar bestanden
sound_map = {
    "up": "Audio/voicebooking-speech (1).mp3",
    "right": "Audio/voicebooking-speech (2).mp3",
    "down": "Audio/voicebooking-speech (3).mp3"
}

print("Controls:")
print("  UP arrow    → play file 1")
print("  RIGHT arrow → play file 2")
print("  DOWN arrow  → play file 3")
print("  ESC         → quit")

while True:
    # Loop over de toetsen in de map
    for key, file in sound_map.items():
        if keyboard.is_pressed(key):
            print(f"{key.upper()} pressed → Playing {file}")
            pygame.mixer.music.load(file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():  # wachten tot klaar
                if keyboard.is_pressed("esc"):
                    print("Exiting...")
                    exit()
    
    # Stoppen met ESC
    if keyboard.is_pressed("esc"):
        print("Exiting...")
        break
