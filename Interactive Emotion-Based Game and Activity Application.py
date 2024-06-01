import cv2
import numpy as np
import random
import pygame
import tensorflow as tf
from keras.models import load_model
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip
import mediapipe as mp
import tkinter as tk
from tkinter import messagebox
import time
# Initialize pygame
pygame.init()

def play_music():
    pygame.mixer.music.play(-1)

# Initialize MediaPipe hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Function to check if the fingers are curled (fist)
def is_fist(hand_landmarks):
    # Define landmarks for thumb, index, middle, ring, and pinky fingers
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    # Calculate distances between finger tips and the base of the fingers
    index_dist = np.linalg.norm(np.array(index_tip.x) - np.array(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x))
    middle_dist = np.linalg.norm(np.array(middle_tip.x) - np.array(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x))
    ring_dist = np.linalg.norm(np.array(ring_tip.x) - np.array(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x))
    pinky_dist = np.linalg.norm(np.array(pinky_tip.x) - np.array(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x))

    # Check if the fingers are curled (fist)
    if index_dist < 0.03 and middle_dist < 0.03 and ring_dist < 0.03 and pinky_dist < 0.03:
        return True
    else:
        return False
    

# Main function to run the boxing game

def boxing_game():
    cap = cv2.VideoCapture(0)

    # Create variables to store the score and hand-touch status
    score = 0
    left_hand_touched = False
    right_hand_touched = False

    # Define the position and size of the boxing bag
    boxing_bag_x = 300
    boxing_bag_y = 200
    boxing_bag_width = 100
    boxing_bag_height = 150

    game_over = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a more natural interaction
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB for hand detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hand landmarks
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on the frame
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Check hand gesture
                if is_fist(hand_landmarks):
                    cv2.putText(frame, "Fist (Punch)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Check if hand is touching the boxing bag
                hand_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1]
                hand_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0]
                if (boxing_bag_x < hand_x < boxing_bag_x + boxing_bag_width) and (boxing_bag_y < hand_y < boxing_bag_y + boxing_bag_height):
                    if results.multi_hand_landmarks.index(hand_landmarks) == 0 and not left_hand_touched:
                        score += 1
                        left_hand_touched = True
                        print("Left hand touched! Score:", score)
                    elif results.multi_hand_landmarks.index(hand_landmarks) == 1 and not right_hand_touched:
                        score += 1
                        right_hand_touched = True
                        print("Right hand touched! Score:", score)
                else:
                    if results.multi_hand_landmarks.index(hand_landmarks) == 0:
                        left_hand_touched = False
                    elif results.multi_hand_landmarks.index(hand_landmarks) == 1:
                        right_hand_touched = False

        # Draw the boxing bag on the frame
        cv2.rectangle(frame, (boxing_bag_x, boxing_bag_y), (boxing_bag_x + boxing_bag_width, boxing_bag_y + boxing_bag_height), (0, 0, 255), 2)

        # Detect face and emotion
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face = gray[y:y + h, x:x + w]
            emotion = detect_emotion(face)
            cv2.putText(frame, f"Emotion: {emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            if emotion == 'Happy':
                cv2.putText(frame, "You are better", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "Game over", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.namedWindow('Boxing Game', cv2.WINDOW_NORMAL)
                cv2.setWindowProperty('Boxing Game', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.imshow('Boxing Game', frame)
                cv2.waitKey(5000)  # Display the message for 5 seconds
                game_over = True
                break
                

        # Show the frame
        cv2.namedWindow('Boxing Game', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Boxing Game', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Boxing Game', frame)

        # Check for exit key (Esc)
        if cv2.waitKey(1) == 27:
            break

    # Release the video capture object and close the OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Initialize the face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the emotion detection model
emotion_model = load_model('Models/emotion_detection_model.h5')

# Dictionary to convert emotion model output to emotion label
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def detect_emotion(face):
    face = cv2.resize(face, (48, 48))
    face = face / 255.0
    face = np.expand_dims(face, axis=0)
    face = np.expand_dims(face, axis=-1)
    prediction = emotion_model.predict(face)
    emotion_label = emotion_labels[np.argmax(prediction)]
    return emotion_label


def play_music_and_video(music_path, video_path):
    # Load and play music
    pygame.mixer.music.load(music_path)
    pygame.mixer.music.play(-1)  # Loop the music indefinitely

    # Open video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    cv2.namedWindow('Relaxing Video', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Relaxing Video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Relaxing Video', frame)
        if cv2.waitKey(30) & 0xFF == 27:  # ESC key to stop the video and music
            break

    # Release video capture and stop music
    cap.release()
    pygame.mixer.music.stop()
    cv2.destroyWindow('Relaxing Video')

    # Ask the user if they feel better
    ask_if_better()

def celebration_animation():
    # Load the GIF
    gif_path = "Gif/be-happy.gif"
    gif = cv2.VideoCapture(gif_path)

    if not gif.isOpened():
        print("Error: Could not open GIF.")
        return

    # Get the dimensions of the GIF
    width = int(gif.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(gif.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a full-screen window
    cv2.namedWindow('Celebration', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Celebration', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = gif.read()
        if not ret:
            break
        cv2.imshow('Celebration', frame)
        if cv2.waitKey(30) & 0xFF == 27:  # ESC key to exit the animation
            break

    gif.release()
    cv2.destroyWindow('Celebration')


def ask_if_better():
    cv2.destroyAllWindows()  # Close all OpenCV windows

    def on_yes():
        # Show celebration animation
        celebration_animation()
        main()

    def on_no():
        root.destroy()
        ask_next_action()

    root = tk.Tk()
    root.withdraw()  # Hide the main window

    if messagebox.askyesno("Question", "Do you feel better now?"):
        on_yes()
    else:
        on_no()

def Sadness_animation():
    # Load the GIF
    gif_path = "Gif/excuse-me-86_512.gif"
    gif = cv2.VideoCapture(gif_path)

    if not gif.isOpened():
        print("Error: Could not open GIF.")
        return

    # Get the dimensions of the GIF
    width = int(gif.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(gif.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a full-screen window
    cv2.namedWindow('Sadness', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Sadness', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = gif.read()
        if not ret:
            break
        cv2.imshow('Sadness', frame)
        if cv2.waitKey(500) & 0xFF == 27:  # ESC key to exit the animation
            break

    gif.release()
    cv2.destroyWindow('Sadness')


def ask_next_action():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    def on_music():
        play_music_and_video("Music/music.mp3",
                             "Video/video.mp4")
        root.destroy()

    def on_game():
        choose_level()
        fruit_game()
        root.destroy()

    response = messagebox.askquestion("Next Action", "Would you like to watch the video and listen to the music again or play a game?", icon='question')
    
    if response == 'yes':
        if messagebox.askyesno("Choose", "Press 'Yes' for Music, 'No' for Game"):
            on_music()
        else:
            on_game()
    else:
        Sadness_animation()
        main()
        

# Initialize MediaPipe pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def draw_pose(image, landmarks):
    if landmarks:
        mp.solutions.drawing_utils.draw_landmarks(image, landmarks, mp_pose.POSE_CONNECTIONS)

def draw_level_selection(frame, hand_landmarks):
    levels = ['Easy', 'Medium', 'Hard']
    block_height = 100
    block_width = screen_width // 3

    for i, level in enumerate(levels):
        x1 = i * block_width
        y1 = screen_height // 2 - block_height // 2
        x2 = x1 + block_width
        y2 = y1 + block_height

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(frame, level, (x1 + 50, y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if hand_landmarks:
            for hand_landmark in hand_landmarks:
                hand_x = int(hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * screen_width)
                hand_y = int(hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * screen_height)

                if x1 < hand_x < x2 and y1 < hand_y < y2 and is_fist(hand_landmark):
                    return levels[i]

    return None

def choose_level():
    global level
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        hand_landmarks = results.multi_hand_landmarks if results.multi_hand_landmarks else []

        selected_level = draw_level_selection(frame, hand_landmarks)
        if selected_level:
            level = {'Easy': 1, 'Medium': 2, 'Hard': 3}[selected_level]
            break

        cv2.namedWindow('Choose Level', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Choose Level', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Choose Level', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break

    cv2.destroyAllWindows()

# Load sound effects and background music
eat_sound = pygame.mixer.Sound("Music/eat_sound.wav")
fall_sound = pygame.mixer.Sound("Music/fall_sound.wav")
pygame.mixer.music.load("Music/background_music.mp3")

# Function to play eat sound
def play_eat_sound():
    eat_sound.play()

# Function to play fall sound
def play_fall_sound():
    fall_sound.play()

# Load the fruit images
fruit_images = {
    'apple': cv2.imread("Images/apple.png", -1),
    'banana': cv2.imread("Images/banana.png", -1),
    'strawberry': cv2.imread("Images/strawberry.png", -1),
    'mango': cv2.imread("Images/mango.png", -1)
}

# Resize fruit images
for key in fruit_images:
    fruit_images[key] = cv2.resize(fruit_images[key], (50, 50))

# Class to represent a falling fruit
class Fruit:
    def __init__(self, x, y, vx, vy, image):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.image = image
        self.angle = random.randint(0, 360)

# Function to rotate image
def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    return result

# Function to calculate distance between two points
def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Load the video stream
cap = cv2.VideoCapture(0)

# Get the screen dimensions
screen_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
screen_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


background_image = cv2.imread('Images/background.jpg')  # Load your background image
background_image = cv2.resize(background_image, (screen_width, screen_height))
# Initialize fruit list
fruits = []

# Initialize fruit eaten counter
fruits_eaten = 0

# Initialize fruits fallen counter
fruits_fallen = 0

level_duration = 20 

def fruit_game():
    global fruits, fruits_eaten, fruits_fallen , level
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Add background
        frame = cv2.addWeighted(frame, 0.5, background_image, 0.5, 0)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face_roi = gray[y:y+h, x:x+w]
            mouth_roi = gray[y+h//2:y+h, x:x+w]  # Assuming the mouth region is at the bottom half of the face

            # Detect emotion to check if the mouth is open
            emotion = detect_emotion(face_roi)
            if emotion == 'Happy' or emotion == 'Surprise':  # Open mouth for catching fruits
                for fruit in fruits:
                    distance_to_mouth = calculate_distance(x + w // 2, y + h, fruit.x + fruit.image.shape[1] // 2, fruit.y + fruit.image.shape[0] // 2)
                    if distance_to_mouth < 50:
                        play_eat_sound()
                        fruits_eaten += 1
                        fruits.remove(fruit)
                        break

        # Spawn new fruits
        if random.random() < 0.02 * level:
            fruit_name = random.choice(list(fruit_images.keys()))
            fruit_image = fruit_images[fruit_name]
            fruit_x = random.randint(0, screen_width - fruit_image.shape[1])
            fruit = Fruit(fruit_x, 0, 0, random.randint(2, 5) * level, fruit_image)
            fruits.append(fruit)

        for fruit in fruits:
            fruit.x += fruit.vx
            fruit.y += fruit.vy
            fruit.angle += 5  # Rotate the fruit

            if fruit.y + fruit.image.shape[0] >= screen_height:
                fruits.remove(fruit)
                play_fall_sound()
                fruits_fallen += 1
                continue

            rotated_image = rotate_image(fruit.image, fruit.angle)
            y1, y2 = max(fruit.y, 0), min(fruit.y + rotated_image.shape[0], screen_height)
            x1, x2 = max(fruit.x, 0), min(fruit.x + rotated_image.shape[1], screen_width)

            if y1 < y2 and x1 < x2:
                fruit_resized = rotated_image[:y2 - y1, :x2 - x1]
                alpha_s = fruit_resized[:, :, 3] / 255.0
                alpha_l = 1.0 - alpha_s
                for c in range(0, 3):
                    frame[y1:y2, x1:x2, c] = (alpha_s * fruit_resized[:, :, c] +
                                              alpha_l * frame[y1:y2, x1:x2, c])

        # Display scores
        cv2.putText(frame, f"Level: {level}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Fruits Eaten: {fruits_eaten}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Fruits Fallen: {fruits_fallen}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        elapsed_time = time.time() - start_time
        remaining_time = max(level_duration - int(elapsed_time), 0)
        cv2.putText(frame, f"Time Left: {remaining_time}s", (screen_width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Check if time is up for the level
        if elapsed_time >= level_duration:
            if fruits_fallen > fruits_eaten:
                cv2.putText(frame, "Game Over", (screen_width // 2 - 150, screen_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
                cv2.imshow('Fruit Eating Game', frame)
                cv2.waitKey(5000)
                cv2.destroyAllWindows()
                main()
            else:
                cv2.putText(frame, "Awesome", (screen_width // 2 - 100, screen_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                cv2.putText(frame, "Let's go to the next level!", (screen_width // 2 - 200, screen_height // 2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                level += 1
                fruits_eaten = 0
                fruits_fallen = 0
                fruits = []
                start_time = time.time()
            cv2.namedWindow('Fruit Eating Game', cv2.WINDOW_NORMAL)
            cv2.setWindowProperty('Fruit Eating Game', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow('Fruit Eating Game', frame)
            cv2.waitKey(5000)


        cv2.namedWindow('Fruit Eating Game', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Fruit Eating Game', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Fruit Eating Game', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            cv2.destroyAllWindows()
            main()

    cap.release()
    cv2.destroyAllWindows()

def main():
    global cap
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Emotion Detection', frame)

        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face = gray[y:y + h, x:x + w]
            emotion = detect_emotion(face)
            cv2.putText(frame, f"Emotion: {emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            if emotion == 'Happy':
                cv2.putText(frame, "You look happy! Let's have a party!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.namedWindow('Emotion Detection', cv2.WINDOW_NORMAL)
                cv2.setWindowProperty('Emotion Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.imshow('Emotion Detection', frame)
                cv2.waitKey(3000)
                cv2.destroyWindow('Emotion Detection')  # Close the current window
                play_music() # Assuming you have a function called play_music() to play the music


                out = cv2.VideoWriter('party_video.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (screen_width, screen_height))
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Pose detection
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(rgb_frame)

                    if results.pose_landmarks:
                        draw_pose(frame, results.pose_landmarks)

                    out.write(frame)
                    cv2.imshow('Party Recording', frame)
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to stop recording
                        break
                out.release()
                pygame.mixer.music.stop()

                # Merge video and audio
                video = VideoFileClip("party_video.avi")
                audio = AudioFileClip("Music/background_music.mp3")
                final_video = video.set_audio(audio)
                final_video.write_videofile("party_video_with_music.mp4", codec="libx264")
            elif emotion == 'Sad':
                cv2.putText(frame, "You look sad. What would you prefer to do?", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, "Press 'm' for Music, 'g' for Game", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.namedWindow('Emotion Detection', cv2.WINDOW_NORMAL)
                cv2.setWindowProperty('Emotion Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.imshow('Emotion Detection', frame)

                while True:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('m'):
                        cv2.destroyAllWindows() 
                        
                        play_music_and_video("Music/music.mp3","Video/video.mp4")
                        while True:
                            for event in pygame.event.get():
                                if event.type == pygame.QUIT:
                                    pygame.quit()
                                    return
                            if cv2.waitKey(1) & 0xFF == 27:  # ESC key to stop music
                                pygame.mixer.music.stop()
                                
                                break
                    elif key == ord('g'):
                        choose_level()
                        fruit_game()
                        break
                    elif key == 27:  # ESC key to exit
                        break
            elif emotion == 'Angry':  # Added condition for calling boxing_game() when emotion is 'Angry'
                cv2.putText(frame, "You look angry! Let's play boxing!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.namedWindow('Emotion Detection', cv2.WINDOW_NORMAL)
                cv2.setWindowProperty('Emotion Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.imshow('Emotion Detection', frame)
                cv2.waitKey(3000)
                boxing_game()
                break
        else:
            cv2.putText(frame, "No face detected.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
