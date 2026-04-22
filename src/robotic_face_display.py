import pygame
import time
import sys
import math
import random
import cv2
import numpy as np
from deepface import DeepFace
import threading
import queue

# Initialize pygame
pygame.init()

# Set up the display for Wave share 5-inch DSI (800x480 typically)
WIDTH, HEIGHT = 800, 480
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# Colors with expanded palette
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (110, 203, 227)
DARK_BLUE = (80, 150, 200)
GREEN = (105, 240, 105)
LIGHT_GREEN = (150, 255, 150)
BRIGHT_GREEN = (120, 255, 120)
PINK = (255, 200, 200)
BRIGHT_PINK = (255, 150, 150)
YELLOW = (255, 255, 100)
ORANGE = (255, 180, 80)
RED = (255, 100, 100)
PURPLE = (200, 100, 255)
GRAY = (200, 200, 200)

# Face parameters
face_center = (WIDTH // 2, HEIGHT // 2)
face_radius = min(WIDTH, HEIGHT) // 3
eye_spacing = face_radius * 0.6
eye_y_position = face_center[1] - face_radius * 0.1
eye_size_base = face_radius // 3

# Blinking parameters
blink_duration = 0.15  # seconds
last_blink = 0
blink_interval = random.uniform(2, 5)  # Random time between blinks
is_blinking = False

# For animated expressions
animation_frame = 0
max_frames = 60  # Increased for smoother animations
expression_start_time = 0
current_expression = "neutral"
transition_duration = 1.0  # seconds to transition between expressions
transition_progress = 1.0  # Start fully in current expression
previous_expression = "neutral"

# For random eye movements
eye_movement_timer = 0
eye_movement_interval = random.uniform(1, 3)
eye_target_position = (0, 0)
current_eye_position = (0, 0)

# For dynamic color changes
color_animation_speed = 0.05
iris_color_base = GREEN
iris_color_current = GREEN

# For particle effects during some expressions
particles = []

# For bouncy animations
bounce_factor = 0
bounce_direction = 1
bounce_speed = 0.1

# Queue for communication between threads
emotion_queue = queue.Queue()

# Flag to control emotion detection thread
running = True

# Map DeepFace emotions to our expressions
emotion_to_expression = {
    'angry': 'angry',
    'disgust': 'angry',
    'fear': 'surprised',
    'happy': 'happy',
    'sad': 'sad',
    'surprise': 'surprised',
    'neutral': 'neutral'
}

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    pygame.quit()
    sys.exit()

def lerp(a, b, t):
    """Linear interpolation between a and b with factor t (0-1)"""
    return a + (b - a) * t

def lerp_color(color1, color2, t):
    """Interpolate between two colors"""
    return (
        int(lerp(color1[0], color2[0], t)),
        int(lerp(color1[1], color2[1], t)),
        int(lerp(color1[2], color2[2], t))
    )

def detect_emotion_thread():
    """Thread function to detect emotions using DeepFace"""
    global running
    
    # Initialize counters for emotion stability
    emotion_counter = {}
    current_dominant_emotion = "neutral"
    stability_threshold = 3  # Number of consecutive detections to switch
    
    while running:
        try:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
                
            # Process frame once every 0.5 seconds (not every frame to reduce CPU usage)
            try:
                analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                
                # Get dominant emotion
                if isinstance(analysis, list):
                    # If multiple faces are detected, use the first one
                    emotions = analysis[0]['emotion']
                else:
                    emotions = analysis['emotion']
                
                # Find the emotion with the highest score
                dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
                
                # Update emotion counter for stability
                if dominant_emotion in emotion_counter:
                    emotion_counter[dominant_emotion] += 1
                else:
                    emotion_counter[dominant_emotion] = 1
                
                # Reset counters for other emotions
                for emotion in emotion_counter:
                    if emotion != dominant_emotion:
                        emotion_counter[emotion] = 0
                
                # Check if we have stable detection
                if emotion_counter.get(dominant_emotion, 0) >= stability_threshold:
                    if dominant_emotion != current_dominant_emotion:
                        current_dominant_emotion = dominant_emotion
                        # Map DeepFace emotion to our expression
                        mapped_expression = emotion_to_expression.get(dominant_emotion, "neutral")
                        # Add to queue for main thread to process
                        if not emotion_queue.full():
                            emotion_queue.put(mapped_expression)
                
            except Exception as e:
                # If DeepFace fails to detect a face or other error
                print(f"DeepFace error: {e}")
                # Default to neutral if there's an error
                if not emotion_queue.full():
                    emotion_queue.put("neutral")
                
            time.sleep(0.5)  # Wait before processing the next frame
            
        except Exception as e:
            print(f"Error in emotion detection thread: {e}")
            time.sleep(1)  # Wait a bit longer if there's an error

def create_particles(position, count, color, speed_range=(1, 3), size_range=(1, 4)):
    """Create particles at the given position"""
    # Ensure color is RGB tuple (without alpha for now)
    if isinstance(color, tuple) or isinstance(color, list):
        if len(color) >= 3:
            # Use only the RGB components
            particle_color = (color[0], color[1], color[2])
            # Add alpha if provided
            if len(color) == 4:
                particle_color = (color[0], color[1], color[2], color[3])
        else:
            # Invalid color, use default
            particle_color = (255, 255, 255)
    else:
        # Not a tuple/list, use default
        particle_color = (255, 255, 255)
    
    for _ in range(count):
        angle = random.uniform(0, math.pi * 2)
        speed = random.uniform(speed_range[0], speed_range[1])
        size = random.uniform(size_range[0], size_range[1])
        lifetime = random.uniform(0.5, 2.0) * 30  # frames
        particles.append({
            'pos': list(position),
            'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
            'size': size,
            'color': particle_color,
            'lifetime': lifetime,
            'max_lifetime': lifetime
        })

def update_particles():
    """Update and draw all particles"""
    global particles
    new_particles = []
    
    for p in particles:
        # Update position
        p['pos'][0] += p['vel'][0]
        p['pos'][1] += p['vel'][1]
        
        # Update lifetime
        p['lifetime'] -= 1
        
        # Calculate alpha based on remaining lifetime
        alpha = p['lifetime'] / p['max_lifetime']
        
        # Get base color and ensure it's properly formatted
        base_color = p['color']
        
        # Create properly formatted RGBA color
        if isinstance(base_color, tuple) or isinstance(base_color, list):
            if len(base_color) == 3:
                # It's RGB, add alpha
                color = (base_color[0], base_color[1], base_color[2], int(255 * alpha))
            elif len(base_color) == 4:
                # It's RGBA, modify alpha
                color = (base_color[0], base_color[1], base_color[2], int(base_color[3] * alpha))
            else:
                # Invalid color format, use default
                color = (255, 255, 255, int(255 * alpha))
        else:
            # Not a tuple/list, use default
            color = (255, 255, 255, int(255 * alpha))
        
        # Create a surface for the particle with alpha
        particle_surface = pygame.Surface((int(p['size'] * 2), int(p['size'] * 2)), pygame.SRCALPHA)
        
        # Draw the circle with the proper color format for pygame
        # For pygame.draw.circle
        pygame.draw.circle(particle_surface, (color[0], color[1], color[2]), 
                          (int(p['size']), int(p['size'])), int(p['size']))
        
        # Apply the alpha to the whole surface instead
        particle_surface.set_alpha(color[3])
        
        # Blit the particle to the screen
        screen.blit(particle_surface, (int(p['pos'][0] - p['size']), int(p['pos'][1] - p['size'])))
        
        # Keep particle if still alive
        if p['lifetime'] > 0:
            new_particles.append(p)
    
    particles = new_particles

def draw_face(expression, frame=0, transition=1.0):
    """Draw the robot face with the given expression"""
    global bounce_factor, bounce_direction, current_eye_position, iris_color_current
    
    # Apply a bouncy effect for the whole face
    if expression in ["happy", "excited"]:
        bounce_factor += bounce_speed * bounce_direction
        if bounce_factor > 1.0:
            bounce_factor = 1.0
            bounce_direction = -1
        elif bounce_factor < 0:
            bounce_factor = 0
            bounce_direction = 1
    else:
        bounce_factor = max(0, bounce_factor - bounce_speed)
        bounce_direction = 1
    
    # Apply bounce to the face position
    bounce_offset = math.sin(bounce_factor * math.pi) * 10
    face_position = (face_center[0], face_center[1] - bounce_offset)
    
    # Clear screen with background gradient
    if expression == "excited":
        for y in range(HEIGHT):
            color_factor = y / HEIGHT
            color = lerp_color(YELLOW, ORANGE, color_factor)
            pygame.draw.line(screen, color, (0, y), (WIDTH, y))
    elif expression == "happy":
        for y in range(HEIGHT):
            color_factor = y / HEIGHT
            color = lerp_color(LIGHT_GREEN, WHITE, color_factor)
            pygame.draw.line(screen, color, (0, y), (WIDTH, y))
    else:
        screen.fill(WHITE)
    
    # Animation progress (0.0 to 1.0)
    progress = frame / max_frames
    
    # Calculate eye positions
    left_eye_pos = (face_position[0] - eye_spacing, eye_y_position)
    right_eye_pos = (face_position[0] + eye_spacing, eye_y_position)
    
    # Create glow effect for face outline
    for i in range(3):
        glow_size = face_radius + 20 - i * 5
        glow_alpha = 100 - i * 30
        glow_color = None
        
        if expression == "happy":
            glow_color = (LIGHT_GREEN[0], LIGHT_GREEN[1], LIGHT_GREEN[2], glow_alpha)
        elif expression == "excited":
            glow_color = (YELLOW[0], YELLOW[1], YELLOW[2], glow_alpha)
        elif expression == "surprised":
            glow_color = (BRIGHT_PINK[0], BRIGHT_PINK[1], BRIGHT_PINK[2], glow_alpha)
        elif expression == "sad":
            glow_color = (BLUE[0], BLUE[1], BLUE[2], glow_alpha)
        elif expression == "angry":
            glow_color = (RED[0], RED[1], RED[2], glow_alpha)
        else:
            glow_color = (BLUE[0], BLUE[1], BLUE[2], glow_alpha)
        
        # Create temporary surface with alpha
        glow_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        pygame.draw.circle(glow_surface, glow_color, face_position, glow_size)
        screen.blit(glow_surface, (0, 0))
    
    # Draw face outline with dynamic color based on expression
    outline_color = BLUE
    if expression == "happy":
        outline_color = lerp_color(BLUE, GREEN, math.sin(frame * 0.1) * 0.3 + 0.7)
    elif expression == "excited":
        outline_color = lerp_color(YELLOW, ORANGE, math.sin(frame * 0.2) * 0.5 + 0.5)
    elif expression == "angry":
        outline_color = lerp_color(RED, DARK_BLUE, math.sin(frame * 0.1) * 0.3 + 0.7)
    elif expression == "sad":
        outline_color = BLUE
    
    # Draw face outline with a wavy effect
    if expression in ["excited", "happy"]:
        num_points = 60
        for i in range(num_points + 1):
            angle1 = 2 * math.pi * i / num_points
            angle2 = 2 * math.pi * ((i + 1) % num_points) / num_points
            
            # Add wave effect
            wave_radius = face_radius + 15
            if expression == "excited":
                wave_radius += math.sin(angle1 * 8 + frame * 0.2) * 5
            else:
                wave_radius += math.sin(angle1 * 6 + frame * 0.1) * 3
                
            x1 = face_position[0] + math.cos(angle1) * wave_radius
            y1 = face_position[1] + math.sin(angle1) * wave_radius
            x2 = face_position[0] + math.cos(angle2) * wave_radius
            y2 = face_position[1] + math.sin(angle2) * wave_radius
            
            pygame.draw.line(screen, outline_color, (x1, y1), (x2, y2), 8)
    else:
        pygame.draw.circle(screen, outline_color, face_position, face_radius + 15, 15)
    
    # Eye size varies with expression and blinking
    eye_size = eye_size_base
    if is_blinking:
        eye_height = eye_size * 0.15
        eye_width = eye_size
    else:
        eye_height = eye_size
        eye_width = eye_size
        
        # Adjust eye size based on expression
        if expression == "surprised":
            eye_height = eye_size * 1.5 + math.sin(frame * 0.2) * eye_size * 0.1
            eye_width = eye_size * 1.5 + math.sin(frame * 0.2) * eye_size * 0.1
        elif expression == "happy":
            eye_height = eye_size * 0.9
            eye_width = eye_size * 1.1
        elif expression == "excited":
            eye_height = eye_size * 1.2 + math.sin(frame * 0.3) * eye_size * 0.15
            eye_width = eye_size * 1.2 + math.sin(frame * 0.3) * eye_size * 0.15
        elif expression == "sad":
            eye_height = eye_size * 0.8
            eye_width = eye_size * 0.9
        elif expression == "angry":
            eye_height = eye_size * 0.7 + math.sin(frame * 0.15) * eye_size * 0.05
            eye_width = eye_size * 1.2

    # Draw eye whites (backgrounds) with gradient
    for eye_pos in [left_eye_pos, right_eye_pos]:
        # Create gradient for eye whites
        for i in range(int(eye_height * 2)):
            y_pos = eye_pos[1] - eye_height + i
            ellipse_width = (1 - ((i / (eye_height * 2) - 0.5) ** 2) * 4) * eye_width * 2
            
            if ellipse_width > 0:
                pygame.draw.line(
                    screen,
                    WHITE,
                    (eye_pos[0] - ellipse_width/2, y_pos),
                    (eye_pos[0] + ellipse_width/2, y_pos)
                )
    
    # Draw eye outlines
    pygame.draw.ellipse(screen, BLACK, (left_eye_pos[0] - eye_width, left_eye_pos[1] - eye_height, eye_width * 2, eye_height * 2), 2)
    pygame.draw.ellipse(screen, BLACK, (right_eye_pos[0] - eye_width, right_eye_pos[1] - eye_height, eye_width * 2, eye_height * 2), 2)
    
    # Dynamic eye movement
    target_offset_x = 0
    target_offset_y = 0
    
    # Expression-specific eye positions
    if expression == "thinking":
        target_offset_y = -eye_size * 0.3 
        target_offset_x = eye_size * 0.3 + math.sin(frame * 0.1) * eye_size * 0.1
    elif expression == "sad":
        target_offset_y = eye_size * 0.2
    elif expression == "angry":
        target_offset_y = -eye_size * 0.2
        target_offset_x = math.sin(frame * 0.2) * eye_size * 0.15
    elif expression == "excited":
        target_offset_x = math.sin(frame * 0.3) * eye_size * 0.3
        target_offset_y = math.cos(frame * 0.25) * eye_size * 0.2
    elif expression == "surprised":
        # Surprised eyes dart around quickly
        if frame % 15 == 0:
            target_offset_x = random.uniform(-0.3, 0.3) * eye_size
            target_offset_y = random.uniform(-0.3, 0.3) * eye_size
    
    # Smoothly move to target position
    current_eye_position = (
        lerp(current_eye_position[0], target_offset_x, 0.1),
        lerp(current_eye_position[1], target_offset_y, 0.1)
    )
    
    # Draw green irises with dynamic color based on expression
    iris_size = eye_size * 0.7
    if not is_blinking:
        # Set target iris color based on expression
        target_iris_color = GREEN
        if expression == "happy":
            target_iris_color = BRIGHT_GREEN
        elif expression == "excited":
            target_iris_color = YELLOW
        elif expression == "angry":
            target_iris_color = RED
        elif expression == "sad":
            target_iris_color = BLUE
        
        # Smoothly transition iris color
        iris_color_current = lerp_color(iris_color_current, target_iris_color, 0.1)
        
        # Apply some subtle animation to iris position
        iris_offset_x = current_eye_position[0]
        iris_offset_y = current_eye_position[1]
        
        # Add subtle oscillation
        if expression != "neutral":
            iris_offset_x += math.sin(frame * 0.1) * eye_size * 0.05
            iris_offset_y += math.cos(frame * 0.08) * eye_size * 0.05
        
        left_iris_pos = (left_eye_pos[0] + iris_offset_x, left_eye_pos[1] + iris_offset_y)
        right_iris_pos = (right_eye_pos[0] + iris_offset_x, right_eye_pos[1] + iris_offset_y)
        
        # Draw iris with radial gradient
        for i in range(int(iris_size), 0, -1):
            factor = i / iris_size
            gradient_color = lerp_color(WHITE, iris_color_current, factor)
            pygame.draw.circle(screen, gradient_color, left_iris_pos, i)
            pygame.draw.circle(screen, gradient_color, right_iris_pos, i)
        
        # Draw highlights in eyes
        highlight_size = iris_size * 0.4
        highlight_offset = iris_size * 0.3
        
        # Primary highlight
        pygame.draw.circle(screen, WHITE, (left_iris_pos[0] - highlight_offset, left_iris_pos[1] - highlight_offset), highlight_size)
        pygame.draw.circle(screen, WHITE, (right_iris_pos[0] - highlight_offset, right_iris_pos[1] - highlight_offset), highlight_size)
        
        # Secondary smaller highlight
        pygame.draw.circle(screen, WHITE, (left_iris_pos[0] + highlight_offset/2, left_iris_pos[1] + highlight_offset/2), highlight_size/2)
        pygame.draw.circle(screen, WHITE, (right_iris_pos[0] + highlight_offset/2, right_iris_pos[1] + highlight_offset/2), highlight_size/2)
        
        # Draw pupils with pulsing effect
        pupil_base_size = iris_size * 0.5
        pupil_pulse = 0
        
        if expression == "surprised":
            pupil_pulse = math.sin(frame * 0.2) * 0.2
        elif expression == "excited":
            pupil_pulse = math.sin(frame * 0.3) * 0.3
            
        pupil_size = pupil_base_size * (1.0 + pupil_pulse)
        pygame.draw.circle(screen, BLACK, left_iris_pos, pupil_size)
        pygame.draw.circle(screen, BLACK, right_iris_pos, pupil_size)
    
    # Draw eyebrows based on expression
    eyebrow_width = eye_size * 1.8
    eyebrow_height = eye_size * 0.4
    eyebrow_y_base = left_eye_pos[1] - eye_height - eyebrow_height * 1.5
    
    # Define eyebrow animation based on expression
    left_eyebrow_points = []
    right_eyebrow_points = []
    
    if expression == "neutral":
        # Straight eyebrows with slight movement
        subtle_move = math.sin(frame * 0.1) * eyebrow_height * 0.1
        left_eyebrow_points = [
            (left_eye_pos[0] - eyebrow_width/2, eyebrow_y_base + subtle_move),
            (left_eye_pos[0] + eyebrow_width/2, eyebrow_y_base + subtle_move),
            (left_eye_pos[0] + eyebrow_width/2, eyebrow_y_base - eyebrow_height + subtle_move),
            (left_eye_pos[0] - eyebrow_width/2, eyebrow_y_base - eyebrow_height + subtle_move)
        ]
        right_eyebrow_points = [
            (right_eye_pos[0] - eyebrow_width/2, eyebrow_y_base + subtle_move),
            (right_eye_pos[0] + eyebrow_width/2, eyebrow_y_base + subtle_move),
            (right_eye_pos[0] + eyebrow_width/2, eyebrow_y_base - eyebrow_height + subtle_move),
            (right_eye_pos[0] - eyebrow_width/2, eyebrow_y_base - eyebrow_height + subtle_move)
        ]
    elif expression == "surprised":
        # Highly raised eyebrows with movement
        raise_amount = eyebrow_height * 1.5 + math.sin(frame * 0.2) * eyebrow_height * 0.3
        eyebrow_y_base -= raise_amount
        left_eyebrow_points = [
            (left_eye_pos[0] - eyebrow_width/2, eyebrow_y_base),
            (left_eye_pos[0] + eyebrow_width/2, eyebrow_y_base),
            (left_eye_pos[0] + eyebrow_width/2, eyebrow_y_base - eyebrow_height),
            (left_eye_pos[0] - eyebrow_width/2, eyebrow_y_base - eyebrow_height)
        ]
        right_eyebrow_points = [
            (right_eye_pos[0] - eyebrow_width/2, eyebrow_y_base),
            (right_eye_pos[0] + eyebrow_width/2, eyebrow_y_base),
            (right_eye_pos[0] + eyebrow_width/2, eyebrow_y_base - eyebrow_height),
            (right_eye_pos[0] - eyebrow_width/2, eyebrow_y_base - eyebrow_height)
        ]
    elif expression == "happy" or expression == "excited":
        # Dynamic raised eyebrows
        move_factor = 0.5
        if expression == "excited":
            move_factor = 0.8 + math.sin(frame * 0.2) * 0.2
            
        left_eyebrow_points = [
            (left_eye_pos[0] - eyebrow_width/2, eyebrow_y_base),
            (left_eye_pos[0] + eyebrow_width/2, eyebrow_y_base - eyebrow_height * move_factor),
            (left_eye_pos[0] + eyebrow_width/2, eyebrow_y_base - eyebrow_height * (1 + move_factor)),
            (left_eye_pos[0] - eyebrow_width/2, eyebrow_y_base - eyebrow_height)
        ]
        right_eyebrow_points = [
            (right_eye_pos[0] - eyebrow_width/2, eyebrow_y_base - eyebrow_height * move_factor),
            (right_eye_pos[0] + eyebrow_width/2, eyebrow_y_base),
            (right_eye_pos[0] + eyebrow_width/2, eyebrow_y_base - eyebrow_height),
            (right_eye_pos[0] - eyebrow_width/2, eyebrow_y_base - eyebrow_height * (1 + move_factor))
        ]
    elif expression == "sad":
        # Animated sad eyebrows
        droop = 0.5 + math.sin(frame * 0.1) * 0.1
        left_eyebrow_points = [
            (left_eye_pos[0] - eyebrow_width/2, eyebrow_y_base - eyebrow_height * droop),
            (left_eye_pos[0] + eyebrow_width/2, eyebrow_y_base + eyebrow_height * droop),
            (left_eye_pos[0] + eyebrow_width/2, eyebrow_y_base + eyebrow_height * droop - eyebrow_height),
            (left_eye_pos[0] - eyebrow_width/2, eyebrow_y_base - eyebrow_height * droop - eyebrow_height)
        ]
        right_eyebrow_points = [
            (right_eye_pos[0] - eyebrow_width/2, eyebrow_y_base + eyebrow_height * droop),
            (right_eye_pos[0] + eyebrow_width/2, eyebrow_y_base - eyebrow_height * droop),
            (right_eye_pos[0] + eyebrow_width/2, eyebrow_y_base - eyebrow_height * droop - eyebrow_height),
            (right_eye_pos[0] - eyebrow_width/2, eyebrow_y_base + eyebrow_height * droop - eyebrow_height)
        ]
    elif expression == "angry":
        # Dynamic angry eyebrows
        fury = 1.0 + math.sin(frame * 0.15) * 0.2
        left_eyebrow_points = [
            (left_eye_pos[0] - eyebrow_width/2, eyebrow_y_base),
            (left_eye_pos[0] + eyebrow_width/2, eyebrow_y_base + eyebrow_height * fury),
            (left_eye_pos[0] + eyebrow_width/2, eyebrow_y_base + eyebrow_height * fury - eyebrow_height),
            (left_eye_pos[0] - eyebrow_width/2, eyebrow_y_base - eyebrow_height)
        ]
        right_eyebrow_points = [
            (right_eye_pos[0] - eyebrow_width/2, eyebrow_y_base + eyebrow_height * fury),
            (right_eye_pos[0] + eyebrow_width/2, eyebrow_y_base),
            (right_eye_pos[0] + eyebrow_width/2, eyebrow_y_base - eyebrow_height),
            (right_eye_pos[0] - eyebrow_width/2, eyebrow_y_base + eyebrow_height * fury - eyebrow_height)
        ]
    elif expression == "thinking":
        # One raised eyebrow with motion
        raise_amount = 1.0 + math.sin(frame * 0.1) * 0.2
        left_eyebrow_points = [
            (left_eye_pos[0] - eyebrow_width/2, eyebrow_y_base),
            (left_eye_pos[0] + eyebrow_width/2, eyebrow_y_base - eyebrow_height * raise_amount),
            (left_eye_pos[0] + eyebrow_width/2, eyebrow_y_base - eyebrow_height * (1 + raise_amount)),
            (left_eye_pos[0] - eyebrow_width/2, eyebrow_y_base - eyebrow_height)
        ]
    
    # Draw eyebrows
    if left_eyebrow_points:
        pygame.draw.polygon(screen, BLACK, left_eyebrow_points)
    if right_eyebrow_points:
        pygame.draw.polygon(screen, BLACK, right_eyebrow_points)
    
    # Draw mouth based on expression
    mouth_width = face_radius * 0.8
    mouth_height = face_radius * 0.4
    mouth_y = face_position[1] + face_radius * 0.3
    
    if expression == "happy" or expression == "excited":
        # Happy smile with dynamic movement
        smile_factor = 0.6
        if expression == "excited":
            smile_factor = 0.8 + math.sin(frame * 0.2) * 0.2
            
        # Create curve points for smile
        points = []
        for i in range(-int(mouth_width/2), int(mouth_width/2) + 1, 5):
            x = face_position[0] + i
            # Parabola for smile: y = ax^2 + c
            y = mouth_height * smile_factor * (i / (mouth_width/2)) ** 2 + mouth_y - mouth_height * smile_factor
            points.append((x, y))
            
        if len(points) > 1:
            pygame.draw.lines(screen, BLACK, False, points, 3)
            
        # Add teeth for excited expression
        if expression == "excited" and frame % 60 < 40:  # Animate teeth visibility
            teeth_height = mouth_height * 0.3
            teeth_top_y = mouth_y - teeth_height/2
            teeth_spacing = mouth_width / 6
            
            for i in range(-2, 3):
                tooth_x = face_position[0] + i * teeth_spacing
                pygame.draw.rect(screen, WHITE, (tooth_x - teeth_spacing/3, teeth_top_y, teeth_spacing/1.5, teeth_height))
                pygame.draw.rect(screen, BLACK, (tooth_x - teeth_spacing/3, teeth_top_y, teeth_spacing/1.5, teeth_height), 1)
            
            # Add particle effects from mouth for excited expression
            if random.random() < 0.2:  # Occasional particles
                particle_pos = (face_position[0], mouth_y)
                create_particles(particle_pos, 5, YELLOW, speed_range=(1, 3), size_range=(2, 5))
                
    elif expression == "sad":
        # Sad frown with animation
        frown_factor = 0.4 + math.sin(frame * 0.1) * 0.1
        
        # Create curve points for frown
        points = []
        for i in range(-int(mouth_width/2), int(mouth_width/2) + 1, 5):
            x = face_position[0] + i
            # Inverted parabola for frown: y = -ax^2 + c
            y = -mouth_height * frown_factor * (i / (mouth_width/2)) ** 2 + mouth_y + mouth_height * frown_factor
            points.append((x, y))
            
        if len(points) > 1:
            pygame.draw.lines(screen, BLACK, False, points, 3)
            
        # Add occasional tear particles for sad expression
        if random.random() < 0.05:  # Rare tear drops
            tear_pos_left = (left_eye_pos[0], left_eye_pos[1] + eye_size)
            tear_pos_right = (right_eye_pos[0], right_eye_pos[1] + eye_size)
            
            if random.random() < 0.5:
                create_particles(tear_pos_left, 1, BLUE, speed_range=(0.5, 1), size_range=(3, 5))
            else:
                create_particles(tear_pos_right, 1, BLUE, speed_range=(0.5, 1), size_range=(3, 5))
    
    elif expression == "surprised":
        # O-shaped mouth with animation
        o_factor = 0.8 + math.sin(frame * 0.2) * 0.2
        mouth_rect = (
            face_position[0] - mouth_width * 0.25 * o_factor,
            mouth_y - mouth_height * 0.25 * o_factor,
            mouth_width * 0.5 * o_factor,
            mouth_height * 0.5 * o_factor
        )
        pygame.draw.ellipse(screen, BLACK, mouth_rect, 3)
        
    elif expression == "angry":
        # Angular upset mouth with movement
        anger_factor = 0.5 + math.sin(frame * 0.15) * 0.1
        
        mouth_points = [
            (face_position[0] - mouth_width * 0.4, mouth_y),
            (face_position[0], mouth_y + mouth_height * 0.3 * anger_factor),
            (face_position[0] + mouth_width * 0.4, mouth_y)
        ]
        pygame.draw.lines(screen, BLACK, False, mouth_points, 3)
        
        # Add steam particles from top of head for angry expression
        if random.random() < 0.1:
            steam_pos = (face_position[0], face_position[1] - face_radius - 10)
            create_particles(steam_pos, 3, (200, 200, 200), speed_range=(0.5, 2), size_range=(3, 6))
            
    else:  # neutral or thinking
        # Neutral mouth with subtle movement
        mouth_points = [
            (face_position[0] - mouth_width * 0.3, mouth_y + math.sin(frame * 0.1) * 2),
            (face_position[0] + mouth_width * 0.3, mouth_y + math.sin(frame * 0.1 + 1) * 2)
        ]
        pygame.draw.lines(screen, BLACK, False, mouth_points, 3)
    
    # Update particles
    update_particles()

def main():
    """Main program loop"""
    global is_blinking, last_blink, blink_interval
    global animation_frame, expression_start_time, current_expression
    global transition_progress, previous_expression
    global eye_movement_timer, eye_movement_interval, eye_target_position
    global running
    
    # Start the emotion detection thread
    emotion_thread = threading.Thread(target=detect_emotion_thread)
    emotion_thread.daemon = True
    emotion_thread.start()
    
    try:
        # Main game loop
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    cap.release()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                        pygame.quit()
                        cap.release()
                        sys.exit()
                    # Debug: Test different expressions with number keys
                    elif event.key == pygame.K_1:
                        previous_expression = current_expression
                        current_expression = "neutral"
                        transition_progress = 0.0
                    elif event.key == pygame.K_2:
                        previous_expression = current_expression
                        current_expression = "happy"
                        transition_progress = 0.0
                    elif event.key == pygame.K_3:
                        previous_expression = current_expression
                        current_expression = "sad"
                        transition_progress = 0.0
                    elif event.key == pygame.K_4:
                        previous_expression = current_expression
                        current_expression = "angry"
                        transition_progress = 0.0
                    elif event.key == pygame.K_5:
                        previous_expression = current_expression
                        current_expression = "surprised"
                        transition_progress = 0.0
                    elif event.key == pygame.K_6:
                        previous_expression = current_expression
                        current_expression = "thinking"
                        transition_progress = 0.0
                    elif event.key == pygame.K_7:
                        previous_expression = current_expression
                        current_expression = "excited"
                        transition_progress = 0.0
            
            # Check emotion queue for new emotions from the detection thread
            if not emotion_queue.empty():
                new_expression = emotion_queue.get()
                if new_expression != current_expression:
                    previous_expression = current_expression
                    current_expression = new_expression
                    transition_progress = 0.0
                    print(f"New expression: {current_expression}")
            
            # Update transition progress
            if transition_progress < 1.0:
                transition_progress += 1.0 / (transition_duration * 60)  # 60 FPS assumption
                transition_progress = min(1.0, transition_progress)
            
            # Handle random blinking
            current_time = time.time()
            if not is_blinking and current_time - last_blink > blink_interval:
                is_blinking = True
                blink_start_time = current_time
            elif is_blinking and current_time - last_blink > blink_duration:
                is_blinking = False
                last_blink = current_time
                blink_interval = random.uniform(2, 5)  # Random time until next blink
            
            # Random eye movements
            eye_movement_timer += 1
            if eye_movement_timer > eye_movement_interval:
                eye_target_position = (
                    random.uniform(-0.3, 0.3) * eye_size_base,
                    random.uniform(-0.3, 0.3) * eye_size_base
                )
                eye_movement_timer = 0
                eye_movement_interval = random.uniform(60, 180)  # Frames until next movement
            
            # Clear the screen
            screen.fill(WHITE)
            
            # Draw robot face with current expression and animation frame
            draw_face(current_expression, animation_frame, transition_progress)
            
            # Update animation frame counter
            animation_frame = (animation_frame + 1) % max_frames
            
            # Update display
            pygame.display.flip()
            
            # Cap the frame rate
            clock.tick(60)
    
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        # Clean up
        running = False
        pygame.quit()
        cap.release()
        sys.exit()

if __name__ == "__main__":
    main()