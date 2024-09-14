#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('python -m spacy download en_core_web_sm')


# In[2]:


import os
import cv2

def get_video_info(video_file):
    # Open video file
    cap = cv2.VideoCapture(video_file)

    # Get video duration (in seconds)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps if fps != 0 else 0

    # Get video resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    resolution = f"{width}x{height}"

    # Release video capture object
    cap.release()

    return duration, resolution, fps

def generate_experiment_table(video_dir):
    table_header = "| Video Filename | Duration (s) | Resolution (pixels) | Frame Rate (fps) |\n"
    table_header += "|----------------|--------------|---------------------|------------------|\n"

    max_filename_len = 15 
    max_duration_len = 10 
    max_resolution_len = 18  
    max_fps_len = 10 

    table_content = ""
    for video_file in os.listdir(video_dir):
        if video_file.endswith(".mp4"):
            video_path = os.path.join(video_dir, video_file)
            duration, resolution, fps = get_video_info(video_path)
            table_content += f"| {video_file.ljust(max_filename_len)} | {duration:.2f} | {resolution.ljust(max_resolution_len)} | {fps:.2f} |\n"

    experiment_table = table_header + table_content
    return experiment_table


video_directory = r"C:\Users\Felloh\Documents\Writing samples\Video generation\TrainValVideo"
result_table = generate_experiment_table(video_directory)
print(result_table)


# In[3]:


import spacy

# Loading the English language model
nlp = spacy.load("en_core_web_sm")

# Defining a function for NLP preprocessing
def preprocess_text(text):
    """
    Perform NLP preprocessing on the input text.
    
    Args:
    text (str): Input text to preprocess.
    
    Returns:
    spacy.tokens.doc.Doc: Processed spaCy Doc object.
    """
    # Applying spaCy NLP pipeline to the input text
    doc = nlp(text)
    return doc


example_text = "A person riding a bicycle along a scenic mountain trail."

# Preprocessing the example text
processed_doc = preprocess_text(example_text)

# Printing tokenized text
print("Tokenized Text:")
for token in processed_doc:
    print(token.text)


# In[4]:


import spacy

# Loading the English language model
nlp = spacy.load("en_core_web_sm")

# Defining a function for NLP preprocessing
def preprocess_text(text):
    """
    Perform NLP preprocessing on the input text.
    
    Args:
    text (str): Input text to preprocess.
    
    Returns:
    spacy.tokens.doc.Doc: Processed spaCy Doc object.
    """
    # Applying spaCy NLP pipeline to the input text
    doc = nlp(text)
    # Removing stop words and punctuation
    tokens = [token for token in doc if not token.is_stop and not token.is_punct]
    return tokens

example_text = "A person riding a bicycle along a scenic mountain trail."

# Preprocessing
processed_tokens = preprocess_text(example_text)

# Printing tokenized text after preprocessing
print("Tokenized Text:")
for token in processed_tokens:
    print(token.text)


# In[5]:


import os

# Define the dataset directory
dataset_directory = r"C:\Users\Felloh\Documents\Writing samples\Video generation\TrainValVideo"

# Define a function to iterate over the videos in the dataset directory
def process_dataset(dataset_dir):
    for video_file in os.listdir(dataset_dir):
        if video_file.endswith(".mp4"):
            video_path = os.path.join(dataset_dir, video_file)
            # Process each video file here
            print("Processing:", video_path)
            # You can add your processing code here

# Process the dataset
process_dataset(dataset_directory)


# In[1]:


import os
import cv2
import matplotlib.pyplot as plt

# Defining the dataset directory
dataset_directory = r"C:\Users\Felloh\Documents\Writing samples\Video generation\TrainValVideo"

# Function to get video durations from the dataset
def get_video_durations(dataset_dir):
    durations = []
    for video_file in os.listdir(dataset_dir):
        if video_file.endswith(".mp4"):
            video_path = os.path.join(dataset_dir, video_file)
            # Opening video file
            cap = cv2.VideoCapture(video_path)
            # Getting frame count
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # Getting frames per second (fps)
            fps = cap.get(cv2.CAP_PROP_FPS)
            # Checking if fps is not zero before division
            if fps != 0:
                # Computing duration and append to list
                duration = frame_count / fps
                durations.append(duration)
            # Releasing video capture object
            cap.release()
    return durations

# Getting video durations from the dataset
video_durations = get_video_durations(dataset_directory)

# Plotting the distribution of video durations
plt.figure(figsize=(10, 6))
plt.hist(video_durations, bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Duration (seconds)')
plt.ylabel('Number of Videos')
plt.title('Distribution of Video Durations')
plt.grid(True)
plt.show()


# In[ ]:


import os
import cv2
import numpy as np

# Defining the directory containing the original videos
video_dir = r"C:\Users\Felloh\Documents\Writing samples\Video generation\TrainValVideo"

# Defining the directory to save augmented videos
augmented_dir = r"C:\Users\Felloh\Documents\Writing samples\Video generation\AugmentedVideos"

# Ensuring the output directory exists, create it if necessary
os.makedirs(augmented_dir, exist_ok=True)

# Function to perform data augmentation on a video
def augment_video(video_path, output_dir):
    # Opening the original video
    cap = cv2.VideoCapture(video_path)
    
    # Getting video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Defining transformation parameters (e.g., flip horizontally)
    flip = True
    
    # Defining the output video path
    video_name = os.path.basename(video_path)
    output_path = os.path.join(output_dir, video_name)
    
    # Defining the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Reading and processing each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Applying data augmentation (e.g., flip horizontally)
        if flip:
            frame = cv2.flip(frame, 1)  # Flip horizontally
        
        # Writing the augmented frame to the output video
        out.write(frame)
    
    # Releasing video capture and writer objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Applying data augmentation to each video in the original directory
for video_file in os.listdir(video_dir):
    if video_file.endswith(".mp4"):
        video_path = os.path.join(video_dir, video_file)
        augment_video(video_path, augmented_dir)

print("Data augmentation completed successfully.")


# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Defining the parameter grid
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf', 'linear']}

# Creating the model
svm_model = SVC()

# Performing grid search
grid_search = GridSearchCV(svm_model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Getting the best parameters
best_params = grid_search.best_params_
print("Best Parameters:", best_params)


# In[ ]:




