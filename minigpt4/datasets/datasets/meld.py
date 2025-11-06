import glob
import os
import json
import pickle
import random
import time
import itertools
import pandas as pd
import json

import torch.nn.functional as F

import numpy as np
from PIL import Image
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
import torch
from torch.utils.data import Dataset
import webdataset as wds
import cv2

from minigpt4.datasets.datasets.base_dataset import BaseDataset

class MELDDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        self.vis_root = vis_root
        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.caption_instruction_pool = [
            "Please describe the details of the expression and tone the video.",
            "Can you provide a description of the facial expression and tone shown by the person in the video?",
            "Could you outline the facial expressions and vocal tones displayed in the video?",
            "Detail the expressions and tone used in the video.",
            "Explain the visual and auditory expressions captured in the video.",
            "Provide an analysis of the expressions and tone featured in the video.",
        ]

        # Match MER2024 emotion instruction format exactly
        self.emotion_instruction_pool = [
            "Please determine which emotion label in the video represents: happy, sad, neutral, angry, worried, surprise, fear, contempt.",
        ]

        self.reason_instruction_pool = [
            "Please analyze all the clues in the video and reason out the emotional label of the person in the video.",
            "What is the emotional state of the person in the video? Please tell me the reason.",
            "What are the facial expressions and vocal tone used in the video? What is the intended meaning behind his words? Which emotion does this reflect?",
            "Please integrate information from various modalities to infer the emotional category of the person in the video.",
            "Could you describe the emotion-related features of the individual in the video? What emotional category do they fall into?",
        ]

        # Match MER2024 task pool exactly
        self.task_pool = [
           "emotion",
        ]

        print("MELD ann_path: ", ann_path)
        self.ann_path = ann_path
        self.file_path = os.path.dirname(ann_path)
        self.feat_path = "/scratch/data/bikash_rs/vivek/dataset/Meld_feat_ext"
        
        # Read CSV file instead of text file
        self.meld_data = pd.read_csv(ann_path)
        print(f'MELD video number: {len(self.meld_data)}')

        # MELD emotions (7 classes) - keep original for ground truth
        emos = ['neutral', 'anger', 'joy', 'sadness', 'fear', 'surprise', 'disgust']
        self.emo2idx, self.idx2emo = {}, {}
        for ii, emo in enumerate(emos): 
            self.emo2idx[emo] = ii
        for ii, emo in enumerate(emos): 
            self.idx2emo[ii] = emo

        # MER2024 emotions for model prediction
        self.mer_emotions = ['neutral', 'angry', 'happy', 'sad', 'worried', 'surprise']

        # Create emotion mapping from MER2024 to MELD
        # self.mer_to_meld_mapping = {
        #     'neutral': 'neutral',
        #     'angry': 'anger', 
        #     'happy': 'joy',
        #     'sad': 'sadness',
        #     'worried': 'fear',  # Map worried to fear (closest match)
        #     'surprise': 'surprise',
        #     # Add reverse mapping for any edge cases
        #     'anger': 'anger',
        #     'joy': 'joy', 
        #     'sadness': 'sadness',
        #     'fear': 'fear',
        #     'disgust': 'disgust'
        # }

        # # Keep the same JSON files for reasoning tasks (same as first_face.py)
        # json_file_path = "/home/user/selected_face/face_emotion/MERR_coarse_grained.json" 
        # with open(json_file_path, 'r') as json_file:
        #     self.MERR_coarse_grained_dict = json.load(json_file)

        # reason_json_file_path = "/home/user/selected_face/face_emotion/MERR_fine_grained.json"
        # with open(reason_json_file_path, 'r') as json_file:
        #     self.MERR_fine_grained_dict = json.load(json_file)

        # Create video name mapping for MELD format
        self.create_video_name_mapping()

    def create_video_name_mapping(self):
        """
        Create mapping from MELD CSV format to video file names
        Assumes video files are named as: dia{Dialogue_ID}_utt{Utterance_ID}
        """
        self.video_name_mapping = {}
        for idx, row in self.meld_data.iterrows():
            dialogue_id = row['Dialogue_ID']  
            utterance_id = row['Utterance_ID']
            video_name = f"dia{dialogue_id}_utt{utterance_id}"
            self.video_name_mapping[idx] = video_name

    def __len__(self):
        return len(self.meld_data)

    def __getitem__(self, index):
        # Get data from CSV
        row = self.meld_data.iloc[index]
        
        # Extract information from CSV
        utterance = row['Utterance']
        emotion_label = row['Emotion'].lower()  # Keep original MELD emotion
        dialogue_id = row['Dialogue_ID']
        utterance_id = row['Utterance_ID']
        
        # Generate video name
        video_name = f"dia{dialogue_id}_utt{utterance_id}"

        # Load pre-extracted features
        FaceMAE_feats, VideoMAE_feats, Audio_feats, EVA_feats = self.get(video_name)
        
        if len(VideoMAE_feats.shape) == 1:
            VideoMAE_feats = VideoMAE_feats.unsqueeze(0)
        if len(Audio_feats.shape) == 1:
            Audio_feats = Audio_feats.unsqueeze(0)
        if len(FaceMAE_feats.shape) == 1:
            FaceMAE_feats = FaceMAE_feats.unsqueeze(0)
        
        video_features = torch.cat((FaceMAE_feats, VideoMAE_feats, Audio_feats), dim=0)

        # Random task selection
        task = random.choice(self.task_pool)
        
        if task == "emotion":
            # Keep original MELD emotion for now - mapping will happen in eval_emotion.py
            caption = emotion_label  # Original MELD emotion
            caption = self.text_processor(caption)
            instruction_pool = self.emotion_instruction_pool  # This still asks for MER emotions

        emotion = self.emo2idx.get(emotion_label, 0)  # Default to neutral if not found
        
        # Use transcript from CSV directly
        sentence = utterance
        character_line = f"The person in video says: {sentence}. "
        
        instruction = "<video><VideoHere></video> <feature><FeatureHere></feature> {} [{}] {} ".format(
            character_line, task, random.choice(instruction_pool))

        # return {
        #     "eva_features": EVA_feats,           # Pre-extracted EVA features [1025, 1408]
        #     "video_features": video_features,    # Other modalities [3, 1024]
        #     "instruction_input": instruction,
        #     "answer": caption,
        #     "emotion": emotion,
        #     "image_id": video_name,
        #     # "speaker": speaker,                  # Additional MELD info
        #     # "sentiment": sentiment,              # Additional MELD info
        #     # "dialogue_id": dialogue_id,          # Additional MELD info
        #     # "utterance_id": utterance_id         # Additional MELD info
        # }

        return {
            "eva_features": EVA_feats,           
            "video_features": video_features,    
            "instruction_input": instruction,
            "answer": caption,  # Original MELD emotion (will be ignored)
            "emotion": emotion, # Original MELD emotion index
            "original_emotion": emotion_label,  # Original MELD emotion string
            "image_id": video_name,
        }

    def get(self, video_name):
        # FaceMAE feature
        FaceMAE_feats_path = os.path.join(self.feat_path, 'test_mae_feat', video_name + '.npy')
        FaceMAE_feats = torch.tensor(np.load(FaceMAE_feats_path))

        # VideoMAE feature
        VideoMAE_feats_path = os.path.join(self.feat_path, 'test_videomae_feat', video_name + '.npy')
        VideoMAE_feats = torch.tensor(np.load(VideoMAE_feats_path))

        # Audio feature
        Audio_feats_path = os.path.join(self.feat_path, 'test_audio_feat', video_name + '.npy')
        Audio_feats = torch.tensor(np.load(Audio_feats_path))

        # EVA-ViT feature (pre-extracted)
        EVA_feats_path = os.path.join(self.feat_path, 'test_global_feat', video_name + '.npy')
        EVA_feats = torch.tensor(np.load(EVA_feats_path))

        return FaceMAE_feats, VideoMAE_feats, Audio_feats, EVA_feats