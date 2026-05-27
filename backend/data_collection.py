import cv2
import mediapipe as mp
import pandas as pd
import os

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

def extract_features(image_path):
    image = cv2.imread(image_path)
    if image is None: return None
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks: return None
    
    pts = results.multi_face_landmarks[0].landmark
    def get_dist(p1, p2):
        return ((pts[p1].x - pts[p2].x)**2 + (pts[p1].y - pts[p2].y)**2)**0.5

    fw = get_dist(234, 454) # Face Width
    if fw == 0: return None

    return [
        get_dist(61, 291) / fw,   # smile_w
        get_dist(13, 14) / fw,    # smile_h
        abs(pts[52].y - pts[33].y) / fw, # brow_h
        abs(pts[159].y - pts[145].y) / fw, # eye_h
        get_dist(55, 285) / fw,   # brow_i
        get_dist(61, 164) / fw,   # mouth_drop
        get_dist(107, 159) / fw,  # brow_v_left
        get_dist(336, 386) / fw,  # brow_v_right
        get_dist(13, 14) / fw,    # lip_dist
        get_dist(10, 152) / fw    # jaw_dist
    ]

def process_folder(base_path):
    data = []
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    for label in emotions:
        folder_path = os.path.join(base_path, label)
        print(f"\n--- Starting Emotion: {label.upper()} ---")
        if not os.path.exists(folder_path): continue
        
        for i, img_name in enumerate(os.listdir(folder_path)):
            features = extract_features(os.path.join(folder_path, img_name))
            if features:
                data.append([label] + features)
            
            if i % 100 == 0: # Print progress every 100 images
                print(f"Processed {i} images in {label}...", end='\r')
    return data

# MAIN EXECUTION
columns = ['label', 'smile_w', 'smile_h', 'brow_h', 'eye_h', 'brow_i', 
           'mouth_drop', 'brow_v_left', 'brow_v_right', 'lip_dist', 'jaw_dist']

# Update these paths to your actual image folders!
print("Starting Data Collection for Train Set...")
train_data = process_folder("train") 
pd.DataFrame(train_data, columns=columns).to_csv("train_features.csv", index=False)

print("\nStarting Data Collection for Test Set...")
test_data = process_folder("test")
pd.DataFrame(test_data, columns=columns).to_csv("test_features.csv", index=False)

print("\nSUCCESS: 10-Feature CSVs created!")