from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
from itertools import chain
import time
from voice import create_voice_request


def xyz(pose_landmark):
    return [pose_landmark.x, pose_landmark.y, pose_landmark.z]


app = Flask(__name__)
import torch
from torch import nn
import torch.nn.functional as F

INPUT_DIM = 102  # X is 102-dimensional
HIDDEN_DIM = INPUT_DIM * 4  # Center-most latent space vector will have length of 408
NUM_CLASSES = 16  # 16 classes

print(INPUT_DIM)
print(HIDDEN_DIM)
print(NUM_CLASSES)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Right arm
    (0, 5), (5, 6), (6, 7), (7, 8),  # Left arm
    (9, 10), (10, 11),  # Right hip
    (9, 12), (12, 13)  # Left hip
    # Add more connections as needed
]


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim * 2)
        self.fc2 = nn.Linear(input_dim * 2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, int(hidden_dim / 2))
        self.fc4 = nn.Linear(int(hidden_dim / 2), int(hidden_dim / 4))
        self.fc5 = nn.Linear(int(hidden_dim / 4), int(hidden_dim / 8))
        self.fc6 = nn.Linear(int(hidden_dim / 8), num_classes)

    def forward(self, x_in):
        z = F.relu(self.fc1(x_in))  # ReLU activation function added!
        z = F.relu(self.fc2(z))
        z = F.relu(self.fc3(z))
        z = F.relu(self.fc4(z))
        z = F.relu(self.fc5(z))
        z = self.fc6(z)
        return z


model = MLP(INPUT_DIM, HIDDEN_DIM, NUM_CLASSES)
checkpoint = torch.load('model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

why = [0.3473447263240814, 0.7478870749473572, -0.6172481179237366, 0.36646026372909546, 0.6813147664070129,
       -0.6316997408866882, 0.38035881519317627, 0.6779729723930359, -0.6322327852249146, 0.3958359360694885,
       0.6724678874015808, -0.6325153708457947, 0.3371714651584625, 0.6808474659919739, -0.5851145386695862,
       0.3299342691898346, 0.6782799959182739, -0.5846661329269409, 0.32315289974212646, 0.6747258901596069,
       -0.5848789215087891, 0.45136478543281555, 0.6622810363769531, -0.5387877821922302, 0.351970374584198,
       0.6685916781425476, -0.2359001487493515, 0.38095521926879883, 0.8133420944213867, -0.5276967287063599,
       0.34834024310112, 0.8131296634674072, -0.47063809633255005, 0.6113532781600952, 0.9090954065322876,
       -0.43060100078582764, 0.310513973236084, 0.8833157420158386, 0.022346632555127144, 0.6917772889137268,
       1.342121958732605, -0.5441672205924988, 0.18342217803001404, 1.0702632665634155, 0.03161010146141052,
       0.6779292821884155, 1.5454636812210083, -0.6467449069023132, 0.09763718396425247, 1.0143853425979614,
       -0.27303966879844666, 0.6883513331413269, 1.6344853639602661, -0.6830201745033264, 0.058266136795282364,
       0.9985447525978088, -0.30261674523353577, 0.6651507616043091, 1.6163396835327148, -0.6954372525215149,
       0.06747840344905853, 1.0011236667633057, -0.33236852288246155, 0.6541649699211121, 1.5832273960113525,
       -0.6536346077919006, 0.09075508266687393, 1.0170166492462158, -0.3049856126308441, 0.5707029104232788,
       1.6867097616195679, -0.1708478331565857, 0.36904722452163696, 1.6520156860351562, 0.17407648265361786,
       0.5336729288101196, 2.2692887783050537, -0.21383264660835266, 0.3312743604183197, 2.2338168621063232,
       0.25058475136756897, 0.5210361480712891, 2.823061227798462, 0.18108680844306946, 0.3288745880126953,
       2.7822484970092773, 0.46153202652931213, 0.5371248722076416, 2.9145054817199707, 0.1998334378004074,
       0.33349063992500305, 2.861675977706909, 0.47271081805229187, 0.46283090114593506, 2.9732205867767334,
       -0.24820591509342194, 0.3257504999637604, 2.9631714820861816, 0.06842140853404999, 1, 0, 0]

# Initialize MediaPipe Pose module
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize OpenCV camera
cap = cv2.VideoCapture(1)
frame_counter = 0
SAMPLE_RATE = 1000000000


def process_frame(frame):
    # Convert frame to RGB (MediaPipe requires RGB input)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame using MediaPipe Pose
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        hi = []
        landmark_list = []
        for i, landmark in enumerate(results.pose_landmarks.landmark):
            x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
            landmark_list.append((x, y))
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # Draw lines to connect landmarks
        connections = mp_pose.POSE_CONNECTIONS
        for connection in connections:
            x1, y1 = landmark_list[connection[0]]
            x2, y2 = landmark_list[connection[1]]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for i, landmark in enumerate(results.pose_landmarks.landmark):
            if i % SAMPLE_RATE == 0:
                landmarks = []  # Store landmarks for processing
                row = list(chain.from_iterable([xyz(landmark) for landmark in results.pose_landmarks.landmark]))
                hi.append(row)

                # model_response = call_to_model()
                # create_voice_request(model_response)

                landmarks.append([landmark.x, landmark.y, landmark.z])
                print(landmarks, "landmarks")
                print(row, "ROW!!!")

    return frame


def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        time.sleep(1)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True, port=3005)
