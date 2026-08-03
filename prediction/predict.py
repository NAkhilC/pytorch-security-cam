import cv2
from PIL import Image
import torch

from config import CONFIDENCE_THRESHOLD
from prediction.distanceCalculator import calculateDistance
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms.functional import to_tensor

print(cv2.__version__)
print(cv2.__file__)
print(hasattr(cv2, "CascadeClassifier"))


def predict_face(face_bgr, transform, model, device, full_dataset):
    """Takes a cropped BGR face image (numpy array) and returns (name, confidence)."""
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(face_rgb)

    tensor = transform(face_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item()

    name = full_dataset.index_to_name.get(pred_idx, f"class_{pred_idx}")
    if confidence < CONFIDENCE_THRESHOLD:
        name = "unknown"
    return name, confidence


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def model_prediction(transform, model, device, full_dataset):
    # 0 = default device camera; try 1, 2... for external cams
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open camera. Check permissions or try a different index (0, 1, 2...).")
        return

    print("Camera open. Press 'q' to quit, 's' to save the current frame.")

    frame_count = 0
    saved_count = 0

    # Cache of the most recent prediction per tracked face, keyed by a rounded
    # position so the label stays attached to the same face across frames
    # instead of flickering blank on frames where we skip re-classifying.
    last_predictions = {}   # {face_key: (name, confidence)}
    PREDICT_EVERY_N_FRAMES = 5   # re-run the CNN this often; detection runs every frame

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame_count += 1

        # Run face DETECTION every frame — this is cheap and keeps the
        # rectangle glued to the face's actual current position instead of
        # lagging behind or disappearing on skipped frames.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5)

        current_keys = set()

        for (x, y, w, h) in faces:
            # Round position to a coarse grid so the same physical face keeps
            # the same cache key even as detection jitters by a few pixels.
            face_key = (x // 20, y // 20)
            current_keys.add(face_key)

            # Only run the (more expensive) CNN prediction periodically, or
            # the first time we see this face. Otherwise reuse the cached
            # label so it doesn't flicker away between refreshes.
            if frame_count % PREDICT_EVERY_N_FRAMES == 0 or face_key not in last_predictions:
                face_crop = frame[y:y + h, x:x + w]
                if face_crop.size > 0:
                    name, confidence = predict_face(
                        face_crop, transform, model, device, full_dataset)

                    last_predictions[face_key] = (name, confidence)

            name, confidence = last_predictions.get(face_key, ("...", 0.0))

            # Draw on every single frame, at this frame's actual detected
            # box — this is what keeps the rectangle and label glued to the
            # face as it moves, instead of only appearing every Nth frame.
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{name} ({confidence:.0%})"
            # Call the distance calculation function
            label += f" : {calculateDistance(12, 31, w):.2f} ft"
            cv2.putText(
                frame, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

        # Drop cached predictions for faces that are no longer in view, so
        # stale labels don't linger if the face leaves and a new one appears
        # near the same spot later.
        for stale_key in list(last_predictions.keys()):
            if stale_key not in current_keys:
                del last_predictions[stale_key]

        cv2.imshow("Face recognition — press q to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            saved_count += 1
            filename = f"captured_{saved_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")

    cap.release()
    cv2.destroyAllWindows()
