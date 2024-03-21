import cv2
import numpy as np
import onnx
import onnxruntime as ort
import vision.utils.box_utils_numpy as box_utils
import time

onnx_path = "models/onnx/version-RFB-320.onnx"
label_path = "models/voc-model-labels.txt"
class_names = [name.strip() for name in open(label_path).readlines()]
ort_session = ort.InferenceSession(onnx_path)
input_name = ort_session.get_inputs()[0].name
threshold = 0.7

# Initialize the FPS counter
fps = 0
frame_counter = 0
start_time = time.time()

def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = box_utils.hard_nms(box_probs,
                                       iou_threshold=iou_threshold,
                                       top_k=top_k,
                                       )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]


rtsp_url = "rtsp://admin:sanslab1@192.168.1.64:554/Streaming/Channels/101"
cap = cv2.VideoCapture(rtsp_url)

# cap = cv2.VideoCapture(1)  # 0 is the index of the webcam. Change if you have multiple webcams.


while True:
    ret, orig_image = cap.read()
    if not ret:
        break

    # Preprocess the image
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (320, 240))
    image_mean = np.array([127, 127, 127])
    image = (image - image_mean) / 128
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)

    # Run the model
    time_time = time.time()
    confidences, boxes = ort_session.run(None, {input_name: image})
    print("cost time:{}".format(time.time() - time_time))

    # Postprocess the results
    boxes, labels, probs = predict(orig_image.shape[1], orig_image.shape[0], confidences, boxes, threshold)
    for i in range(boxes.shape[0]):
        box = boxes[i, :]
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"

        cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)

    # Calculate FPS
    frame_counter += 1
    elapsed_time = time.time() - start_time
    if elapsed_time >= 1:  # Every second
        fps = frame_counter / elapsed_time
        start_time = time.time()  # Reset time
        frame_counter = 0  # Reset counter

    # Display FPS on the frame
    cv2.putText(orig_image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the image
    cv2.imshow('Live Face Detection', orig_image)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()