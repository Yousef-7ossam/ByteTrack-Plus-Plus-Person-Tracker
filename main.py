import cv2
import numpy as np
import onnxruntime as ort
from collections import deque


MODEL_PATH = "./weights/v8_n_fp32.onnx"
IMG_SIZE = 320

CONF_THRES = 0.5
NMS_THRES = 0.45
PERSON_CLASS = 0

TARGET_ID = None
CLICK_POINT = None
TARGET_LAST_CENTER = None   
TARGET_HIST = None          

ROBOT_FOLLOW_MODE = False          
ROBOT_DEADZONE_X = 0.15            
ROBOT_DEADZONE_DEPTH = 0.10        

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

cv2.namedWindow("OUTPUT", cv2.WINDOW_NORMAL)
cv2.resizeWindow("OUTPUT", 1280, 720)

def mouse(event, x, y, flags, param):
    global CLICK_POINT
    if event == cv2.EVENT_LBUTTONDOWN:
        CLICK_POINT = (x, y)

cv2.setMouseCallback("OUTPUT", mouse)

def preprocess(img):
    h, w = img.shape[:2]
    scale = IMG_SIZE / max(h, w)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((IMG_SIZE, IMG_SIZE, 3), 114, dtype=np.uint8)
    pad_x = (IMG_SIZE - nw) // 2
    pad_y = (IMG_SIZE - nh) // 2
    canvas[pad_y:pad_y+nh, pad_x:pad_x+nw] = resized
    blob = canvas[:, :, ::-1].transpose(2, 0, 1)
    blob = np.ascontiguousarray(blob, np.float32) / 255.0
    return blob[None], scale, pad_x, pad_y

def nms(boxes, scores, iou_thres):
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        iou = (w * h) / (areas[i] + areas[order[1:]] - w * h + 1e-6)
        order = order[1:][iou < iou_thres]
    return keep

def compute_histogram(frame, bbox):
    """
    HSV color histogram of the bounding box region.
    Used for re-identification after occlusion.
    """
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

    if x2 <= x1 or y2 <= y1:
        return None

    roi = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [18, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist.flatten()

def hist_similarity(h1, h2):
    if h1 is None or h2 is None:
        return 0.0
    return float(np.dot(h1, h2) / (np.linalg.norm(h1) * np.linalg.norm(h2) + 1e-6))
class KalmanBox:
    """
    Simple constant-velocity Kalman filter for [cx, cy, w, h].
    State: [cx, cy, vx, vy, w, h]
    """
    def __init__(self, bbox):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w  = x2 - x1
        h  = y2 - y1

        self.kf = cv2.KalmanFilter(6, 4)
        self.kf.transitionMatrix = np.array([
            [1,0,1,0,0,0],
            [0,1,0,1,0,0],
            [0,0,1,0,0,0],
            [0,0,0,1,0,0],
            [0,0,0,0,1,0],
            [0,0,0,0,0,1],
        ], dtype=np.float32)

        self.kf.measurementMatrix = np.array([
            [1,0,0,0,0,0],
            [0,1,0,0,0,0],
            [0,0,0,0,1,0],
            [0,0,0,0,0,1],
        ], dtype=np.float32)

        self.kf.processNoiseCov     = np.eye(6, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
        self.kf.errorCovPost        = np.eye(6, dtype=np.float32)

        self.kf.statePost = np.array([[cx],[cy],[0],[0],[w],[h]], dtype=np.float32)

    def predict(self):
        s = self.kf.predict()
        cx, cy, _, _, w, h = s.flatten()
        return cx - w/2, cy - h/2, cx + w/2, cy + h/2

    def update(self, bbox):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w  = x2 - x1
        h  = y2 - y1
        m  = np.array([[cx],[cy],[w],[h]], dtype=np.float32)
        self.kf.correct(m)
class ByteTracker:
    """
    Two-stage association (high + low confidence),
    Kalman prediction for spatial matching,
    appearance histogram for re-ID after occlusion.
    """
    def __init__(self):
        self.tracks      = {}    
        self.lost_tracks = {}    
        self.next_id     = 0

    @staticmethod
    def iou_matrix(bboxes_a, bboxes_b):
        if len(bboxes_a) == 0 or len(bboxes_b) == 0:
            return np.zeros((len(bboxes_a), len(bboxes_b)))
        ax1,ay1,ax2,ay2 = np.array(bboxes_a).T
        bx1,by1,bx2,by2 = np.array(bboxes_b).T
        xx1 = np.maximum(ax1[:,None], bx1[None,:])
        yy1 = np.maximum(ay1[:,None], by1[None,:])
        xx2 = np.minimum(ax2[:,None], bx2[None,:])
        yy2 = np.minimum(ay2[:,None], by2[None,:])
        inter = np.maximum(0,xx2-xx1)*np.maximum(0,yy2-yy1)
        aa = (ax2-ax1)*(ay2-ay1)
        ab = (bx2-bx1)*(by2-by1)
        return inter / (aa[:,None] + ab[None,:] - inter + 1e-6)

    @staticmethod
    def _greedy_match(cost_matrix, threshold):
        """Returns list of (det_idx, track_idx) pairs."""
        pairs = []
        if cost_matrix.size == 0:
            return pairs
        used_d, used_t = set(), set()
        flat = np.argsort(-cost_matrix.ravel())
        for idx in flat:
            d, t = divmod(int(idx), cost_matrix.shape[1])
            if d in used_d or t in used_t:
                continue
            if cost_matrix[d, t] < threshold:
                break
            pairs.append((d, t))
            used_d.add(d)
            used_t.add(t)
        return pairs

    def update(self, dets, frame):
        """
        dets: Nx5 array [x1,y1,x2,y2,score]
        frame: BGR image (for appearance histograms)
        """
        high_dets = dets[dets[:,4] >= CONF_THRES]
        low_dets  = dets[(dets[:,4] >= 0.1) & (dets[:,4] < CONF_THRES)]

        track_ids   = list(self.tracks.keys())
        track_list  = [self.tracks[t] for t in track_ids]

        predicted_boxes = []
        for t in track_list:
            pb = t["kalman"].predict()
            predicted_boxes.append(pb)

        matched_pairs_1 = []
        unmatched_dets_1 = list(range(len(high_dets)))
        unmatched_tracks_1 = list(range(len(track_list)))

        if len(high_dets) and len(predicted_boxes):
            iou = self.iou_matrix(high_dets[:,:4].tolist(), predicted_boxes)
            matched_pairs_1 = self._greedy_match(iou, 0.3)
            md = {p[0] for p in matched_pairs_1}
            mt = {p[1] for p in matched_pairs_1}
            unmatched_dets_1   = [i for i in range(len(high_dets))  if i not in md]
            unmatched_tracks_1 = [i for i in range(len(track_list)) if i not in mt]

        matched_pairs_2 = []
        unmatched_dets_2 = list(range(len(low_dets)))

        if len(low_dets) and len(unmatched_tracks_1):
            sub_pred = [predicted_boxes[i] for i in unmatched_tracks_1]
            iou2 = self.iou_matrix(low_dets[:,:4].tolist(), sub_pred)
            raw2 = self._greedy_match(iou2, 0.2)
            for (di, ti_sub) in raw2:
                matched_pairs_2.append((di, unmatched_tracks_1[ti_sub]))
            md2 = {p[0] for p in raw2}
            unmatched_dets_2 = [i for i in range(len(low_dets)) if i not in md2]

        new_tracks = {}
        for (di, ti) in matched_pairs_1:
            tid = track_ids[ti]
            det = high_dets[di]
            t   = track_list[ti]
            t["kalman"].update(det[:4])
            t["bbox"]    = tuple(det[:4])
            cx = (det[0]+det[2])/2; cy = (det[1]+det[3])/2
            t["center"]  = (cx, cy)
            t["history"].append((cx, cy))
            t["height_history"].append(det[3]-det[1])
            t["hist"]    = compute_histogram(frame, det[:4])
            t["lost"]    = 0
            new_tracks[tid] = t

        for (di, ti) in matched_pairs_2:
            tid = track_ids[ti]
            det = low_dets[di]
            t   = track_list[ti]
            t["kalman"].update(det[:4])
            t["bbox"]    = tuple(det[:4])
            cx = (det[0]+det[2])/2; cy = (det[1]+det[3])/2
            t["center"]  = (cx, cy)
            t["history"].append((cx, cy))
            t["height_history"].append(det[3]-det[1])
            t["hist"]    = compute_histogram(frame, det[:4])
            t["lost"]    = 0
            new_tracks[tid] = t

        all_matched_track_indices = {p[1] for p in matched_pairs_1} | {p[1] for p in matched_pairs_2}
        for ti in range(len(track_list)):
            if ti not in all_matched_track_indices:
                tid = track_ids[ti]
                t   = track_list[ti]
                t["lost"] += 1
                if t["lost"] < 30:
                    new_tracks[tid] = t
                    self.lost_tracks[tid] = t

        remaining_dets = [high_dets[i] for i in unmatched_dets_1]
        truly_new = []

        for det in remaining_dets:
            h = compute_histogram(frame, det[:4])
            best_sim = 0.60     
            best_tid = None

            for ltid, lt in list(self.lost_tracks.items()):
                sim = hist_similarity(h, lt.get("hist"))
                if sim > best_sim:
                    best_sim = sim
                    best_tid = ltid

            if best_tid is not None:
                t = self.lost_tracks.pop(best_tid)
                t["kalman"].update(det[:4])
                t["bbox"]   = tuple(det[:4])
                cx = (det[0]+det[2])/2; cy = (det[1]+det[3])/2
                t["center"] = (cx, cy)
                t["history"].append((cx, cy))
                t["height_history"].append(det[3]-det[1])
                t["hist"]   = h
                t["lost"]   = 0
                new_tracks[best_tid] = t
                print(f"[RE-ID] Recovered ID {best_tid} (sim={best_sim:.2f})")
            else:
                truly_new.append(det)

        for det in truly_new:
            tid = self.next_id
            self.next_id += 1
            cx = (det[0]+det[2])/2; cy = (det[1]+det[3])/2
            new_tracks[tid] = {
                "bbox":           tuple(det[:4]),
                "center":         (cx, cy),
                "history":        deque([(cx, cy)], maxlen=20),
                "height_history": deque([det[3]-det[1]], maxlen=20),
                "hist":           compute_histogram(frame, det[:4]),
                "kalman":         KalmanBox(det[:4]),
                "lost":           0,
            }

        self.tracks = new_tracks
        self.lost_tracks = {k:v for k,v in self.lost_tracks.items() if v["lost"] < 60}

        return self.tracks

tracker = ByteTracker()

def postprocess(out, scale, pad_x, pad_y):
    pred = out[0]
    if pred.ndim == 3:
        pred = pred[0]
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T

    boxes     = pred[:, :4]
    cls_scores = pred[:, 4:]
    cls_ids   = np.argmax(cls_scores, axis=1)
    scores    = np.max(cls_scores, axis=1)

    mask = scores > 0.1
    boxes    = boxes[mask]
    scores   = scores[mask]
    cls_ids  = cls_ids[mask]

    mask = cls_ids == PERSON_CLASS
    boxes  = boxes[mask]
    scores = scores[mask]

    if len(boxes) == 0:
        return np.empty((0, 5))

    cx, cy, bw, bh = boxes.T
    x1 = (cx - bw/2 - pad_x) / scale
    y1 = (cy - bh/2 - pad_y) / scale
    x2 = (cx + bw/2 - pad_x) / scale
    y2 = (cy + bh/2 - pad_y) / scale

    boxes = np.stack([x1, y1, x2, y2], axis=1)
    keep  = nms(boxes, scores, NMS_THRES)
    return np.hstack([boxes[keep], scores[keep].reshape(-1, 1)])

def estimate_motion(history, height_history):
    """
    Uses bounding box height as depth proxy:
      taller bbox → closer (FORWARD movement of person toward camera)
      shorter bbox → farther (BACKWARD)
    Also uses XY for LEFT/RIGHT.
    """
    direction = "STABLE"
    pred = (0, 0)

    if len(history) >= 2:
        pts = np.array(history)
        dx  = np.mean(pts[-3:, 0]) - np.mean(pts[:3, 0]) if len(pts) >= 6 else pts[-1,0]-pts[0,0]
        dy  = np.mean(pts[-3:, 1]) - np.mean(pts[:3, 1]) if len(pts) >= 6 else pts[-1,1]-pts[0,1]
        last_x, last_y = history[-1]
        pred = (int(last_x + dx*2), int(last_y + dy*2))

    if len(height_history) >= 6:
        heights = np.array(height_history)
        dh = np.mean(heights[-3:]) - np.mean(heights[:3])   

        xy_mag = 0.0
        if len(history) >= 6:
            pts = np.array(history)
            dx  = np.mean(pts[-3:, 0]) - np.mean(pts[:3, 0])
            dy  = np.mean(pts[-3:, 1]) - np.mean(pts[:3, 1])
            xy_mag = np.sqrt(dx**2 + dy**2)
        else:
            dx = dy = 0.0

        depth_sig  = abs(dh) > 3        
        lateral_sig = xy_mag > 5

        if not depth_sig and not lateral_sig:
            direction = "STABLE"
        elif depth_sig and abs(dh) >= xy_mag * 0.5:
            direction = "APPROACHING" if dh > 0 else "RECEDING"
        elif lateral_sig:
            if abs(dy) > abs(dx):
                direction = "DOWN" if dy > 0 else "UP"
            else:
                direction = "RIGHT" if dx > 0 else "LEFT"
    elif len(history) >= 4:
        pts = np.array(history)
        dx  = pts[-1,0] - pts[0,0]
        dy  = pts[-1,1] - pts[0,1]
        mag = np.sqrt(dx**2+dy**2)
        if mag > 5:
            if abs(dy) > abs(dx):
                direction = "DOWN" if dy > 0 else "UP"
            else:
                direction = "RIGHT" if dx > 0 else "LEFT"

    return direction, pred

def compute_robot_command(frame, track):
    """
    Returns steering string based on target position in frame.
    Logic:
      X-axis: left/right steering
      Bbox height ratio: forward/backward (depth)
    """
    if track is None:
        return "SEARCH"

    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = track["bbox"]
    cx = ((x1 + x2) / 2) / fw     
    bbox_h_ratio = (y2 - y1) / fh  

    TARGET_DEPTH = 0.45            

    off_x     = cx - 0.5          
    off_depth = bbox_h_ratio - TARGET_DEPTH  

    cmds = []

    if abs(off_x) > ROBOT_DEADZONE_X:
        cmds.append("TURN_RIGHT" if off_x > 0 else "TURN_LEFT")

    if abs(off_depth) > ROBOT_DEADZONE_DEPTH:
        cmds.append("STOP" if off_depth > 0 else "ADVANCE")  

    return " + ".join(cmds) if cmds else "HOLD"

def draw_hud(frame, tracks, target_id, robot_mode):
    fh, fw = frame.shape[:2]

    if robot_mode:
        cv2.rectangle(frame, (0, fh-36), (fw, fh), (0,0,0), -1)
        t = tracks.get(target_id)
        cmd = compute_robot_command(frame, t)
        cv2.putText(frame, f"ROBOT FOLLOW: {cmd}", (10, fh-10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,255,255), 2)

    cv2.putText(frame, f"Tracks: {len(tracks)}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    cv2.putText(frame, "Q=quit  R=robot_follow  click=lock_target",
                (10, fh-46 if robot_mode else fh-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)

cap = cv2.VideoCapture(0)
print("[INFO] ByteTrack++ running — R: robot follow | click: lock target | Q: quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    blob, scale, px, py = preprocess(frame)
    out  = session.run(None, {input_name: blob})
    dets = postprocess(out, scale, px, py)

    tracks = tracker.update(dets, frame)

    if CLICK_POINT is not None:
        cx, cy = CLICK_POINT
        for tid, t in tracks.items():
            x1, y1, x2, y2 = t["bbox"]
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                TARGET_ID = tid
                TARGET_LAST_CENTER = t["center"]
                TARGET_HIST = t.get("hist")
                print(f"[INFO] TARGET LOCKED: {TARGET_ID}")
                break
        CLICK_POINT = None

    if TARGET_ID is not None and TARGET_ID in tracks:
        TARGET_LAST_CENTER = tracks[TARGET_ID]["center"]
        TARGET_HIST = tracks[TARGET_ID].get("hist")

    elif TARGET_ID is not None and TARGET_ID not in tracks and len(tracks) > 0:
        best_tid = None
        best_score = -1.0

        for tid, t in tracks.items():
            score = 0.0

            if TARGET_HIST is not None:
                sim = hist_similarity(TARGET_HIST, t.get("hist"))
                score += sim * 0.7  

            if TARGET_LAST_CENTER is not None:
                cx_t, cy_t = t["center"]
                cx_l, cy_l = TARGET_LAST_CENTER
                dist = np.sqrt((cx_t - cx_l)**2 + (cy_t - cy_l)**2)
                proximity_score = max(0.0, 1.0 - dist / 200.0)
                score += proximity_score * 0.3

            if score > best_score:
                best_score = score
                best_tid = tid

        if best_tid is not None and best_score > 0.45:
            print(f"[TARGET-RECOVERY] ID تغير من {TARGET_ID} → {best_tid} (score={best_score:.2f})")
            TARGET_ID = best_tid

    for tid, t in tracks.items():
        x1, y1, x2, y2 = map(int, t["bbox"])
        direction, pred = estimate_motion(t["history"], t["height_history"])

        is_target = (tid == TARGET_ID)
        color = (0, 0, 255) if is_target else (0, 220, 80)
        label = f"TARGET | {direction}" if is_target else f"ID {tid} | {direction}"

        thickness = 3 if is_target else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1-th-8), (x1+tw+4, y1), color, -1)
        cv2.putText(frame, label, (x1+2, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

        px2, py2 = pred
        cv2.circle(frame, (px2, py2), 5, (255, 80, 0), -1)

        hist = list(t["history"])
        for i in range(1, len(hist)):
            alpha = i / len(hist)
            tc = tuple(int(c * alpha) for c in color)
            cv2.line(frame,
                     (int(hist[i-1][0]), int(hist[i-1][1])),
                     (int(hist[i][0]),   int(hist[i][1])),
                     tc, 1)

    draw_hud(frame, tracks, TARGET_ID, ROBOT_FOLLOW_MODE)
    cv2.imshow("OUTPUT", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        ROBOT_FOLLOW_MODE = not ROBOT_FOLLOW_MODE
        print(f"[INFO] Robot follow mode: {'ON' if ROBOT_FOLLOW_MODE else 'OFF'}")

cap.release()
cv2.destroyAllWindows()