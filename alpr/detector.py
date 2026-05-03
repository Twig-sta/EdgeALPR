import cv2
from alpr.utils import preprocess_image


def _clip_bbox(x, y, w, h, frame_width, frame_height, pad=0):
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(frame_width, x + w + pad)
    y2 = min(frame_height, y + h + pad)
    return x1, y1, x2 - x1, y2 - y1


def _expand_plate_bbox(x, y, w, h, frame_width, frame_height):
    aspect_ratio = w / float(h)

    if aspect_ratio < 3.2:
        target_width = int(h * 4.2)
        extra_width = max(0, target_width - w)
    else:
        extra_width = int(w * 0.35)

    extra_height = int(h * 0.28)

    x -= extra_width // 2
    y -= extra_height // 2
    w += extra_width
    h += extra_height

    return _clip_bbox(x, y, w, h, frame_width, frame_height)


def _scaled_kernel(frame_width, width, height):
    scale = max(1, round(frame_width / 1200))
    return cv2.getStructuringElement(cv2.MORPH_RECT, (width * scale, height * scale))


def _bbox_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0


def _plate_score(x, y, w, h, frame_width, frame_height, fill_ratio=0):
    aspect_ratio = w / float(h)
    aspect_score = max(0, 1 - abs(aspect_ratio - 3.0) / 3.0)
    center_x = x + w / 2
    center_y = y + h / 2
    center_score = 1 - min(1, abs(center_x - frame_width / 2) / (frame_width / 2))
    lower_half_score = min(1, max(0, (center_y - frame_height * 0.25) / (frame_height * 0.5)))
    return aspect_score * 2.0 + center_score + lower_half_score + fill_ratio


def _add_candidate(candidates, frame, bbox, score, min_width=50, min_height=18, expand=True):
    frame_height, frame_width = frame.shape[:2]
    x, y, w, h = bbox

    if w < min_width or h < min_height:
        return

    aspect_ratio = w / float(h)
    area = w * h
    frame_area = frame_width * frame_height

    if not (1.5 <= aspect_ratio <= 6.5):
        return
    max_area_ratio = 0.35 if not expand else 0.15
    if not (frame_area * 0.001 <= area <= frame_area * max_area_ratio):
        return

    clipped = _expand_plate_bbox(x, y, w, h, frame_width, frame_height) if expand else _clip_bbox(
        x,
        y,
        w,
        h,
        frame_width,
        frame_height
    )

    cx, cy, cw, ch = clipped
    if cx == 0 or cx + cw == frame_width:
        score -= 3.0

    for candidate in candidates:
        if _bbox_iou(candidate["bbox"], clipped) > 0.45:
            if score > candidate["score"]:
                candidate.update({
                    "bbox": clipped,
                    "image": frame[cy:cy + ch, cx:cx + cw],
                    "score": score,
                })
            return

    candidates.append({
        "bbox": clipped,
        "image": frame[cy:cy + ch, cx:cx + cw],
        "score": score,
    })


def _add_dark_text_candidates(candidates, frame):
    frame_height, frame_width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dark_mask = cv2.inRange(gray, 0, 70)
    dark_mask[:int(frame_height * 0.40), :] = 0

    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    character_boxes = []

    min_char_width = max(18, int(frame_width * 0.012))
    max_char_width = int(frame_width * 0.13)
    min_char_height = int(frame_height * 0.09)
    max_char_height = int(frame_height * 0.28)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        area = w * h

        if x <= frame_width * 0.04 or x + w >= frame_width * 0.96:
            continue
        if not (min_char_width <= w <= max_char_width):
            continue
        if not (min_char_height <= h <= max_char_height):
            continue
        if not (0.12 <= aspect_ratio <= 1.05):
            continue
        if area < frame_width * frame_height * 0.001:
            continue

        character_boxes.append((x, y, w, h))

    best_group = []
    best_width = 0

    for seed in character_boxes:
        sx, sy, sw, sh = seed
        seed_center_y = sy + sh / 2
        group = []

        for box in character_boxes:
            x, y, w, h = box
            center_y = y + h / 2
            if abs(center_y - seed_center_y) <= frame_height * 0.08 and abs(h - sh) <= frame_height * 0.08:
                group.append(box)

        if len(group) < 4:
            continue

        x1 = min(x for x, y, w, h in group)
        x2 = max(x + w for x, y, w, h in group)
        width = x2 - x1

        if width > best_width:
            best_group = group
            best_width = width

    if not best_group:
        return

    x1 = min(x for x, y, w, h in best_group)
    y1 = min(y for x, y, w, h in best_group)
    x2 = max(x + w for x, y, w, h in best_group)
    y2 = max(y + h for x, y, w, h in best_group)

    text_width = x2 - x1
    text_height = y2 - y1
    text_aspect_ratio = text_width / float(text_height)

    if text_width < frame_width * 0.22 or not (2.4 <= text_aspect_ratio <= 8.0):
        return

    plate_x = int(x1 - text_width * 0.10)
    plate_y = int(y1 - text_height * 0.45)
    plate_w = int(text_width * 1.22)
    plate_h = int(text_height * 1.65)

    score = 10.0 + len(best_group) + text_width / float(frame_width)
    _add_candidate(candidates, frame, (plate_x, plate_y, plate_w, plate_h), score, expand=False)


def detect_plates(frame):
    if frame is None:
        return []

    frame_height, frame_width = frame.shape[:2]
    processed = preprocess_image(frame)

    candidates = []

    _add_dark_text_candidates(candidates, frame)

    edges = cv2.Canny(processed, 60, 180)
    edge_kernel = _scaled_kernel(frame_width, 17, 5)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, edge_kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        score = _plate_score(x, y, w, h, frame_width, frame_height)
        _add_candidate(candidates, frame, (x, y, w, h), score)

    bright_mask = cv2.inRange(processed, 160, 255)
    bright_mask[:int(frame_height * 0.30), :] = 0
    bright_kernel = _scaled_kernel(frame_width, 11, 5)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, bright_kernel, iterations=2)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, (3, 3), iterations=1)

    contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        fill_ratio = cv2.contourArea(contour) / area if area else 0
        score = _plate_score(x, y, w, h, frame_width, frame_height, fill_ratio)
        _add_candidate(candidates, frame, (x, y, w, h), score)

    candidates.sort(key=lambda item: item["score"], reverse=True)
    for candidate in candidates:
        candidate.pop("score", None)

    return candidates[:5]
