# Some VLMs are capable of object grounding by pointing or 2D/3D bounding boxes.
# This completion parser extracts grounding content from the text-completion of Qwen3-VL-style models and copies it to the grounding-completion.
# This way its possible to conveniently use the model as an open vocabulary detector.
# Qwen 2D coordinates use [0, 1000] and are normalized to [0, 1] with inclusive bounds.
# Each grounded object has one of the following forms:
# {'x': float, 'y': float, 'label': str, 'type': "point_2d_normalized"}
# {'x1': float, 'y1': float, 'x2': float, 'y2': float, 'label': str, 'type': "bbox_2d_normalized"}
# {'center_x': float, 'center_y': float, 'center_z': float, 'size_x': float, 'size_y': float, 'size_z': float, 'roll': float, 'pitch': float, 'yaw': float, 'label': str, 'type': "bbox_3d_in_camera"}
# Any additional key found for a grounded object is forwarded.

import re
import json
import math

def _normalized(value):
    """Convert one Qwen coordinate to the inclusive normalized range [0, 1]."""
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(value) or not 0 <= value <= 1000:
        return None
    return value / 1000.0

def _iter_json_arrays(text):
    """Yield JSON arrays found in fenced ```json blocks"""
    for m in re.finditer(r"```json\s*(.*?)\s*```", text, re.I | re.S):
        inner = m.group(1).strip()
        # yield if the fenced block itself is a JSON array
        try:
            arr = json.loads(inner)
            if isinstance(arr, list):
                yield arr
                continue
            if isinstance(arr, dict) and any(k in arr for k in ('point_2d', 'bbox_2d', 'bbox_3d')):
                yield [arr]
                continue
        except Exception:
            pass
        # attempt extraction if the fenced block contains a JSON array sub-string
        for n in re.finditer(r"\[\s*{.*?}\s*\]", inner, re.S):
            try:
                arr = json.loads(n.group(0))
                if isinstance(arr, list):
                    yield arr
            except Exception:
                pass

def extract_grounding(text):
    """Extract Qwen grounding, normalizing [0, 1000] coordinates to [0, 1].

    Normalized point endpoints are inclusive. Box minima are inclusive and
    maxima exclusive, and boxes must have positive width and height.
    """
    results = []

    # XML points2d
    pattern = re.compile(r'<points\s+([^>]*)>([^<]+)</points>', re.I | re.S)
    for attrs, label in pattern.findall(text):
        label = label.replace("_", " ").replace(".", "")
        xs = {}
        for i, v in re.findall(r'\bx(\d+)\s*=\s*"([^"]*)"', attrs):
            value = _normalized(v)
            if value is not None:
                xs[int(i)] = value
        ys = {}
        for i, v in re.findall(r'\by(\d+)\s*=\s*"([^"]*)"', attrs):
            value = _normalized(v)
            if value is not None:
                ys[int(i)] = value
        for i in sorted(xs.keys() & ys.keys()):
            results.append({
                'x': xs[i], 'y': ys[i],
                'label': label,
                'type': 'point_2d_normalized',
            })

    # JSON arrays, point2d, bbox2d or bbox3d
    for arr in _iter_json_arrays(text):
        for obj in arr:
            if not isinstance(obj, dict):
                continue
            # point_2d
            if 'point_2d' in obj and isinstance(obj['point_2d'], (list, tuple)) and len(obj['point_2d']) >= 2:
                x, y = (_normalized(value) for value in obj['point_2d'][:2])
                if x is not None and y is not None:
                    item = {
                        'x': x,
                        'y': y,
                        'label': obj.get('label', ''),
                        'type': 'point_2d_normalized',
                    }
                    # forward extra attributes
                    for k, v in obj.items():
                        if k not in {'point_2d', 'x', 'y', 'label', 'type'}:
                            item[k] = v
                    results.append(item)

            # bbox_2d
            if 'bbox_2d' in obj and isinstance(obj['bbox_2d'], (list, tuple)) and len(obj['bbox_2d']) >= 4:
                x1, y1, x2, y2 = (_normalized(value) for value in obj['bbox_2d'][:4])
                if None not in (x1, y1, x2, y2) and x1 < x2 and y1 < y2:
                    item = {
                        'x1': x1, 'y1': y1,
                        'x2': x2, 'y2': y2,
                        'label': obj.get('label', ''),
                        'type': 'bbox_2d_normalized',
                    }
                    for k, v in obj.items():
                        if k not in {'bbox_2d', 'x1', 'y1', 'x2', 'y2', 'label', 'type'}:
                            item[k] = v
                    results.append(item)

            # bbox_3d
            if 'bbox_3d' in obj and isinstance(obj['bbox_3d'], (list, tuple)) and len(obj['bbox_3d']) >= 9:
                try:
                    cx, cy, cz, sx, sy, sz, roll, pitch, yaw = map(float, obj['bbox_3d'][:9])
                except (TypeError, ValueError, OverflowError):
                    continue
                item = {
                    'center_x': cx, 'center_y': cy, 'center_z': cz,
                    'size_x': sx, 'size_y': sy, 'size_z': sz,
                    'roll': roll, 'pitch': pitch, 'yaw': yaw,
                    'label': obj.get('label', ''),
                    'type': 'bbox_3d_in_camera',
                }
                for k, v in obj.items():
                    if k not in item and k != 'bbox_3d':
                        item[k] = v
                results.append(item)

    return results

def parse(self, success, message, completion):
    if 'text' in completion:
        if isinstance(completion['text'], str):
            grounding_content = extract_grounding(completion['text'])
            if len(grounding_content) > 0:
                if 'grounding' in completion:
                    if isinstance(completion['grounding'], list):
                        completion['grounding'].extend(grounding_content)
                        completion['logs'].append("Extracted grounding from text-completion and appended to existing grounding-completion.")
                        self._logger.info(completion['logs'][-1])
                    else:
                        completion['logs'].append(f"Extracted grounding from text-completion but cannot append it to existing grounding-completion of type '{type(completion['grounding']).__name__}'.")
                        self._logger.info(completion['logs'][-1])
                else:
                    completion['grounding'] = grounding_content
                    completion['logs'].append("Extracted grounding from text-completion and set as grounding-completion.")
                    self._logger.info(completion['logs'][-1])
            else:
                completion['logs'].append("There is no grounding content in the text-completion.")
                self._logger.info(completion['logs'][-1])
        else:
            completion['logs'].append(f"Cannot extract grounding from text-completion of type '{type(completion['text']).__name__}' instead of 'str'.")
    else:
        completion['logs'].append("Cannot extract grounding without text-completion.")

    return success, message, completion

# [2026-03-18 20:39:57.177][INFO ][ChatCompletions]: Text-completion:
#                                                  | '
#                                                  | ```xml
#                                                  | <faces>
#                                                  |      <face x1="190" y1="363" x2="198" y2="146" x3="298" y3="99" x4="528" y4="138" x5="630" y5="111" x6="754" y6="113" x7="566" y7="361"
#                                                  | alt="faces">faces</face>
#                                                  | </faces>
#                                                  | ```
#                                                  | '
# [2026-03-18 20:39:57.182][INFO ][ChatCompletions]: There is no grounding content in the text-completion.

# [2026-03-19 13:38:42.566][INFO ][ChatCompletions]: Text-completion:
#                                                  | '
#                                                  | ```xml
#                                                  | <objects>
#                                                  |   <object>
#                                                  |     <label>person</label>
#                                                  |     <bbox>79,314,308,938">69,308,303,930</bbox>
#                                                  |   </object>
#                                                  |   <object>
#                                                  |     <label>person</label>
#                                                  |     <bbox>453,290,708,952">451,288,705,946</bbox>
#                                                  |   </object>
#                                                  |   <object>
#                                                  |     <label>person</label>
#                                                  |     <bbox>683,45,953,934">682,44,951,927</bbox>
#                                                  |   </object>
#                                                  |   <object>
#                                                  |     <label>person</label>
#                                                  |     <bbox>574,66,709,774">576,71,710,787</bbox>
#                                                  |   </object>
#                                                  |   <object>
#                                                  |     <label>person</label>
#                                                  |     <bbox>456,95,577,435">458,97,578,437</bbox>
#                                                  |   </object>
#                                                  |   <object>
#                                                  |     <label>person</label>
#                                                  |     <bbox>251,67,408,708">252,71,406,708</bbox>
#                                                  |   </object>
#                                                  |   <object>
#                                                  |     <label>person</label>
#                                                  |     <bbox>95,77,273,567">99,83,276,575</bbox>
#                                                  |   </object>
#                                                  | </objects>
#                                                  | ```
#                                                  | '
# [2026-03-19 13:38:42.570][INFO ][ChatCompletions]: There is no grounding content in the text-completion.
