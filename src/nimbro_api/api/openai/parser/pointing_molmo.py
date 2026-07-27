# Some VLMs are capable of object grounding by pointing or 2D/3D bounding boxes.
# This completion parser extracts grounding content from the text-completion of Molmo-style models and copies it to the grounding-completion.
# This way its possible to conveniently use the model as an open vocabulary detector.
# Molmo 1 coordinates use [0, 100] and Molmo 2 coordinates use [0, 1000].
# Both are normalized to [0, 1] with inclusive bounds.
# Here, each grounded object has the form {'x': float, 'y': float, 'label': str, 'type': "point_2d_normalized"}

import re
import math

def _normalized(value, scale):
    """Convert a Molmo coordinate to the inclusive normalized range [0, 1]."""
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(value) or not 0 <= value <= scale:
        return None
    return value / scale

def extract_points(text):
    """Extract Molmo points as normalized coordinates in inclusive range [0, 1].

    Both endpoints are valid and map to the first and final image pixels.
    Malformed and out-of-range points are ignored.
    """
    pattern = re.compile(
        r'<point\s+([^>]*)>([^<]+)</point>'
        r'|<points\s+([^>]*)>([^<]+)</points>',
        re.I | re.S
    )

    def clean_label(label):
        return label.replace("_", " ").replace(".", "").strip()

    results = []

    for m in pattern.finditer(text):
        # Molmo 1 single-point format: <point ...>label</point>
        if m.group(1):
            attrs, label = m.group(1), clean_label(m.group(2))

            x = re.search(r'\bx\s*=\s*"([^"]*)"', attrs)
            y = re.search(r'\by\s*=\s*"([^"]*)"', attrs)

            if x and y:
                xv, yv = _normalized(x.group(1), 100), _normalized(y.group(1), 100)
                if xv is not None and yv is not None:
                    results.append({
                        'x': xv,
                        'y': yv,
                        'label': label,
                        'type': 'point_2d_normalized'
                    })
        else:
            # <points ...>label</points>
            attrs, label = m.group(3), clean_label(m.group(4))

            # Molmo 1 multi-point format (x1=..., y1=...)
            xs = {}
            for i, v in re.findall(r'\bx(\d+)\s*=\s*"([^"]*)"', attrs):
                value = _normalized(v, 100)
                if value is not None:
                    xs[int(i)] = value
            ys = {}
            for i, v in re.findall(r'\by(\d+)\s*=\s*"([^"]*)"', attrs):
                value = _normalized(v, 100)
                if value is not None:
                    ys[int(i)] = value

            if xs and ys:
                for i in sorted(set(xs) & set(ys)):
                    results.append({
                        'x': xs[i],
                        'y': ys[i],
                        'label': label,
                        'type': 'point_2d_normalized'
                    })
                continue

            # Molmo 2 format: coords="i x y i x y ..."
            coords_match = re.search(r'coords\s*=\s*"([^"]+)"', attrs)
            if not coords_match:
                continue

            try:
                vals = list(map(float, coords_match.group(1).split()))
            except Exception:
                continue

            if len(vals) < 2:
                continue

            # detect optional leading frame index
            offset = 1 if len(vals) % 3 == 1 else 0

            parsed_any = False
            for i in range(offset, len(vals) - 2, 3):
                _, xv, yv = vals[i:i + 3]

                xv, yv = _normalized(xv, 1000), _normalized(yv, 1000)
                if xv is not None and yv is not None:
                    results.append({
                        'x': xv,
                        'y': yv,
                        'label': label,
                        'type': 'point_2d_normalized'
                    })
                    parsed_any = True

            # fallback: only if nothing parsed
            if not parsed_any and len(vals) >= 2:
                xv, yv = vals[-2], vals[-1]
                xv, yv = _normalized(xv, 1000), _normalized(yv, 1000)
                if xv is not None and yv is not None:
                    results.append({
                        'x': xv,
                        'y': yv,
                        'label': label,
                        'type': 'point_2d_normalized'
                    })

    return results

def parse(self, success, message, completion):
    if 'text' not in completion:
        completion['logs'].append("Cannot extract grounding without text-completion.")
        return success, message, completion

    if not isinstance(completion['text'], str):
        completion['logs'].append(f"Cannot extract grounding from text-completion of type '{type(completion['text']).__name__}' instead of 'str'.")
        return success, message, completion

    grounding_content = extract_points(completion['text'])

    if not grounding_content:
        log = "There is no grounding content in the text-completion."
    elif 'grounding' in completion:
        if isinstance(completion['grounding'], list):
            completion['grounding'].extend(grounding_content)
            log = "Extracted grounding from text-completion and appended to existing grounding-completion."
        else:
            log = f"Extracted grounding from text-completion but cannot append it to existing grounding-completion of type '{type(completion['grounding']).__name__}'."
    else:
        completion['grounding'] = grounding_content
        log = "Extracted grounding from text-completion and set as grounding-completion."

    completion['logs'].append(log)
    self._logger.info(log)

    return success, message, completion
