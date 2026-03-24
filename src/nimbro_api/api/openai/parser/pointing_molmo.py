# Some VLMs are capable of object grounding by pointing or 2D/3D bounding boxes.
# This completion parser extracts grounding content from the text-completion of Molmo-style models and copies it to the grounding-completion.
# This way its possible to conveniently use the model as an open vocabulary detector.
# Here, each grounded object has the form {'x': float, 'y': float, 'label': str, 'type': "point_2d_normalized"}

import re

def extract_points(text):
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

            x = re.search(r'\bx\s*=\s*"([\d.]+)"', attrs)
            y = re.search(r'\by\s*=\s*"([\d.]+)"', attrs)

            if x and y:
                xv, yv = float(x.group(1)), float(y.group(1))
                if 0 <= xv <= 100 and 0 <= yv <= 100:
                    results.append({
                        'x': xv / 100.,
                        'y': yv / 100.,
                        'label': label,
                        'type': 'point_2d_normalized'
                    })
        else:
            # <points ...>label</points>
            attrs, label = m.group(3), clean_label(m.group(4))

            # Molmo 1 multi-point format (x1=..., y1=...)
            xs = {int(i): float(v) for i, v in re.findall(r'\bx(\d+)\s*=\s*"([\d.]+)"', attrs)}
            ys = {int(i): float(v) for i, v in re.findall(r'\by(\d+)\s*=\s*"([\d.]+)"', attrs)}

            if xs and ys:
                for i in sorted(set(xs) & set(ys)):
                    xv, yv = xs[i], ys[i]
                    if 0 <= xv <= 100 and 0 <= yv <= 100:
                        results.append({
                            'x': xv / 100.,
                            'y': yv / 100.,
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

                if 0 <= xv <= 1000 and 0 <= yv <= 1000:
                    results.append({
                        'x': xv / 1000.,
                        'y': yv / 1000.,
                        'label': label,
                        'type': 'point_2d_normalized'
                    })
                    parsed_any = True

            # fallback: only if nothing parsed
            if not parsed_any and len(vals) >= 2:
                xv, yv = vals[-2], vals[-1]
                if 0 <= xv <= 1000 and 0 <= yv <= 1000:
                    results.append({
                        'x': xv / 1000.,
                        'y': yv / 1000.,
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
