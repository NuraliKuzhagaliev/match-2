"""Supplementary contrast view; the full scene is always retained for the model."""
from PIL import Image


def text_views(image):
    framed = frame_text(image)
    return [image] if framed is image else [image, framed]


def frame_text(image):
    width, height = image.size
    # Wide plates already fill their useful frame. Keep banners available for the VLM.
    if not .55 <= width / height <= 1.8 or min(width, height) < 100:
        return image
    gray = image.convert("L").tobytes()
    boxes = []
    minimum_area = max(8, width * height // 100000)
    for light in (False, True):
        mask = bytearray((value >= 220 if light else value <= 64) for value in gray)
        for start in range(len(mask)):
            if not mask[start]:
                continue
            mask[start] = 0
            pending = [start]
            count = 0
            x0 = x1 = start % width
            y0 = y1 = start // width
            while pending:
                index = pending.pop()
                x, y = index % width, index // width
                count += 1
                x0, x1 = min(x0, x), max(x1, x)
                y0, y1 = min(y0, y), max(y1, y)
                for neighbor in ((index-1 if x else -1), (index+1 if x+1 < width else -1),
                                 (index-width if y else -1), (index+width if y+1 < height else -1)):
                    if neighbor >= 0 and mask[neighbor]:
                        mask[neighbor] = 0
                        pending.append(neighbor)
            bw, bh = x1-x0+1, y1-y0+1
            # Remove sign outlines/backgrounds and sensor specks, retaining glyphs.
            if count >= minimum_area and 2 <= bw <= width*.6 and max(6, height*.015) <= bh <= height*.5 and count/(bw*bh) >= .1:
                boxes.append((x0,y0,x1+1,y1+1))
    if len(boxes) < 2:
        return image
    margin = max(8, round(min(width, height)*.025))
    bounds = (max(0,min(b[0] for b in boxes)-margin), max(0,min(b[1] for b in boxes)-margin),
              min(width,max(b[2] for b in boxes)+margin), min(height,max(b[3] for b in boxes)+margin))
    # Only trim meaningful empty margins. Never erase most of the image on weak evidence.
    area = (bounds[2]-bounds[0]) * (bounds[3]-bounds[1])
    if not width*height*.02 <= area <= width*height*.85:
        return image
    return image.crop(bounds)
