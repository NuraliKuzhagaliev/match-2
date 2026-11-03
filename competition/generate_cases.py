"""Ten self-generated scene-text fixtures. Never included in the inference image."""
import argparse
import hashlib
import json
from pathlib import Path
import random
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


def find_font(candidates):
    return next((Path(p) for p in candidates if Path(p).is_file()), None)


def generate(output, latin_font=None, cjk_font=None):
    latin_font = latin_font or find_font(["C:/Windows/Fonts/arialbd.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"])
    cjk_font = cjk_font or find_font(["C:/Windows/Fonts/msyh.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"])
    if not latin_font or not cjk_font:
        raise ValueError("Provide --latin-font and --cjk-font; missing CJK glyphs must not silently replace test text")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    manifest = []
    def font(size, cjk=False):
        return ImageFont.truetype(str(cjk_font if cjk else latin_font), size)
    def center(image, text, y, size, fill, cjk=False):
        draw = ImageDraw.Draw(image)
        bbox = draw.textbbox((0, 0), text, font=font(size, cjk))
        draw.text(((image.width - bbox[2]) / 2, y), text, font=font(size, cjk), fill=fill)
    def save(name, image, expected, kind, conditions, fmt="PNG"):
        extension = {"PNG": "png", "JPEG": "jpg", "TIFF": "tiff"}[fmt]
        path = output / f"{name}.{extension}"
        image.save(path, format=fmt)
        manifest.append({"file": path.name, "expected": expected, "kind": kind, "conditions": conditions,
                         "origin": "self-generated; synthetic, not real-world performance evidence",
                         "license": "CC0-1.0 image content; font files not redistributed",
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    def plate(registration, banner, slogan):
        image = Image.new("RGB", (820, 350), "#eeeedd")
        ImageDraw.Draw(image).rounded_rectangle((12, 12, 807, 337), radius=25, outline="#161b21", width=8)
        center(image, banner, 32, 35, "#24548a")
        center(image, registration, 112, 110, "#18283c")
        center(image, slogan, 278, 25, "#24548a")
        return image
    save("us_clean", plate("8MQZ204", "OREGON", "PACIFIC WONDERLAND"), "8MQZ204", "us_plate", ["clean"])
    angled = plate("7LPN684", "TEXAS", "THE LONE STAR STATE").transform(
        (820, 350), Image.Transform.PERSPECTIVE, (1.03, .09, -25, -.025, 1.08, 12, .00016, .00008), Image.Resampling.BICUBIC)
    save("us_angled", angled, "7LPN684", "us_plate", ["off_axis"], "JPEG")
    for name, registration, dim in [("chinese_clean", "粤C·60983", False), ("chinese_glare", "浙D·60609", True)]:
        image = Image.new("RGB", (820, 250), "#155198")
        ImageDraw.Draw(image).rounded_rectangle((10, 10, 810, 240), radius=15, outline="white", width=5)
        center(image, registration, 62, 108, "white", True)
        if dim:
            image = ImageEnhance.Brightness(image).enhance(.42)
            glare = Image.new("RGBA", image.size, (0, 0, 0, 0))
            ImageDraw.Draw(glare).ellipse((430, 35, 760, 190), fill=(255, 255, 225, 70))
            image = Image.alpha_composite(image.convert("RGBA"), glare).convert("RGB")
        save(name, image, registration, "chinese_plate", ["low_light", "glare"] if dim else ["clean"], "JPEG" if dim else "PNG")
    stop = Image.new("RGB", (600, 600), "#dfe5db")
    ImageDraw.Draw(stop).regular_polygon((300, 300, 250), n_sides=8, rotation=22.5, fill="#bc221d", outline="white", width=8)
    center(stop, "STOP", 225, 110, "white")
    save("stop_clean", stop, "STOP", "stop_sign", ["clean"])
    noisy = stop.copy()
    draw = ImageDraw.Draw(noisy)
    for _ in range(26000):
        gray = rng.randrange(256)
        draw.point((rng.randrange(600), rng.randrange(600)), fill=(gray, gray, gray))
    save("stop_noise", noisy, "STOP", "stop_sign", ["sensor_noise"], "TIFF")
    speed = Image.new("RGB", (650, 850), "white")
    ImageDraw.Draw(speed).rounded_rectangle((20, 20, 630, 830), radius=35, outline="black", width=12)
    center(speed, "SPEED", 90, 110, "black")
    center(speed, "LIMIT", 240, 110, "black")
    center(speed, "45", 405, 250, "black")
    save("speed", speed, "SPEED LIMIT 45", "speed_sign", ["multiline"], "JPEG")
    warning = Image.new("RGB", (800, 800), "#dce3d6")
    ImageDraw.Draw(warning).polygon([(400, 20), (780, 400), (400, 780), (20, 400)], fill="#eea742", outline="#25271f", width=8)
    for y, text in [(230, "LANE"), (335, "CLOSED"), (440, "AHEAD")]:
        center(warning, text, y, 78, "#22271f")
    save("warning", warning, "LANE CLOSED AHEAD", "warning_sign", ["multiline"])
    advisory = Image.new("RGB", (520, 520), "#e4b74e")
    ImageDraw.Draw(advisory).rounded_rectangle((20, 20, 500, 500), radius=30, outline="black", width=10)
    center(advisory, "25", 120, 215, "black")
    save("advisory", advisory, "25", "advisory_sign", ["digits_only"], "TIFF")
    blur = plate("4VNK708", "WASHINGTON", "EVERGREEN STATE").filter(ImageFilter.GaussianBlur(2.2))
    save("us_blur", blur, "4VNK708", "us_plate", ["blur"])
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/competition"))
    parser.add_argument("--latin-font", type=Path)
    parser.add_argument("--cjk-font", type=Path)
    args = parser.parse_args()
    print(f"Generated {len(generate(args.output, args.latin_font, args.cjk_font))} synthetic cases")


if __name__ == "__main__":
    main()
