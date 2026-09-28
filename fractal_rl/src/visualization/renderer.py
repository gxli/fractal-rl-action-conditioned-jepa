"""Local camera view and trajectory overlay in image frames."""
import numpy as np
from PIL import Image, ImageDraw
from src.terrain.periodic_field import sample


def frame(env, trail, canvas_size=640, view_world=96.0):
    n = canvas_size
    coordinate = (np.arange(n) - (n - 1) / 2) * view_world / n
    xx, yy = np.meshgrid(coordinate, -coordinate)
    pos = env.body.position
    image_field = sample(env.terrain.potential, pos[0] + xx, pos[1] + yy, env.terrain.pixel_size)
    minimum = 0.0
    maximum = max(float(env.terrain.potential.max()), 1e-8)
    normalized = np.clip((image_field - minimum) / max(maximum - minimum, 1e-8), 0, 1)
    image = Image.fromarray(np.uint8(255 * (1 - normalized)), mode='L').convert('RGB')
    draw = ImageDraw.Draw(image)
    scale = n / view_world
    points = [(n / 2 + (p[0] - pos[0]) * scale, n / 2 - (p[1] - pos[1]) * scale) for p in trail]
    if len(points) > 1:
        draw.line(points, fill=(40, 110, 240), width=2)
    cx = cy = n / 2
    fov = env.cfg['agent']['fov_size'] * env.cfg['agent']['fov_pixel_size'] * scale / 2
    draw.rectangle((cx - fov, cy - fov, cx + fov, cy + fov), outline=(255, 210, 30), width=2)
    draw.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), fill=(240, 50, 50))
    direction = env.body.velocity / max(np.linalg.norm(env.body.velocity), 1e-8)
    draw.line((cx, cy, cx + direction[0] * 28, cy - direction[1] * 28), fill=(255, 40, 40), width=3)
    draw.text((10, 10), f'Speed: {np.linalg.norm(env.body.velocity):.2f}  Mean: {env.distance / max(env.steps * env.cfg["physics"]["dt"], 1e-8):.2f}', fill=(255, 40, 40), stroke_width=1, stroke_fill='white')
    return np.asarray(image)
