"""MP4 or GIF export via imageio; GIF works without ffmpeg."""
from pathlib import Path
import imageio.v2 as imageio


def save_video(frames, path, fps=30):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == '.gif':
        imageio.mimsave(path, frames, duration=1.0 / fps)
    else:
        with imageio.get_writer(path, fps=fps, codec='libx264', macro_block_size=16) as writer:
            for frame in frames:
                writer.append_data(frame)
    return path
