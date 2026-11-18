"""Interactive demo: upload a street-scene image, get its segmentation.

    pip install -e ".[demo]"
    python app.py --checkpoint outputs/unet_r34/checkpoints/best.ckpt
"""

from __future__ import annotations

import argparse

import gradio as gr
import numpy as np
from PIL import Image

from cityseg.inference import default_device, load_model, overlay, segment_image
from cityseg.labels import CLASSES, colorize


def build_demo(checkpoint: str) -> gr.Blocks:
    model = load_model(checkpoint, default_device())

    def run(image: Image.Image, flip: bool):
        if image is None:
            return None, None, ""
        train_ids = segment_image(model, image, flip=flip)
        color = colorize(train_ids)
        present = np.unique(train_ids)
        share = [(CLASSES[i].name, (train_ids == i).mean()) for i in present if i < len(CLASSES)]
        share.sort(key=lambda item: -item[1])
        summary = "\n".join(f"- **{name}**: {100 * part:.1f}% of pixels" for name, part in share)
        return overlay(image, color), Image.fromarray(color), summary

    with gr.Blocks(title="Urban scene segmentation") as demo:
        gr.Markdown("# Urban scene segmentation\n19-class Cityscapes semantic segmentation.")
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Street scene")
                flip = gr.Checkbox(label="Flip test-time augmentation (slower, more accurate)")
                button = gr.Button("Segment", variant="primary")
            with gr.Column():
                blended = gr.Image(label="Overlay")
                mask = gr.Image(label="Segmentation")
                summary = gr.Markdown()
        button.click(run, inputs=[image, flip], outputs=[blended, mask, summary])
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--share", action="store_true", help="create a public Gradio link")
    args = parser.parse_args()
    build_demo(args.checkpoint).launch(share=args.share)
