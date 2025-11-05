import gradio as gr
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

MODEL_ID = "mattmdjaga/segformer_b2_clothes"
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModelForSemanticSegmentation.from_pretrained(MODEL_ID)

def clothes_palette():
    """Clothes segmentation palette (18 classes)."""
    return [
        [0, 0, 0],        # Background
        [128, 0, 0],      # Hat
        [255, 0, 0],      # Hair
        [0, 85, 0],       # Sunglasses
        [170, 0, 51],     # Upper-clothes
        [255, 85, 0],     # Skirt
        [0, 0, 85],       # Pants
        [0, 119, 221],    # Dress
        [85, 85, 0],      # Belt
        [0, 85, 85],      # Left-shoe
        [85, 51, 0],      # Right-shoe
        [52, 86, 128],    # Face
        [0, 128, 0],      # Left-leg
        [0, 0, 128],      # Right-leg
        [128, 128, 0],    # Left-arm
        [128, 0, 128],    # Right-arm
        [0, 128, 128],    # Bag
        [128, 128, 128],  # Scarf
    ]

labels_list = []
with open("labels.txt", "r", encoding="utf-8") as fp:
    for line in fp:
        labels_list.append(line.rstrip("\n"))

colormap = np.asarray(ade_palette(), dtype=np.uint8)

def label_to_color_image(label):
    if label.ndim != 2:
        raise ValueError("Expect 2-D input label")
    if np.max(label) >= len(colormap):
        raise ValueError("label value too large.")
    return colormap[label]

def draw_plot(pred_img, seg_np):
    fig = plt.figure(figsize=(20, 15))
    grid_spec = gridspec.GridSpec(1, 2, width_ratios=[6, 1])

    plt.subplot(grid_spec[0])
    plt.imshow(pred_img)
    plt.axis('off')

    LABEL_NAMES = np.asarray(labels_list)
    FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
    FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

    unique_labels = np.unique(seg_np.astype("uint8"))
    ax = plt.subplot(grid_spec[1])
    plt.imshow(FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation="nearest")
    ax.yaxis.tick_right()
    plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
    plt.xticks([], [])
    ax.tick_params(width=0.0, labelsize=25)
    return fig

def run_inference(input_img):
    # input: numpy array from gradio -> PIL
    img = Image.fromarray(input_img.astype(np.uint8)) if isinstance(input_img, np.ndarray) else input_img
    if img.mode != "RGB":
        img = img.convert("RGB")

    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # (1, C, h/4, w/4)

    # resize to original
    upsampled = torch.nn.functional.interpolate(
        logits, size=img.size[::-1], mode="bilinear", align_corners=False
    )
    seg = upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)  # (H,W)

    # colorize & overlay
    color_seg = colormap[seg]                                # (H,W,3)
    pred_img = (np.array(img) * 0.5 + color_seg * 0.5).astype(np.uint8)

    fig = draw_plot(pred_img, seg)
    return fig

demo = gr.Interface(
    fn=run_inference,
    inputs=gr.Image(type="numpy", label="Input Image"),
    outputs=gr.Plot(label="Overlay + Legend"),
    examples=[
        "ADE_val_00000001.jpeg",
        "ADE_val_00001159.jpg",
        "ADE_val_00001248.jpg",
        "ADE_val_00001472.jpg"
    ],
    flagging_mode="never",
    cache_examples=False,
)

if __name__ == "__main__":4
    demo.launch()
