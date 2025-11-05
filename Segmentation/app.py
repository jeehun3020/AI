import gradio as gr
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

MODEL_ID = "nvidia/segformer-b5-finetuned-ade-640-640"
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModelForSemanticSegmentation.from_pretrained(MODEL_ID)

def ade_palette():
    """ADE20K palette that maps each class to RGB values."""
    return [
        [204, 87, 92],[112, 185, 212],[45, 189, 106],[234, 123, 67],[78, 56, 123],[210, 32, 89],
        [90, 180, 56],[155, 102, 200],[33, 147, 176],[255, 183, 76],[67, 123, 89],[190, 60, 45],
        [134, 112, 200],[56, 45, 189],[200, 56, 123],[87, 92, 204],[120, 56, 123],[45, 78, 123],
        [156, 200, 56],[32, 90, 210],[56, 123, 67],[180, 56, 123],[123, 67, 45],[45, 134, 200],
        [67, 56, 123],[78, 123, 67],[32, 210, 90],[45, 56, 189],[123, 56, 123],[56, 156, 200],
        [189, 56, 45],[112, 200, 56],[56, 123, 45],[200, 32, 90],[123, 45, 78],[200, 156, 56],
        [45, 67, 123],[56, 45, 78],[45, 56, 123],[123, 67, 56],[56, 78, 123],[210, 90, 32],
        [123, 56, 189],[45, 200, 134],[67, 123, 56],[123, 45, 67],[90, 32, 210],[200, 45, 78],
        [32, 210, 90],[45, 123, 67],[165, 42, 87],[72, 145, 167],[15, 158, 75],[209, 89, 40],
        [32, 21, 121],[184, 20, 100],[56, 135, 15],[128, 92, 176],[1, 119, 140],[220, 151, 43],
        [41, 97, 72],[148, 38, 27],[107, 86, 176],[21, 26, 136],[174, 27, 90],[91, 96, 204],
        [108, 50, 107],[27, 45, 136],[168, 200, 52],[7, 102, 27],[42, 93, 56],[140, 52, 112],
        [92, 107, 168],[17, 118, 176],[59, 50, 174],[206, 40, 143],[44, 19, 142],[23, 168, 75],
        [54, 57, 189],[144, 21, 15],[15, 176, 35],[107, 19, 79],[204, 52, 114],[48, 173, 83],
        [11, 120, 53],[206, 104, 28],[20, 31, 153],[27, 21, 93],[11, 206, 138],[112, 30, 83],
        [68, 91, 152],[153, 13, 43],[25, 114, 54],[92, 27, 150],[108, 42, 59],[194, 77, 5],
        [145, 48, 83],[7, 113, 19],[25, 92, 113],[60, 168, 79],[78, 33, 120],[89, 176, 205],
        [27, 200, 94],[210, 67, 23],[123, 89, 189],[225, 56, 112],[75, 156, 45],[172, 104, 200],
        [15, 170, 197],[240, 133, 65],[89, 156, 112],[214, 88, 57],[156, 134, 200],[78, 57, 189],
        [200, 78, 123],[106, 120, 210],[145, 56, 112],[89, 120, 189],[185, 206, 56],[47, 99, 28],
        [112, 189, 78],[200, 112, 89],[89, 145, 112],[78, 106, 189],[112, 78, 189],[156, 112, 78],
        [28, 210, 99],[78, 89, 189],[189, 78, 57],[112, 200, 78],[189, 47, 78],[205, 112, 57],
        [78, 145, 57],[200, 78, 112],[99, 89, 145],[200, 156, 78],[57, 78, 145],[78, 57, 99],
        [57, 78, 145],[145, 112, 78],[78, 89, 145],[210, 99, 28],[145, 78, 189],[57, 200, 136],
        [89, 156, 78],[145, 78, 99],[99, 28, 210],[189, 78, 47],[28, 210, 99],[78, 145, 57],
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

if __name__ == "__main__":
    demo.launch()
