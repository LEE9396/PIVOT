#!/usr/bin/env python3
"""Create base/support/head masks for one desk-lamp RGB frame with SAM3."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# 박스를 그릴 때 쓰는 색. **마스크 미리보기(preview.png)와 같은 값**이어야
# 한다. 예전에는 base 가 여기서는 주황, 미리보기에서는 파랑이라 사람이 두
# 화면을 대조할 수 없었다.
BOX_COLORS = {"base": "#1464ff", "support": "#28d228", "head": "#f0281e",
              "moving_link": "#f0281e"}


def _legend_path(explicit):
    """부위 이름표 그림을 찾는다. --legend > 환경변수 > 없음."""
    import os

    for candidate in (explicit, os.environ.get("PIVOT_PART_LEGEND")):
        if candidate:
            path = Path(candidate).expanduser()
            if path.is_file():
                return path
            print(f"[마스크] 이름표 그림이 없다: {path}")
    return None


def select_boxes(image, names, legend=None):
    """사람이 부위마다 박스를 그린다. 옆에 **부위 이름표 그림**을 띄운다.

    이름표가 없으면 화면에 "1/3 base" 라는 글자만 나오고, 어느 덩어리가
    base 인지 알려 주는 것이 아무것도 없다. 여기서 이름을 틀리면 예외도
    경고도 없이 FoundationPose 가 엉뚱한 부위를 추적하고, PIVOT 은 그
    각도를 그대로 믿는다 (my_work/NAMING.md 의 사고와 같은 구조).

    이름표는 tools/make_part_legend.py 가 배달물 메시에서 굽는다.
    """
    import tkinter as tk
    from PIL import Image, ImageTk

    scale = min(1040 / image.width, 585 / image.height, 1.0)
    shown = image.resize((round(image.width * scale), round(image.height * scale)))
    root = tk.Tk()
    root.title("SAM 램프 박스 선택 — 오른쪽 이름표를 보고 그리세요")
    photo = ImageTk.PhotoImage(shown)

    row = tk.Frame(root)
    row.pack()
    canvas = tk.Canvas(row, width=shown.width, height=shown.height,
                       cursor="cross")
    canvas.create_image(0, 0, image=photo, anchor="nw")
    canvas.pack(side="left")
    legend_photo = None
    if legend is not None:
        legend_image = Image.open(legend).convert("RGB")
        wide = min(560 / legend_image.width, shown.height / legend_image.height, 1.0)
        legend_image = legend_image.resize(
            (round(legend_image.width * wide), round(legend_image.height * wide)))
        legend_photo = ImageTk.PhotoImage(legend_image)
        tk.Label(row, image=legend_photo, borderwidth=2,
                 relief="groove").pack(side="left", padx=(8, 0))
    else:
        tk.Label(row, width=40, justify="left", wraplength=300,
                 fg="#a00000", font=("Sans", 11),
                 text="부위 이름표 그림이 없습니다.\n"
                      "tools/make_part_legend.py 로 구운 뒤\n"
                      "PIVOT_PART_LEGEND 로 알려 주세요.\n\n"
                      "이름을 틀리면 조용히 틀린 각도가 나옵니다."
                 ).pack(side="left", padx=(8, 0))

    label = tk.Label(root, font=("Sans", 16, "bold"))
    label.pack(pady=4)
    boxes, current, start = [], None, None
    colors = tuple(BOX_COLORS.get(n, "#808080") for n in names)

    def update_label():
        name = names[len(boxes)]
        label.config(text=f"{len(boxes) + 1}/{len(names)}  {name}  "
                          "— 이 색으로 칠해진 부위를 왼쪽에서 드래그",
                     fg=colors[len(boxes)])

    def press(event):
        nonlocal start, current
        start = (event.x, event.y)
        if current is not None:
            canvas.delete(current[0])
        item = canvas.create_rectangle(event.x, event.y, event.x, event.y,
                                       outline=colors[len(boxes)], width=3)
        current = (item, event.x, event.y, event.x, event.y)

    def drag(event):
        nonlocal current
        if start is None:
            return
        x = np.clip(event.x, 0, shown.width - 1)
        y = np.clip(event.y, 0, shown.height - 1)
        canvas.coords(current[0], start[0], start[1], x, y)
        current = (current[0], start[0], start[1], x, y)

    def confirm():
        nonlocal current, start
        if current is None:
            return
        _, x0, y0, x1, y1 = current
        x0, x1 = sorted((x0 / scale, x1 / scale))
        y0, y1 = sorted((y0 / scale, y1 / scale))
        if x1 - x0 < 5 or y1 - y0 < 5:
            return
        boxes.append((x0, y0, x1, y1))
        current = start = None
        if len(boxes) == len(names):
            root.destroy()
        else:
            update_label()

    canvas.bind("<ButtonPress-1>", press)
    canvas.bind("<B1-Motion>", drag)
    canvas.bind("<ButtonRelease-1>", drag)
    tk.Button(root, text="확정", command=confirm, width=18).pack(pady=(0, 6))
    update_label()
    root.mainloop()
    if len(boxes) != len(names):
        raise RuntimeError("박스 선택이 완료되지 않았습니다.")
    return dict(zip(names, boxes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sam3-root", type=Path, required=True)
    parser.add_argument("--manual", action="store_true")
    parser.add_argument("--legend", default=None,
                        help="부위 이름표 그림 (tools/make_part_legend.py 산출). "
                             "생략하면 환경변수 PIVOT_PART_LEGEND 를 본다")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    import torch
    from PIL import Image

    sys.path.insert(0, str(args.sam3_root.resolve()))
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    image = Image.open(args.image).convert("RGB")
    boxes = (select_boxes(image, ("base", "support", "head"),
                          _legend_path(args.legend))
             if args.manual else None)
    processor = Sam3Processor(build_sam3_image_model(), confidence_threshold=0.05)

    def best(*prompts):
        for prompt in prompts:
            state = processor.set_text_prompt(prompt, processor.set_image(image))
            scores = state["scores"].float().cpu().numpy()
            if len(scores):
                index = int(np.argmax(scores))
                return (state["masks"][index, 0].cpu().numpy().astype(bool),
                        float(scores[index]), prompt)
        raise RuntimeError(f"SAM3 found no mask for {prompts!r}")

    def from_box(name):
        x0, y0, x1, y1 = boxes[name]
        box = [(x0 + x1) / (2 * image.width),
               (y0 + y1) / (2 * image.height),
               (x1 - x0) / image.width, (y1 - y0) / image.height]
        state = processor.add_geometric_prompt(
            box, True, processor.set_image(image))
        scores = state["scores"].float().cpu().numpy()
        if not len(scores):
            raise RuntimeError(f"SAM3 found no mask in the {name} box")
        index = int(np.argmax(scores))
        mask = state["masks"][index, 0].cpu().numpy().astype(bool)
        region = np.zeros(mask.shape, dtype=bool)
        region[round(y0):round(y1), round(x0):round(x1)] = True
        return mask & region, float(scores[index]), "manual box"

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        if boxes is None:
            base, base_score, base_prompt = best("lamp base", "desk lamp base")
            head, head_score, head_prompt = best(
                "lamp light bar", "lamp shade", "lamp head")
            whole, whole_score, whole_prompt = best("desk lamp", "lamp")
            base &= whole
            head &= whole & ~base
            support = whole & ~(base | head)
        else:
            base, base_score, base_prompt = from_box("base")
            support, whole_score, whole_prompt = from_box("support")
            head, head_score, head_prompt = from_box("head")
            support &= ~base
            head &= ~(base | support)
    masks = {"base": base, "support": support, "head": head}
    pixels = {name: int(mask.sum()) for name, mask in masks.items()}
    min_pixels = 500 if boxes is not None else 1000
    if (any(count < min_pixels for count in pixels.values())
            or (boxes is None and whole.mean() > 0.3)):
        raise RuntimeError(
            f"suspicious lamp masks: {pixels}\n"
            "  SAM3 의 글자 프롬프트가 부위를 못 갈랐다 (0 픽셀인 부위가 있으면\n"
            "  그 부위는 아예 안 잡힌 것이다). 사람이 박스를 그리는 쪽으로 가라:\n"
            "    make_desk_lamp_masks.py ... --manual --legend <이름표.png>\n"
            "  런처에서는 setup/experiment.conf 의 MANUAL_MASK=1 을 켜면 된다.")

    view = np.asarray(image).copy()
    colors = {"base": (20, 100, 255), "support": (40, 210, 40),
              "head": (240, 40, 30)}
    for name, mask in masks.items():
        Image.fromarray(mask.astype(np.uint8) * 255).save(args.output / f"{name}.png")
        view[mask] = (0.35 * view[mask]
                      + 0.65 * np.asarray(colors[name])).astype(np.uint8)
    Image.fromarray(view).save(args.output / "preview.png")
    summary = {
        "method": ("SAM3 manual box masks" if boxes is not None else
                   "SAM3 text masks: support = whole lamp - base - head"),
        "prompts": {"base": base_prompt, "head": head_prompt,
                    "whole": whole_prompt},
        "scores": {"base": base_score, "head": head_score,
                   "whole": whole_score},
        "pixels": pixels,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
