import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops, checkpoint
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(model_config_path=None, model_checkpoint_path=None):
    args = SLConfig.fromfile(model_config_path or checkpoint.DEFAULT_CONFIG_PATH)
    args.device = device
    model = build_model(args)
    model_checkpoint_path = checkpoint.ensure_checkpoint(model_checkpoint_path)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    model = model.to(device)
    model.eval()
    return model

@torch.no_grad()
def predict(model, image, caption: str, text_threshold=0.5, box_threshold=0.5):
    outputs = model(image, captions=[caption])
    prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # prediction_logits.shape = (nq, 256)
    prediction_boxes = outputs["pred_boxes"].cpu()[0]  # prediction_boxes.shape = (nq, 4)

    mask = prediction_logits.max(dim=1)[0] > box_threshold
    logits = prediction_logits[mask]  # logits.shape = (n, 256)
    boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)
    text_match = logits > text_threshold
    logits = logits.max(dim=1)[0]

    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)

    phrases = np.array([
        get_phrases_from_posmap(b, tokenized, tokenizer).replace('.', '')
        for b in text_match
    ])

    return boxes, logits, phrases





@torch.no_grad()
def run(src, *classes, out_file=True, box_threshold=0.5, text_threshold=0.5, **kw):
    """Run multi-target tracker on a particular sequence.
    """
    import tqdm
    import supervision as sv
    model = load_model()

    if out_file is True:
        out_file='dino_'+os.path.basename(src)
    assert out_file
    print("Writing to:", os.path.abspath(out_file))

    print("classes:", classes)
    text_prompt = ' . '.join(classes)

    box_annotator = sv.BoxAnnotator()

    video_info = sv.VideoInfo.from_video_path(src)
    with sv.VideoSink(out_file, video_info=video_info) as s:
        for i, frame in tqdm.tqdm(enumerate(sv.get_video_frames_generator(src)), total=video_info.total_frames):
            # if i > 100: break
            im = torch.as_tensor(frame.transpose(2, 0, 1)[None]).float().to(device)
            boxes, logits, phrases = predict(model, im, text_prompt)
            h, w = frame.shape[:2]
            detections = sv.Detections(
                xyxy=box_convert(boxes=boxes * torch.Tensor([w, h, w, h]), in_fmt="cxcywh", out_fmt="xyxy").numpy(),
                confidence=logits.numpy(),
            )

            frame = frame.copy()
            frame = box_annotator.annotate(
                scene=frame,
                detections=detections,
                labels=[
                    f"{label} {confidence:0.2f}"
                    for (_, _, confidence, _, _), label
                    in zip(detections, phrases)
                ]
            )
            s.write_frame(frame)
    if s._VideoSink__writer is not None:
        s._VideoSink__writer.release()
    

if __name__ == '__main__':
    import fire
    fire.Fire(run)