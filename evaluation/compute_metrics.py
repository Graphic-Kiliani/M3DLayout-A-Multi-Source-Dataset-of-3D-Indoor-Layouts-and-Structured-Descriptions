'''

pip install torch torchvision
pip install pillow
pip install tqdm
pip install git+https://github.com/openai/CLIP.git

'''

'''
###arg
input:
--json: a path of json file containing text descriptions for each scene, it is used to compute the CLIP score
--path_to_test_real: path to the real images for FID/KID computation
--path_to_test_fake: path to the generated images for FID/KID computation

output:
--output: output file path to save the results
'''

import os
import json
import numpy as np
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
import clip
import glob
from cleanfid import fid
import random

def load_clip_model(model_name: str = "ViT-B/32"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Loading CLIP model: {model_name}")
    model, preprocess = clip.load(model_name, device=device)
    return model, preprocess, device


def compute_clip_scores(
    json_path: str,
    images_dir: str,
    model_name: str = "ViT-B/32",
    batch_size: int = 8,  
):
    model, preprocess, device = load_clip_model(model_name)

    # Load the JSON file, which stores the output results from the model's inference phase. Here, the prompts are primarily used to compute the CLIP score.
    with open(json_path, "r") as f:
        scenes_data = json.load(f)

    print(f"[INFO] Loading {len(scenes_data)} scene descriptions")

    scores = []
    all_pairs = []

    # Load all the generated images, which correspond one-to-one with the elements in the JSON file.
    existing_scenes = glob.glob(os.path.join(images_dir, "*.png"), recursive=True)
    existing_scenes = [os.path.basename(f) for f in existing_scenes]

    print(f"[INFO] Found {len(existing_scenes)} rendered scenes: {existing_scenes}")
    
    assert len(scenes_data) == len(existing_scenes), "The number of files is not the same."

    for scene_index, scenex in enumerate(tqdm(scenes_data, desc="Processing scenes")):
   
        if "text" in scenex:
            text_description = scenex['text']
        else:
            text_description = scenex['description'] # text、description
        

        if not text_description:
            print(f"[WARN] Scene {scene_index} has no text description")
            continue
        
        # Truncate excessively long text descriptions.
        if len(text_description) > 300:
            text_description = text_description[:300] + "..."
            print(f"[INFO] Scene {scene_index} text truncated to 300 characters")

        image_path = os.path.join(images_dir, f"scene_{scene_index}.png")


        try:
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")

            image_tensor = preprocess(image).unsqueeze(0).to(device)
            text = clip.tokenize([text_description]).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image_tensor)
                text_features = model.encode_text(text)

                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                similarity = ( image_features @ text_features.T).item()  #100.0 *

        except Exception as e:
            print(f"[WARN] Error when processing {image_path}: {e}")
            continue

        scores.append(similarity)
        all_pairs.append(
            {
                "index": scene_index,
                "text": text_description,
                "image": str(image_path),
                "clip_score": similarity,
            }
        )

    mean_score = float(np.mean(scores)) if scores else 0.0
    return scores, mean_score, all_pairs


def save_results(fid_score, kid_score, scores, mean_score, pairs, output_file: str):
    sorted_pairs = sorted(pairs, key=lambda x: x["clip_score"], reverse=True)

    with open(output_file, "w", encoding='utf-8') as f:
        f.write(f"FID score: {fid_score:.4f}\n")
        f.write(f"KID score: {kid_score:.5f}\n")
        f.write(f"Average CLIP score: {mean_score:.4f}\n")

        if scores:
            f.write(f"The highest CLIP score: {max(scores):.4f}\n")
            f.write(f"The lowest CLIP score: {min(scores):.4f}\n")
            f.write(f"The median CLIP score: {np.median(scores):.4f}\n")
            f.write(f"The standard deviation: {np.std(scores):.4f}\n\n")
        else:
            f.write("No valid scores.\n\n")

        f.write("The CLIP scores of each scene (sorted in descending order):\n")
        f.write("-" * 80 + "\n")

        for pair in sorted_pairs:
            index = pair["index"]
            score = pair["clip_score"]
            text = pair["text"]
            image = pair["image"]

            if len(text) > 60:
                text = text[:57] + "..."

            f.write(f"Scene {index}: {score:.4f} - {text}\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write(f"Total number of scene-picture pairs: {len(scores)}\n")

def compute_fidkid(path_to_test_real, path_to_test_fake):
    # Compute the FID score
    fid_score = fid.compute_fid(path_to_test_real, path_to_test_fake, device=torch.device("cpu"))
    print('fid score:', fid_score)
    # Compute the KID score
    kid_score = fid.compute_kid(path_to_test_real, path_to_test_fake, device=torch.device("cpu"))
    print('kid score:', kid_score)

    return fid_score, kid_score

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate CLIP Score between text descriptions and images"
    )
    parser.add_argument(
        "--json",
        default="./data/bedroom_livingroom_diningroom.json",
        help="JSON file path",
    )
    parser.add_argument(
        "--path_to_test_real",
        default="./data/real/infinigen",
        help="FID/KID real image path",
    )
    parser.add_argument(
        "--path_to_test_fake",
        default="./data/fake/infinigen",
        help="FID/KID Rendered image path",
    )
    parser.add_argument(
        "--output",
        default="./results/ar_ablation.txt",
        help="Output path",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="(reserved) batch size"
    )
    parser.add_argument(
        "--model",
        default="ViT-B/32",
        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14"],
        help="CLIP model edition",
    )

    args = parser.parse_args()

    print("[INFO] Starting calculating CLIP score")
    scores, mean_score, pairs = compute_clip_scores(
        args.json, args.path_to_test_fake, model_name=args.model, batch_size=args.batch_size
    )

    print("[INFO] Starting calculating FID/KID")
    fid_score, kid_score = compute_fidkid(args.path_to_test_real, args.path_to_test_fake)
    print("fid_score, kid_score:", fid_score, kid_score)
    print(f"[INFO] Evaluating {len(scores)} scene-image pairs")
    print(f"[INFO] Average CLIP score: {mean_score:.4f}")

    save_results(fid_score, kid_score, scores, mean_score, pairs, args.output)
    print(f"[INFO] Results saved to {args.output}")


if __name__ == "__main__":
    main()





