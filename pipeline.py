import ast
import json
import sys
import os
import io
import gc
from shutil import ExecError
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager

import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image

from helpers import Deepfaune, crop_normalized_bbox_square, predict_batch, class_names


@contextmanager
def suppress_stdout():
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout

@contextmanager
def capture_stdout():
    old_stdout = sys.stdout
    buffer = io.StringIO()
    sys.stdout = buffer
    try:
        yield buffer
    finally:
        sys.stdout = old_stdout





def run_megadetector(images_directory: str, BATCH_SIZE_MD: int, N_CORES: int):
    # images
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    directory = Path(images_directory)
    images_paths = [
        str(p) for p in directory.rglob("*")
        if p.suffix.lower() in image_extensions
    ]

    # run MD
    with capture_stdout() as detector_output:
        results = load_and_run_detector_batch("MDV5A", images_paths, confidence_threshold=0.2, batch_size=BATCH_SIZE_MD, n_cores=N_CORES)

    output_text = detector_output.getvalue()
    if "CUDA out of memory" in output_text:
        raise ExecError("Lower detector batch size. Your GPU doesn't have this much memory.")

    # save output
    images_paths_sec, categories, confs, bboxes, n_animals = [], [], [], [], []

    for result in results:
        images_paths_sec.append(result.get('file'))
        
        detections = result.get('detections')
        if detections:
            detection = max(detections, key=lambda d: d["conf"])
            categories.append(detection.get('category'))
            confs.append(detection.get('conf'))
            bboxes.append(detection.get('bbox'))
            n_animals.append(len(detections))
        else:
            categories.append(None)
            confs.append(None)
            bboxes.append(None)
            n_animals.append(0)

    results_df = pd.DataFrame({'image_path': images_paths_sec, 'category': categories, 'conf': confs, 'bbox': bboxes, 'n_animals': n_animals})
    results_df.to_csv(f'megadetector_results.csv')

    with open('megadetector_raw_results.json', 'w') as output_file:
        json.dump(results, output_file, indent=4)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def run_pipeline(BATCH_SIZE: int):
    checkpoint_path = os.path.join('model', 'deepfaune_polish_lr4_checkpoint.pt')

    # classifier model
    with suppress_stdout() as detector_output:
        model_wrapper = Deepfaune(checkpoint_path)
    classifier = model_wrapper.model.base_model
    classifier.to('cuda')
    transforms = model_wrapper.transforms

    images = pd.read_csv('megadetector_results.csv', index_col=0)
    
    images['bbox'] = images["bbox"].apply(
        lambda b: ast.literal_eval(b) if isinstance(b, str) else None)

    batch = []
    paths = []
    results = pd.DataFrame({'image': [], 'detected_animal': [], 'confidence': []})

    for _, row in tqdm(images.iterrows(), total=len(images)):
        image_path = row['image_path']

        # only animals
        category = row['category']
        if category != 1:
            results.loc[len(results)] = [image_path, 'empty', 0]
            continue

        # image
        try:
            image = Image.open(image_path).convert("RGB")
            cropped_image = crop_normalized_bbox_square(image, row['bbox'])
        except Exception as e:
            # print(f'Error in image {image_path}: {e}')
            continue

        paths.append(image_path)
        batch.append(cropped_image)

        # run classifier every N images (e.g. 32)
        if len(batch) == BATCH_SIZE:
            preds = predict_batch(classifier, batch, transforms, class_names)
            # if confidence (prediction[0][1]) is less than 0.1, classify as other
            detections = [
                prediction[0][0] if prediction[0][1] > 0.1 else 'other' for prediction in preds]
            confs = [prediction[0][1] for prediction in preds]

            batch_results = pd.DataFrame(
                {'image': paths, 'detected_animal': detections, 'confidence': confs})
            results = pd.concat([results, batch_results], ignore_index=True)
            # if confidence less than threshold: other
            batch = []
            paths = []

    if len(batch) > 0:
        preds = predict_batch(classifier, batch, transforms, class_names)
        detections = [
            prediction[0][0] if prediction[0][1] > 0.1 else 'other' for prediction in preds]
        confs = [prediction[0][1] for prediction in preds]

        batch_results = pd.DataFrame(
            {'image': paths, 'detected_animal': detections, 'confidence': confs})
        results = pd.concat([results, batch_results], ignore_index=True)

    now = datetime.now().strftime('%Y_%m_%d_%H_%M')
    results.to_csv(f'results_{now}.csv')

    # --- GPU cleanup: move model to CPU, delete references, collect garbage ---
    try:
        # classifier was moved to cuda earlier; move it back to free GPU memory
        classifier.to('cpu')
    except Exception:
        pass

    try:
        # remove references to model objects
        del classifier
    except Exception:
        pass

    try:
        del model_wrapper
    except Exception:
        pass

    # Force python GC and release CUDA cache
    gc.collect()
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        pass
