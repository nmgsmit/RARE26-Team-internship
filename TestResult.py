import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torchvision.transforms.v2 import Compose, Normalize, Resize, ToDtype, ToImage
from tqdm import tqdm

from model import Model


LABEL_TO_INDEX = {
	"NDBT": 0,  # healthy tissue
	"ACHD": 1,  # malignant tumor
}


def parse_args():
	parser = argparse.ArgumentParser("Evaluate a checkpoint on EVC_Barretts_FullSet images")
	parser.add_argument(
		"--model-path",
		type=str,
		required=True,
		help="Path to a trained checkpoint (.pt/.pth) containing model.state_dict().",
	)
	parser.add_argument(
		"--images-dir",
		type=str,
		default="data/EVC_Barretts_FullSet/images",
		help="Directory containing EVC images named like patXX_imY_ACHD.png or patXX_imY_NDBT.png.",
	)
	parser.add_argument(
		"--image-size",
		type=int,
		default=224,
		help="Square resize used before inference.",
	)
	parser.add_argument(
		"--batch-size",
		type=int,
		default=32,
		help="Batch size used for inference.",
	)
	parser.add_argument(
		"--backbone-name",
		type=str,
		default="vit_base_patch16_dinov3",
		help="Backbone model name passed to timm.",
	)
	parser.add_argument(
		"--no-pretrained",
		action="store_true",
		help="Do not initialize with pretrained timm weights before loading checkpoint.",
	)
	parser.add_argument(
		"--save-csv",
		type=str,
		default=None,
		help="Optional output CSV path to save per-image predictions.",
	)
	return parser.parse_args()


def build_transform(image_size):
	return Compose(
		[
			ToImage(),
			Resize((image_size, image_size)),
			ToDtype(torch.float32, scale=True),
			Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		]
	)


def infer_label_from_filename(image_path):
	stem = image_path.stem.upper()
	if stem.endswith("_ACHD"):
		return LABEL_TO_INDEX["ACHD"]
	if stem.endswith("_NDBT"):
		return LABEL_TO_INDEX["NDBT"]
	raise ValueError(
		f"Could not infer class from filename '{image_path.name}'. "
		"Expected suffix _ACHD or _NDBT before extension."
	)


def load_batch(paths, transform, device):
	images = []
	for img_path in paths:
		img = Image.open(img_path).convert("RGB")
		images.append(transform(img))
	return torch.stack(images, dim=0).to(device)


def evaluate(model, image_paths, transform, device, batch_size):
	y_true = []
	y_pred = []
	records = []

	model.eval()
	with torch.no_grad():
		for start in tqdm(range(0, len(image_paths), batch_size), desc="Evaluating"):
			batch_paths = image_paths[start : start + batch_size]
			inputs = load_batch(batch_paths, transform, device)
			logits = model(inputs)
			preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()

			for path, pred in zip(batch_paths, preds):
				true_label = infer_label_from_filename(path)
				y_true.append(true_label)
				y_pred.append(pred)
				records.append((path.name, true_label, pred))

	return np.array(y_true), np.array(y_pred), records


def compute_metrics(y_true, y_pred):
	cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
	tn, fp, fn, tp = cm.ravel()

	metrics = {
		"accuracy": accuracy_score(y_true, y_pred),
		"precision_achd": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
		"recall_achd_sensitivity": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
		"specificity_ndbt": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
		"f1_achd": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
		"tn": int(tn),
		"fp": int(fp),
		"fn": int(fn),
		"tp": int(tp),
	}
	return metrics, cm


def save_records_csv(csv_path, records):
	header = "image_name,true_label,pred_label,true_name,pred_name\n"
	rows = [header]
	index_to_name = {0: "NDBT", 1: "ACHD"}

	for name, true_idx, pred_idx in records:
		rows.append(
			f"{name},{true_idx},{pred_idx},{index_to_name.get(true_idx, 'UNK')},{index_to_name.get(pred_idx, 'UNK')}\n"
		)

	output_path = Path(csv_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text("".join(rows), encoding="utf-8")


def main():
	args = parse_args()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	images_dir = Path(args.images_dir)
	if not images_dir.exists():
		raise FileNotFoundError(f"Images directory not found: {images_dir}")

	image_paths = sorted(images_dir.glob("*.png"))
	if len(image_paths) == 0:
		raise ValueError(f"No .png images found in {images_dir}")

	model = Model(
		in_channels=3,
		n_classes=2,
		backbone_name=args.backbone_name,
		pretrained=not args.no_pretrained,
	).to(device)

	state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
	model.load_state_dict(state_dict, strict=True)

	transform = build_transform(args.image_size)
	y_true, y_pred, records = evaluate(model, image_paths, transform, device, args.batch_size)
	metrics, cm = compute_metrics(y_true, y_pred)

	print("\nEvaluation on EVC_Barretts_FullSet")
	print(f"Total images: {len(image_paths)}")
	print("Label mapping: NDBT=0 (healthy), ACHD=1 (malignant)")
	print(f"Accuracy: {metrics['accuracy']:.4f}")
	print(f"Precision (ACHD): {metrics['precision_achd']:.4f}")
	print(f"Recall/Sensitivity (ACHD): {metrics['recall_achd_sensitivity']:.4f}")
	print(f"Specificity (NDBT): {metrics['specificity_ndbt']:.4f}")
	print(f"F1 (ACHD): {metrics['f1_achd']:.4f}")
	print("Confusion matrix [[TN, FP], [FN, TP]]:")
	print(cm)

	if args.save_csv:
		save_records_csv(args.save_csv, records)
		print(f"Saved per-image predictions to: {args.save_csv}")


if __name__ == "__main__":
	main()
