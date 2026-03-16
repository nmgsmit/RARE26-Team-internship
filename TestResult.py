import argparse
import inspect
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2 import Compose, Normalize, Resize, ToDtype, ToImage
from tqdm import tqdm

from model import Model


LABEL_TO_INDEX = {
	"NDBT": 0,  # healthy tissue
	"ACHD": 1,  # malignant tumor
}

IMAGES_DIR = Path("data/EVC_Barretts_FullSet/images")
THRESHOLD = 0.5


def _get_model_defaults():
	sig = inspect.signature(Model.__init__)
	backbone_default = sig.parameters["backbone_name"].default
	pretrained_default = sig.parameters["pretrained"].default
	return backbone_default, pretrained_default


def _get_training_defaults():
	batch_size_default = 32
	image_size_default = 224

	try:
		import train as train_module

		train_defaults = vars(train_module.get_args_parser().parse_args([]))
		batch_size_default = int(train_defaults.get("batch_size", batch_size_default))
		image_size_default = int(getattr(train_module, "DEFAULT_IMAGE_SIZE", image_size_default))
	except Exception:
		pass

	return batch_size_default, image_size_default


BACKBONE_NAME, PRETRAINED = _get_model_defaults()
BATCH_SIZE, IMAGE_SIZE = _get_training_defaults()


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
		default=str(IMAGES_DIR),
		help="Directory containing input .png images.",
	)
	parser.add_argument(
		"--image-size",
		type=int,
		default=IMAGE_SIZE,
		help="Square resize used before inference.",
	)
	parser.add_argument(
		"--batch-size",
		type=int,
		default=BATCH_SIZE,
		help="Batch size used for inference.",
	)
	parser.add_argument(
		"--backbone-name",
		type=str,
		default=BACKBONE_NAME,
		help="Backbone model name passed to timm.",
	)
	parser.add_argument(
		"--pretrained",
		action="store_true",
		help="Initialize timm backbone with pretrained weights before loading checkpoint.",
	)
	parser.add_argument(
		"--no-pretrained",
		action="store_false",
		dest="pretrained",
		help="Disable pretrained timm weights.",
	)
	parser.add_argument(
		"--threshold",
		type=float,
		default=THRESHOLD,
		help="Decision threshold for turning ACHD probability into class label.",
	)
	parser.set_defaults(pretrained=PRETRAINED)
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


class EVCDataset(Dataset):
	def __init__(self, image_paths, transform):
		self.image_paths = image_paths
		self.transform = transform
		self.labels = [infer_label_from_filename(p) for p in image_paths]

	def __len__(self):
		return len(self.image_paths)

	def __getitem__(self, idx):
		img = Image.open(self.image_paths[idx]).convert("RGB")
		return self.transform(img), self.labels[idx]


def evaluate(model, image_paths, transform, device, batch_size):
	dataset = EVCDataset(image_paths, transform)
	loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

	y_true = []
	y_score = []

	model.eval()
	with torch.no_grad():
		for images, labels in tqdm(loader, desc="Evaluating"):
			images = images.to(device)
			logits = model(images)
			probs_achd = torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist()
			y_true.extend(labels.tolist())
			y_score.extend(probs_achd)

	return np.array(y_true), np.array(y_score)


def compute_metrics(y_true, y_score, threshold):
	has_both_classes = len(np.unique(y_true)) == 2
	y_pred = (y_score >= threshold).astype(int)
	cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
	tn, fp, fn, tp = cm.ravel()
	precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
	recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
	f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

	if not has_both_classes:
		return float("nan"), float("nan"), precision, recall, f1, int(tp), int(fp), int(fn), int(tn), cm

	auroc = roc_auc_score(y_true, y_score)
	auprc = average_precision_score(y_true, y_score)
	return auroc, auprc, precision, recall, f1, int(tp), int(fp), int(fn), int(tn), cm


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
		pretrained=args.pretrained,
	).to(device)

	state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
	model.load_state_dict(state_dict, strict=True)

	transform = build_transform(args.image_size)
	y_true, y_score = evaluate(model, image_paths, transform, device, args.batch_size)
	auroc, auprc, precision, recall, f1, tp, fp, fn, tn, cm = compute_metrics(y_true, y_score, args.threshold)

	print("\nEvaluation on EVC_Barretts_FullSet")
	print(f"Total images: {len(image_paths)}")
	print(f"Backbone: {args.backbone_name}")
	print(f"Threshold: {args.threshold:.2f}")
	print(f"AUROC: {auroc:.4f}")
	print(f"AUPRC: {auprc:.4f}")
	print(f"Precision: {precision:.4f}")
	print(f"Recall: {recall:.4f}")
	print(f"F1: {f1:.4f}")
	print(f"TP: {tp}")
	print(f"FP: {fp}")
	print(f"FN: {fn}")
	print(f"TN: {tn}")
	print("Confusion matrix [[TN, FP], [FN, TP]]:")
	print(cm)


if __name__ == "__main__":
	main()
