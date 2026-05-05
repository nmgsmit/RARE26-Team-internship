import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch

from data import TwoViewDataset
from gradcam import (
    _compute_cam_activation_stats,
    _overlay_soft_consensus,
    _select_display_cam_tensor,
    build_expert_consensus_masks,
    compute_soft_mask_mass,
    compute_pixel_average_precision,
)
from roi_guidance import (
    build_roi_record_from_cam,
    compute_normalized_bbox_from_binary_mask,
    load_roi_records_from_json,
    load_roi_records_from_masks,
    save_roi_records_to_json,
)
from testdata import build_barrett_gradcam_samples


class GradcamMetricTests(unittest.TestCase):
    def test_compute_pixel_average_precision_perfect_ranking(self):
        score_map = torch.tensor([
            [0.1, 0.2, 0.3],
            [0.2, 0.9, 0.8],
            [0.1, 0.7, 0.6],
        ], dtype=torch.float32)
        target_mask = torch.tensor([
            [0, 0, 0],
            [0, 1, 1],
            [0, 1, 1],
        ], dtype=torch.float32)

        ap = compute_pixel_average_precision(score_map, target_mask)
        self.assertAlmostEqual(ap, 1.0, places=6)

    def test_compute_pixel_average_precision_disjoint_ranking_is_low(self):
        score_map = torch.tensor([
            [0.9, 0.8, 0.7],
            [0.6, 0.5, 0.4],
            [0.3, 0.2, 0.1],
        ], dtype=torch.float32)
        target_mask = torch.tensor([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 1],
        ], dtype=torch.float32)

        ap = compute_pixel_average_precision(score_map, target_mask)
        self.assertLess(ap, 0.2)

    def test_build_expert_consensus_masks(self):
        expert_masks = torch.tensor([
            [[1, 0], [1, 0]],
            [[1, 1], [0, 0]],
            [[1, 0], [1, 0]],
            [[0, 1], [1, 0]],
            [[1, 0], [0, 0]],
        ], dtype=torch.float32)

        union_mask, majority_mask, soft_consensus = build_expert_consensus_masks(expert_masks)

        expected_union = torch.tensor([[1, 1], [1, 0]], dtype=torch.float32)
        expected_majority = torch.tensor([[1, 0], [1, 0]], dtype=torch.float32)
        expected_soft = torch.tensor([[0.8, 0.4], [0.6, 0.0]], dtype=torch.float32)

        self.assertTrue(torch.equal(union_mask, expected_union))
        self.assertTrue(torch.equal(majority_mask, expected_majority))
        self.assertTrue(torch.allclose(soft_consensus, expected_soft))

    def test_signed_cam_stats_do_not_mark_negative_structure_as_flat(self):
        signed_cam = torch.tensor([
            [-0.6, -0.2],
            [-0.1, -0.4],
        ], dtype=torch.float32)

        stats = _compute_cam_activation_stats(signed_cam)

        self.assertGreater(stats["raw_max_abs_activation"], 0.0)
        self.assertGreater(stats["raw_std_activation"], 0.0)
        self.assertEqual(stats["is_flat_or_near_zero"], 0.0)

    def test_display_cam_falls_back_to_abs_signed_map_when_positive_cam_is_blank(self):
        cam_tensor = torch.zeros((2, 2), dtype=torch.float32)
        signed_cam = torch.tensor([
            [-0.8, -0.2],
            [-0.1, -0.4],
        ], dtype=torch.float32)

        display_cam, label = _select_display_cam_tensor(cam_tensor, signed_cam)

        self.assertEqual(label, "abs(signed cam)")
        self.assertAlmostEqual(float(display_cam.max().item()), 1.0, places=6)
        self.assertGreater(float(display_cam.std().item()), 0.0)

    def test_soft_consensus_overlay_uses_high_contrast_fill_and_outline(self):
        base_rgb = np.zeros((5, 5, 3), dtype=np.uint8)
        soft_mask = torch.zeros((5, 5), dtype=torch.float32)
        soft_mask[1:4, 1:4] = 1.0

        overlay = _overlay_soft_consensus(base_rgb, soft_mask, outline_threshold=0.5, max_alpha=0.55)

        self.assertTrue(np.array_equal(overlay[1, 1], np.array([255, 255, 255], dtype=np.uint8)))
        self.assertGreater(int(overlay[2, 2, 2]), int(overlay[2, 2, 1]))
        self.assertGreater(int(overlay[2, 2, 1]), int(overlay[2, 2, 0]))

    def test_soft_consensus_mass_measures_fraction_of_cam_inside_consensus(self):
        score_map = torch.tensor([
            [1.0, 1.0],
            [0.0, 0.0],
        ], dtype=torch.float32)
        soft_mask = torch.tensor([
            [1.0, 0.5],
            [0.0, 0.0],
        ], dtype=torch.float32)

        mass = compute_soft_mask_mass(score_map, soft_mask)

        self.assertAlmostEqual(mass, 0.75, places=6)


class BarrettGroupingTests(unittest.TestCase):
    def _save_binary_mask(self, path, array):
        Image.fromarray((array.astype(np.uint8) * 255), mode="L").convert("1").save(path)

    def test_build_barrett_gradcam_samples_groups_all_experts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir)
            images_dir = dataset_root / "images"
            annotations_dir = dataset_root / "annotations_bmp"
            images_dir.mkdir()
            annotations_dir.mkdir()

            image_a = np.zeros((4, 4, 3), dtype=np.uint8)
            image_b = np.ones((4, 4, 3), dtype=np.uint8) * 64
            Image.fromarray(image_a, mode="RGB").save(images_dir / "pat01_im1_ACHD.png")
            Image.fromarray(image_b, mode="RGB").save(images_dir / "pat02_im1_NDBT.png")

            positive_mask = np.zeros((4, 4), dtype=np.uint8)
            positive_mask[1:3, 1:3] = 1
            negative_mask = np.zeros((4, 4), dtype=np.uint8)

            for expert_idx in range(1, 6):
                self._save_binary_mask(
                    annotations_dir / f"pat01_im1_ACHD_exp{expert_idx}.bmp",
                    positive_mask,
                )
                self._save_binary_mask(
                    annotations_dir / f"pat02_im1_NDBT_exp{expert_idx}.bmp",
                    negative_mask,
                )

            samples, qa_stats = build_barrett_gradcam_samples(dataset_root)

            self.assertEqual(len(samples), 2)
            self.assertEqual(qa_stats["image_count"], 2)
            self.assertEqual(qa_stats["positive_image_count"], 1)
            self.assertEqual(qa_stats["negative_image_count"], 1)
            self.assertEqual(qa_stats["annotations_per_image_min"], 5)
            self.assertEqual(qa_stats["annotations_per_image_max"], 5)
            self.assertEqual(samples[0]["mask_paths"][0].name, "pat01_im1_ACHD_exp1.bmp")
            self.assertEqual(samples[0]["mask_paths"][-1].name, "pat01_im1_ACHD_exp5.bmp")


class RoiGuidanceTests(unittest.TestCase):
    def test_compute_normalized_bbox_from_binary_mask(self):
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[1:3, 2:4] = 1

        bbox = compute_normalized_bbox_from_binary_mask(mask)

        self.assertEqual(bbox, (0.5, 0.25, 1.0, 0.75))

    def test_build_roi_record_from_cam_uses_thresholded_hot_region(self):
        cam = np.array([
            [0.1, 0.1, 0.1, 0.1],
            [0.1, 0.9, 0.9, 0.1],
            [0.1, 0.9, 0.9, 0.1],
            [0.1, 0.1, 0.1, 0.1],
        ], dtype=np.float32)

        roi_record = build_roi_record_from_cam(cam, threshold=0.6, score=0.87)

        self.assertIsNotNone(roi_record)
        self.assertEqual(roi_record["source"], "gradcam")
        self.assertAlmostEqual(roi_record["score"], 0.87, places=6)
        self.assertEqual(roi_record["bbox"], (0.25, 0.25, 0.75, 0.75))

    def test_load_roi_records_from_masks_matches_training_images(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            masks_dir = root / "masks"
            masks_dir.mkdir()
            image_path = root / "case01_ACHD.png"
            Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8), mode="RGB").save(image_path)

            mask = np.zeros((8, 8), dtype=np.uint8)
            mask[2:6, 1:5] = 255
            Image.fromarray(mask, mode="L").save(masks_dir / "case01_ACHD_mask.png")

            roi_records, matched_images, unmatched_images = load_roi_records_from_masks(
                [image_path],
                masks_dir,
            )

            self.assertEqual(len(roi_records), 1)
            self.assertEqual(matched_images, [str(image_path)])
            self.assertEqual(unmatched_images, [])
            self.assertEqual(roi_records[str(image_path)]["source"], "mask")

    def test_roi_records_json_round_trip_preserves_bbox_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "train_rois.json"
            roi_records = {
                "image_a.png": {
                    "bbox": (0.1, 0.2, 0.9, 0.8),
                    "coverage": 0.25,
                    "score": 0.91,
                    "source": "gradcam",
                }
            }
            metadata = {"checkpoint": "model.pt", "roi_records_total": 1}

            save_roi_records_to_json(json_path, roi_records, metadata=metadata)
            loaded_records, loaded_metadata = load_roi_records_from_json(json_path)

            self.assertEqual(loaded_metadata, metadata)
            self.assertEqual(loaded_records["image_a.png"]["bbox"], roi_records["image_a.png"]["bbox"])
            self.assertEqual(loaded_records["image_a.png"]["source"], "gradcam")
            self.assertAlmostEqual(loaded_records["image_a.png"]["score"], 0.91, places=6)

    def test_two_view_dataset_replaces_second_view_with_roi_crop(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "case01_ACHD.png"
            image = np.zeros((10, 10, 3), dtype=np.uint8)
            image[3:7, 3:7] = 255
            Image.fromarray(image, mode="RGB").save(image_path)

            df = pd.DataFrame({"img": [str(image_path)], "label": [1]})
            dataset = TwoViewDataset(
                df,
                transform1=lambda image: image.size,
                transform2=lambda image: image.size,
                roi_transform2=lambda image: image.size,
                roi_focus_prob=1.0,
                roi_context_scale=1.0,
                roi_min_crop_scale=0.1,
                roi_center_jitter=0.0,
            )
            dataset.set_roi_records(
                {
                    str(image_path): {
                        "bbox": (0.3, 0.3, 0.7, 0.7),
                        "coverage": 0.16,
                        "score": 1.0,
                        "source": "mask",
                    }
                }
            )

            view1_size, view2_size, label = dataset[0]

            self.assertEqual(label, 1)
            self.assertEqual(view1_size, (10, 10))
            self.assertEqual(view2_size, (4, 4))


if __name__ == "__main__":
    unittest.main()
