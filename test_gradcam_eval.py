import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from gradcam import build_expert_consensus_masks, compute_pixel_average_precision
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


if __name__ == "__main__":
    unittest.main()
