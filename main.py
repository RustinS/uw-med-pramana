import logging
import time
import traceback

import cv2
import numpy as np
import torch
from model_arch import ModelArch
from PIL import Image
from src.inline_algorithm.inline_algo_queue_processor import InlineAlgoQueueProcessor
from transformers import AutoProcessor, CLIPVisionModel

PATCH_SIZE = 256


def apply_otsu_threshold(patch):
    _, binary_patch = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_patch


class TestChild(InlineAlgoQueueProcessor):
    def __init__(self, port, host, docker_mode=True):
        super().__init__(port, host, docker_mode)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        self.init_processor()

    def init_processor(self):
        self.mag_levels = [40, 20, 10]
        self.features = {40: [], 20: [], 10: []}
        self.coordinates = {40: [], 20: [], 10: []}

        self.mag_power = None
        self.tile_width = None
        self.tile_height = None

        self.embed_model = CLIPVisionModel.from_pretrained("wisdomik/QuiltNet-B-32")
        for param in self.embed_model.parameters():
            param.requires_grad = False
        self.embed_dim = 768

        self.transformlist = AutoProcessor.from_pretrained("wisdomik/QuiltNet-B-32")

        self.diag_model = ModelArch(backbone_dim=self.embed_dim, n_classes=5)
        self.diag_model.load_state_dict(torch.load("saved_models/diag_model.pt"))
        self.diag_model.to("cpu")
        self.diag_model = torch.nn.DataParallel(self.diag_model)
        self.diag_model.eval()

        self.log_file_address = "/homes/gws/rustin/UW-Med/uw-med-pramana/log.txt"

        for handler in self.logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                self.logger.removeHandler(handler)

        file_handler = logging.FileHandler(self.log_file_address, mode="w")
        formatter = logging.Formatter("%(message)s")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        self.logger.info("Model ready for tiles.\n\n")
        print("Model ready for tiles.")

    def on_server_start(self):
        print("loading model")
        self.init_processor()

    def on_server_end(self):
        print("freeing memory")
        del self.embed_model, self.diag_model, self.transformlist, self.features, self.coordinates

    def on_scan_start(self, message):
        self.init_processor()

        self.base_mag_power = float(message.available_magnifications[0][:-1])
        self.tile_width = message.tile_width
        self.tile_height = message.tile_height
        self.embed_model.to("cuda")

        self.logger.info(f"Base mag power: {self.base_mag_power} - Tile Width: {self.tile_width} - Tile Height: {self.tile_height}\n")

    def on_scan_end(self, message):
        return self.aggregate()

    def on_scan_abort(self, message):
        self.init_processor()

    def process(self, message):
        try:
            process_start = time.perf_counter()

            load_start = time.perf_counter()
            orig_tile = cv2.cvtColor(cv2.imread(message.tile_image_path), cv2.COLOR_BGR2RGB)
            h, w = orig_tile.shape[:2]

            if h != self.tile_height or w != self.tile_width:
                return
            load_time = (time.perf_counter() - load_start) * 1000
            self.logger.info(f"Image load time: {load_time:.2f}ms")

            base_x = message.col_idx * self.tile_width
            base_y = message.row_idx * self.tile_height

            mag_times = {}
            for mag_level in self.mag_levels:
                mag_start = time.perf_counter()

                resize_start = time.perf_counter()
                scale_factor = int(40 / mag_level)
                target_w = int(w / scale_factor)
                target_h = int(h / scale_factor)

                patchable_w = ((target_w // PATCH_SIZE) + (1 if target_w % PATCH_SIZE >= PATCH_SIZE // 2 else 0)) * PATCH_SIZE
                patchable_h = ((target_h // PATCH_SIZE) + (1 if target_h % PATCH_SIZE >= PATCH_SIZE // 2 else 0)) * PATCH_SIZE

                resized = cv2.resize(
                    orig_tile, (patchable_w, patchable_h), interpolation=cv2.INTER_AREA if mag_level < 40 else cv2.INTER_LANCZOS4
                )
                resize_time = (time.perf_counter() - resize_start) * 1000

                patch_start = time.perf_counter()
                tile_gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
                tile_gray[tile_gray <= 8] = 0

                patches_gray = np.lib.stride_tricks.sliding_window_view(tile_gray, (PATCH_SIZE, PATCH_SIZE))[::PATCH_SIZE, ::PATCH_SIZE]
                patches_color = np.lib.stride_tricks.sliding_window_view(resized, (PATCH_SIZE, PATCH_SIZE, 3))[::PATCH_SIZE, ::PATCH_SIZE]

                binary_patches = np.array([apply_otsu_threshold(patch) for patch in patches_gray.reshape(-1, PATCH_SIZE, PATCH_SIZE)])
                tissue_percentages = np.mean(binary_patches.astype(bool), axis=(1, 2))

                valid_mask = tissue_percentages > 0.15
                valid_color_patches = patches_color.reshape(-1, PATCH_SIZE, PATCH_SIZE, 3)[valid_mask]
                patch_time = (time.perf_counter() - patch_start) * 1000

                embed_time = 0
                if len(valid_color_patches) > 0:
                    embed_start = time.perf_counter()
                    inputs = self.transformlist(images=valid_color_patches, return_tensors="pt")
                    with torch.no_grad():
                        outputs = self.embed_model(pixel_values=inputs.pixel_values.to("cuda"))

                    self.features[mag_level].extend(outputs.pooler_output.cpu())
                    embed_time = (time.perf_counter() - embed_start) * 1000

                    coord_start = time.perf_counter()
                    yy, xx = np.mgrid[0:patchable_h:PATCH_SIZE, 0:patchable_w:PATCH_SIZE]
                    valid_coords = np.column_stack([xx.ravel(), yy.ravel()])[valid_mask]
                    self.coordinates[mag_level].extend([(base_x + x, base_y + y) for x, y in valid_coords])
                    coord_time = (time.perf_counter() - coord_start) * 1000
                else:
                    coord_time = 0

                mag_total = (time.perf_counter() - mag_start) * 1000
                mag_times[mag_level] = mag_total

                self.logger.info(
                    f"Mag {mag_level}x processing: "
                    f"Resize: {resize_time:.2f}ms | "
                    f"Patches: {patch_time:.2f}ms | "
                    f"Embed: {embed_time:.2f}ms | "
                    f"Coords: {coord_time:.2f}ms | "
                    f"Total: {mag_total:.2f}ms"
                )

            total_time = (time.perf_counter() - process_start) * 1000
            self.logger.info(
                f"Processing complete | "
                f"Total: {total_time:.2f}ms | "
                f"40x: {len(self.features[40])} | "
                f"20x: {len(self.features[20])} | "
                f"10x: {len(self.features[10])}\n\n"
            )

        except Exception as e:
            self.logger.error(traceback.format_exc())
            self.logger.error(f"Error: {e}")
            self.logger.error((patchable_w, patchable_h))
            self.logger.error(orig_tile.shape)
            raise e

    def aggregate(self):
        self.logger.info("Aggregating features...\n")
        input_tensors = {}
        for mag_level in self.mag_levels:
            combined = list(zip(self.features[mag_level], self.coordinates[mag_level]))
            combined_sorted = sorted(combined, key=lambda item: (item[1][1], item[1][0]))
            sorted_coordinates, sorted_tensors = zip(*combined_sorted)
            wsi_emb = torch.cat(sorted_tensors, dim=0)
            input_tensors[mag_level] = wsi_emb
            self.logger.info(f"Mag Level: {mag_level} - Tensor Shape: {wsi_emb.shape}\n")

        self.logger.info("Aggregating finished...\n")
        self.logger.info("Running diagnosis model...\n")

        self.diag_model.to("cuda")
        with torch.no_grad():
            logits = self.diag_model(input_tensors)
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
        self.diag_model.to("cpu")

        self.logger.info(f"Running diagnosis model finished. Prediction : {predictions}\n")

        return predictions


if __name__ == "__main__":
    obj = TestChild(8000, "localhost", docker_mode=False)
    print("Running server...")
    obj.run()
