"""
Gradio demo for M3DLayout: 3D Scene Layout Generation from Text Descriptions
"""

import logging
import os
import random
import sys
import json
import time
import tempfile
from typing import List, Dict, Optional
import gradio as gr

import numpy as np
import torch
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from scene_synthesis.networks.autoregressive import build_network
from viz3dl import LayoutVisualizer, SceneLayout, SceneObject

def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=Loader)
    return config

DATASET_STATS = {
    "bounds_translations": [-13.487928, 0.005005026236176491, -23.7458945, 14.401651999999999, 8.350419044494629, 33.324425000000005],
    "bounds_sizes": [0.00578713696449995, 0.0016704245936125517, 0.00101662, 14.0783, 4.469733238220215, 6.72356653213501],
    "bounds_angles": [-3.1415927410125732, 3.1415927410125732],
    "class_labels": ["appliances", "armchair", "balloon", "bathtub", "bed", "beverage_fridge", "book", "book_column", "book_stack", "bookshelf", "bottle", "bowl", "box", "cabinet", "can", "ceiling_lamp", "cell_shelf", "chair", "chaise_longue_sofa", "chest_of_drawers", "children_cabinet", "chinese_chair", "chopsticks", "clothes", "clutter", "coffee_table", "console_table", "corner_side_table", "counter", "cup", "curtain", "cushion", "decoration", "desk", "desk_lamp", "dining_chair", "dining_table", "dishwasher", "dressing_chair", "dressing_table", "floor_lamp", "food_bag", "food_box", "fork", "fruit_container", "glass_panel_door", "hardware", "jar", "kids_bed", "kitchen_cabinet", "kitchen_space", "knife", "l_shaped_sofa", "large_plant_container", "large_shelf", "lazy_sofa", "lighting", "lite_door", "lounge_chair", "louver_door", "loveseat_sofa", "microwave", "mirror", "monitor", "multi_seat_sofa", "nature_shelf_trinkets", "nightstand", "oven", "pan", "panel_door", "pendant_lamp", "picture", "plant", "plant_container", "plate", "pot", "round_end_table", "rug", "seating", "shelf", "shelving", "shower", "side_table", "sink", "sofa", "spoon", "standing_sink", "stool", "table", "toilet", "towel", "trashcan", "tv", "tv_monitor", "tv_stand", "vase", "wall_art", "wardrobe", "window", "wine_cabinet", "wineglass", "start", "end"]
}

class InferenceDataset:
    def __init__(self, config):
        self.config = config
        self.encoding_type = config.get("encoding_type", "diffusion")
        
        # Load stats
        self.stats = DATASET_STATS
        self._class_labels = self.stats["class_labels"]
        
        # Compute bounds
        self._centroids = (
            np.array(self.stats["bounds_translations"][:3]),
            np.array(self.stats["bounds_translations"][3:])
        )
        self._sizes = (
            np.array(self.stats["bounds_sizes"][:3]),
            np.array(self.stats["bounds_sizes"][3:])
        )
        self._angles = (
            np.array(self.stats["bounds_angles"][:1]), 
            np.array(self.stats["bounds_angles"][1:])
        )
        
        # Apply scaling
        self._apply_scaling()
        
        self._max_length = self.config.get("sample_num_points", 270)

    def _apply_scaling(self):
        t_min, t_max = self._centroids
        s_min, s_max = self._sizes
        a_min, a_max = self._angles
        
        self._translation_scale = (t_max - t_min) / 2.0
        self._translation_offset = (t_max + t_min) / 2.0
        self._size_scale = (s_max - s_min) / 2.0
        self._size_offset = (s_max + s_min) / 2.0
        self._angle_scale = (a_max - a_min) / 2.0
        self._angle_offset = (a_max + a_min) / 2.0

    @property
    def class_labels(self):
        return self._class_labels

    @property
    def n_classes(self):
        return len(self._class_labels)

    @property
    def max_length(self):
        return self._max_length

    @property
    def feature_size(self):
        if self.encoding_type == "diffusion":
            return 8 + self.n_classes
        elif self.encoding_type == "autoregressive":
            return 7 + self.n_classes
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type}")

    def post_process(self, sample_params):
        processed_params = {}
        for k, v in sample_params.items():
            if k == "class_labels" or k == "room_layout" or k == "description":
                processed_params[k] = v
            elif k == "translations":
                processed_params[k] = v * (self._translation_scale + 1e-8) + self._translation_offset
            elif k == "sizes":
                processed_params[k] = v * (self._size_scale + 1e-8) + self._size_offset
            elif k == "angles":
                if v.shape[-1] == 2:
                    denormalized_cos_sin = v * (self._angle_scale + 1e-8) + self._angle_offset
                    if len(denormalized_cos_sin.shape) == 3:
                        angles = np.arctan2(denormalized_cos_sin[:, :, 1:2], denormalized_cos_sin[:, :, 0:1])
                    else:
                        angles = np.arctan2(denormalized_cos_sin[:, 1:2], denormalized_cos_sin[:, 0:1])
                    processed_params[k] = angles
                elif v.shape[-1] == 1:
                    processed_params[k] = v * (self._angle_scale[0] + 1e-8) + self._angle_offset[0]
            else:
                processed_params[k] = v
        return processed_params

class M3DLayoutArGenerator:
    """Autoregressive scene generator used by the Gradio demo."""

    def __init__(
        self,
        config_file: Optional[str] = None,
        weight_file: str = "weights/autoregressive_59000.pth",
        max_boxes: Optional[int] = None,
        default_seed: int = 0,
    ):
        self.weight_file = os.path.abspath(weight_file)
        if not os.path.exists(self.weight_file):
            raise FileNotFoundError(f"Weight file not found: {self.weight_file}")
        self.config_file = config_file or os.path.abspath("config/m3dlayout_autoregressive.yaml")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.default_seed = default_seed

        logging.getLogger("trimesh").setLevel(logging.ERROR)

        print(f"Running on device: {self.device}")

        self.config = load_config(self.config_file)
        print(f"Loaded config from {self.config_file}")
        
        self.dataset = InferenceDataset(self.config["dataset"])
        self.large_object_categories = self.config["dataset"].get("large_object_categories", [])
        self.max_boxes = max_boxes or self.dataset.max_length
        print(f"Large object categories: {len(self.large_object_categories)} types")

        self.network, _, _ = build_network(
            self.dataset.feature_size,
            self.dataset.n_classes,
            self.config,
            self.weight_file,
            device=str(self.device),
            n_dependency_classes=self.dataset.max_length + 1,
        )
        self.network.eval()
        self.classes = np.array(self.dataset.class_labels)
        print(f"Class labels: {self.classes}, count: {len(self.classes)}")

        self.color_map = self.load_color_map()
        
    def load_color_map(self):
        """Load category color mapping"""
        try:
            with open("category_color_map.json", "r") as f:
                return json.load(f)
        except:
            return {}
    
    def generate_scene_from_text(self, text_description: str, seed: Optional[int] = None) -> Dict:
        """Generate 3D scene from text description"""
        if not text_description.strip():
            raise ValueError("Please input text description")
        
        print(f"Generating scene from text: {text_description}")
        
        t1 = time.perf_counter()
        
        self._set_seed(seed)

        room_mask = torch.ones((1, 1, 64, 64), device=self.device)
        boxes = self.network.generate_boxes(
            room_mask=room_mask,
            max_boxes=self.max_boxes,
            device=self.device,
            text=text_description
        )

        processed_boxes = self.dataset.post_process(boxes)
        filtered_boxes = self._filter_start_end_tokens(processed_boxes)

        class_labels = filtered_boxes["class_labels"][0]
        class_names = [self.classes[torch.argmax(cc)] for cc in class_labels]
        translations = filtered_boxes["translations"][0].cpu().numpy().tolist()
        sizes = filtered_boxes["sizes"][0].cpu().numpy().tolist()
        angles = filtered_boxes["angles"][0].cpu().numpy().tolist()
        
        time_elapsed = time.perf_counter() - t1

        print(f'Generated {len(class_names)} objects: {class_names}')
        print(f'Generation time: {time_elapsed:.2f}s')
        
        # Build result
        result = {
            "text": text_description,
            "class_names": class_names,
            "translations": translations,
            "sizes": sizes,
            "angles": angles,
            "time_elapsed": time_elapsed,
            "object_count": len(class_names)
        }
        
        return result

    def _filter_start_end_tokens(self, boxes: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Remove the autoregressive start/end tokens from the decoded boxes."""

        class_labels = boxes["class_labels"][0]
        end_indices = torch.where(class_labels[:, -1] == 1)[0]
        end_idx = end_indices[0].item() if len(end_indices) > 0 else class_labels.shape[0]
        valid_slice = slice(1, end_idx)

        filtered = {}
        for key in ("class_labels", "translations", "sizes", "angles"):
            filtered[key] = boxes[key][:, valid_slice, :]
        return filtered

    def _set_seed(self, seed: Optional[int]):
        """Seed all RNGs for reproducible generation."""

        seed_value = self.default_seed if seed is None else int(seed)
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)
        print(f"Using seed: {seed_value}")
    
    def visualize_scene(self, scene_data: Dict, output_path: str) -> str:
        """Visualize scene and generate animation"""
        scene_objects = []

        print("Visualization data shapes:")
        print(f"  class_names: {len(scene_data['class_names'])}")
        print(f"  translations: {len(scene_data['translations'])}")
        print(f"  sizes: {len(scene_data['sizes'])}")
        print(f"  angles: {len(scene_data['angles'])}")

        for obj_name, translation, size, rotation in zip(
            scene_data["class_names"],
            scene_data["translations"],
            scene_data["sizes"],
            scene_data["angles"],
        ):
            if obj_name not in self.color_map:
                self.color_map[obj_name] = (np.random.rand(3) * 220 + 20).astype(np.int32).tolist()
            color = tuple(int(c) for c in self.color_map[obj_name])
            yaw = rotation[0] if isinstance(rotation, (list, tuple)) else rotation
            if isinstance(rotation, np.ndarray):
                yaw = rotation.flat[0]
            scene_objects.append(
                SceneObject(
                    name=obj_name,
                    location=[float(v) for v in translation],
                    dimensions=[abs(float(v)) for v in size],
                    yaw=float(yaw),
                    color=color,
                )
            )

        scene_layout = SceneLayout(scene_objects)

        min_bounds, max_bounds = scene_layout.get_scene_bounds()
        print(f"Scene bounds - Min: {min_bounds}, Max: {max_bounds}")
        print(f"Scene center: {scene_layout.get_scene_center()}")
        print(f"Scene size: {scene_layout.get_scene_size()}")

        scene_layout.scale_to_unit_cube(target_extent=5.0)
        scene_layout.normalize_to_origin(align_floor=True)
        print("After scaling & alignment:")
        print(f"Scene bounds: {scene_layout.get_scene_bounds()}")

        visualizer = LayoutVisualizer(scene_layout)
        visualizer.render_rotation(
            output_path=output_path,
            radius=10.0,
            elevation=-10.0,
            start_azimuth=-40.0,
            end_azimuth=40.0,
            step=4.0,
            frame_duration_ms=100,
        )

        print(f"Animation saved to: {output_path}")
        return output_path


# Initialize generator at startup
print("Initializing M3DLayoutAr generator...")
generator = M3DLayoutArGenerator()
print("Generator initialized successfully!")

def generate_and_visualize(text_input: str, seed_value: Optional[float]) -> tuple:
    """Generate scene and visualize"""
    global generator
    
    if not text_input.strip():
        return None, "Please input text description", ""
    
    seed_int = None
    if seed_value is not None and seed_value != "":
        try:
            seed_int = int(seed_value)
        except (TypeError, ValueError):
            return None, "Seed must be an integer", ""

    try:
        # Generate scene
        scene_data = generator.generate_scene_from_text(text_input, seed=seed_int)
        
        # Create temporary file for saving animation
        temp_dir = tempfile.mkdtemp()
        gif_path = os.path.join(temp_dir, "scene_animation.gif")
        
        # Visualize scene
        generator.visualize_scene(scene_data, gif_path)
        
        # Build result info
        info_text = f"""
    **Seed:** {seed_int if seed_int is not None else generator.default_seed}

    **Generated objects:**
    {', '.join(scene_data['class_names'])}
        """
        
        return gif_path, "Scene generation successful!", info_text
        
    except Exception as e:
        error_msg = f"Generation failed: {str(e)}"
        print(error_msg)
        return None, error_msg, ""

def load_example_texts() -> List[str]:
    """Load example texts"""
    example_texts = []
    # Use default examples if no examples loaded
    if not example_texts:
        example_texts = [
            "The room is a dining room. The scene includes many detailed elements.",
            "The room has a sofa, a coffee table and a dining table.",
            "The room has a double bed and two nightstands. Do not include small decorative items.",
            "The room is a kitchen. The room contains a kitchen space and kitchen cabinets.",
            "The bathroom contains a toilet, a standing sink, and a bathhub.",
            "The room is a dining room. The room contains a large dining table and multiple chairs.",
        ]
    
    return example_texts

def create_gradio_interface():
    """Create Gradio interface"""
    
    # Load example texts
    examples = load_example_texts()
    
    # Create interface
    with gr.Blocks(title="M3DLayout: A Multi-Source Dataset of 3D Indoor Layouts and Structured Descriptions for 3D Generation") as demo:
        
        gr.Markdown("""
        # M3DLayout: A Multi-Source Dataset of 3D Indoor Layouts and Structured Descriptions for 3D Generation

        Generate 3D indoor scene layouts from natural language descriptions.
        
        ## Instructions:
        1. Input room description in the text box, and click "Generate Scene" button
        2. Wait a few seconds to view the generated 3D scene animation
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Text input
                text_input = gr.Textbox(
                    label="Room Description", 
                    placeholder="Describe the room you want, e.g.: The room has a double bed and two nightstands. Do not include small decorative items.",
                    lines=3,
                    max_lines=5
                )

                seed_input = gr.Number(
                    label="Random Seed",
                    value=0,
                    precision=0,
                    info="Use the same seed to reproduce a layout"
                )
                
                # Generate button
                generate_button = gr.Button("Generate Scene", variant="primary")
                generation_status = gr.Textbox(label="Generation Status", interactive=False)
                
                # Generation info
                generation_info = gr.Markdown(label="Generation Info")
                
            with gr.Column(scale=2):
                # Result display
                result_gif = gr.Image(label="Generated 3D Scene Animation", type="filepath")
        
        # Example texts
        gr.Markdown("## Example Texts (Click to use)")
        example_buttons = []
        
        with gr.Row():
            for i in range(0, len(examples), 2):
                with gr.Column():
                    for j in range(2):
                        if i + j < len(examples):
                            example_text = examples[i + j]
                            # Truncate long text for button display
                            button_text = example_text[:200] + "..." if len(example_text) > 200 else example_text
                            btn = gr.Button(button_text)
                            example_buttons.append((btn, example_text))
        
        # Event binding
        generate_button.click(
            fn=generate_and_visualize,
            inputs=[text_input, seed_input],
            outputs=[result_gif, generation_status, generation_info]
        )
        
        # Example text click events
        for btn, example_text in example_buttons:
            btn.click(
                fn=lambda text=example_text: text,
                outputs=[text_input]
            )
    
    return demo

if __name__ == "__main__":
    # Create and launch Gradio interface
    demo = create_gradio_interface()
    
    # Launch service
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    )
