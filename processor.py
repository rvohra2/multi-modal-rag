import torch
from transformers import BitsAndBytesConfig
from colpali_engine.models import ColPali
from colpali_engine.models.paligemma.colpali.processing_colpali import ColPaliProcessor
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor
from colpali_engine.utils.torch_utils import ListDataset, get_torch_device
from typing import List, cast
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

class Processor:
    def __init__(self, device: str = None, model_name: str = "vidore/colpali-v1.2"):
        """
        Initializes the ColpaliManager with the specified model and device.

        Args:
            device (str, optional): Device to run the model on (e.g., "cuda" or "cpu"). Defaults to None.
            model_name (str, optional): Name of the pre-trained model. Defaults to "vidore/colpali-v1.2".
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,  # Load the model in 4-bit precision
                    bnb_4bit_use_double_quant=True,  # Use double quantization for better memory efficiency
                    bnb_4bit_quant_type="nf4",  # Use NF4 quantization type
                    bnb_4bit_compute_dtype=torch.float16  # Use float16 for computation
                )
        self.model = ColPali.from_pretrained(
            model_name,
            quantization_config=None if self.device.type == "cpu" else bnb_config,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            offload_folder="offload"
        ).eval()

        self.processor = cast(ColPaliProcessor, ColPaliProcessor.from_pretrained(model_name))

        print(f"Initializing ColpaliManager with device {self.device} and model {model_name}")


    def load_images(self, image_paths: List[str]) -> List[Image.Image]:
        """
        Loads images from the specified paths.

        Args:
            image_paths (List[str]): List of paths to the images.

        Returns:
            List[Image.Image]: List of PIL Image objects.
        """
        return [Image.open(path) for path in image_paths]

    def generate_image_embeddings(self, image_paths: List[str], batch_size: int = 1) -> List:
        """
        Generates embeddings for the given images.

        Args:
            image_paths (List[str]): List of paths to the images.
            batch_size (int, optional): Batch size for processing. Defaults to 1.

        Returns:
            List[torch.Tensor]: List of image embeddings as NumPy arrays.
        """
        print(f"Processing {len(image_paths)} image_paths")
        
        images = self.load_images(image_paths)
        
        dataloader = DataLoader(
            dataset=images,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda x: self.processor.process_images(x),
            # num_workers=4,  # Adjust based on your CPU cores
            # pin_memory=True
        )

        embeddings = []
        for batch in tqdm(dataloader, desc="Processing Images"):
            with torch.no_grad():
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                output = self.model(**batch)
            embeddings.extend(output.unbind())

        return [embed.float().cpu().numpy() for embed in embeddings]

    
    def generate_text_embeddings(self, texts: List[str], batch_size: int = 1) -> List[torch.Tensor]:
        """
        Generates embeddings for the given texts.

        Args:
            texts (List[str]): List of text inputs.

        Returns:
            List[torch.Tensor]: List of text embeddings as NumPy arrays.
        """
        print(f"Processing {len(texts)} texts")

        dataloader = DataLoader(
            dataset=texts,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda x: self.processor.process_queries(x),
        )

        embeddings = []
        for batch in dataloader:
            with torch.no_grad():
                batch = {k: v.to(self.device) for k, v in batch.items()}
                output = self.model(**batch)
            embeddings.extend(output.unbind())

        return [embed.float().cpu().numpy() for embed in embeddings]