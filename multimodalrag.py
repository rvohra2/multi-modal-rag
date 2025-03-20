import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import google.generativeai as genai
import logging
import os

class MultiModalRAG:
    def __init__(self):
        """ Initialize models and processors only once to optimize efficiency. """
        # self.qwen_model, self.qwen_processor = self._load_qwen_model()
        self.gemini_model = self._initialize_gemini()

    def _load_qwen_model(self):
        """ Loads the Qwen2VL model and processor with optimized settings. """
        try:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )

            model = Qwen2VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-VL-2B-Instruct",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map="cpu",
                torch_dtype=torch.float16,
                offload_folder="offload"
            ).eval()

            processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen2-VL-2B-Instruct",
                trust_remote_code=False
            )

            logging.info("Qwen2VL model successfully loaded.")
            return model, processor

        except Exception as e:
            logging.error(f"Error loading Qwen2VL model: {e}")
            return None, None
        
    def _initialize_gemini(self):
        """ Initializes the Gemini API client. """
        try:
            genai.configure(api_key=os.environ['GEMINI_API_KEY'])
            return genai.GenerativeModel('gemini-1.5-flash')
        except Exception as e:
            logging.error(f"Error initializing Gemini API: {e}")
            return None

    def query_qwen2vl(self, query: str, image_paths: list[str]) -> str:
        """ Queries Qwen2VL model with an image and text prompt. """
        if not self.qwen_model or not self.qwen_processor:
            return "Error: Qwen2VL model not initialized."

        torch.cuda.empty_cache()
        logging.info(f"Querying Qwen2VL with: {query}")

        try:
            images = [Image.open(path).resize((512, 512)) for path in image_paths]

            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": images[0]},  # Only using the first image
                    {"type": "text", "text": query}
                ]}
            ]

            text_input = self.qwen_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            image_inputs, _ = process_vision_info(messages)

            inputs = self.qwen_processor(
                text=[text_input],
                images=image_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.qwen_model.device)

            with torch.no_grad():
                generated_ids = self.qwen_model.generate(**inputs, max_new_tokens=15)

            response_text = self.qwen_processor.batch_decode(
                [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)],
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]

            logging.info(f"Qwen2VL response: {response_text}")
            return response_text

        except Exception as e:
            logging.error(f"Error in Qwen2VL query: {e}")
            return f"Error: {str(e)}"

    def query_gemini(self, query: str, image_paths: list[str]) -> str:
        """
        Queries the Gemini model with a text query and a list of image paths.

        Args:
            query (str): The text query to ask the model.
            image_paths (List[str]): List of paths to the images.

        Returns:
            str: The model's response to the query.
        """
        if not self.gemini_model:
            return "Error: Gemini model not initialized."

        logging.info(f"Querying Gemini with: {query}")

        try:
            images = [Image.open(path) for path in image_paths]
            chat_session = self.gemini_model.start_chat()
            response = chat_session.send_message([*images, query])
            
            answer = response.text
            logging.info(f"Gemini response: {answer}")
            return answer

        except Exception as e:
            logging.error(f"Error in Gemini query: {e}")
            return f"Error: {str(e)}"
