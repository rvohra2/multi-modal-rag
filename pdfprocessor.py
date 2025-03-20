import os
import shutil
from pathlib import Path
from pdf2image import convert_from_bytes, convert_from_path
import PyPDF2
from io import BytesIO

class PDFProcessor:
    def __init__(self):
        # No initialization logic needed
        pass 
        
    def reset_directory(self, folder_path: str) -> None:
        """
        Clears the specified directory and recreates it.
        
        Args:
            folder_path (str): Path to the directory to clear and recreate.
        """
        print(f"Clearing output folder {folder_path}")
        folder = Path(folder_path)
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)

    def extract_images(self, pdf_id: str, pdf_file: str, max_pages: int = None, page_numbers: list[int] = None) -> list[str]:
        """
        Extracts images from a PDF and saves them to a directory.

        Args:
            pdf_id (str): Unique identifier for the document.
            pdf_file (str): Path to the PDF file.
            max_pages (int, optional): Maximum number of pages to process. Defaults to None.
            page_numbers (list[int], optional): Specific pages to process. Defaults to None.

        Returns:
            list[str]: List of file paths to the saved images.
        """
        try:
            # Define the output folder path
            output_dir = Path(f"pdf_images/{pdf_id}")

            self.reset_directory(output_dir)

            # # Convert PDF pages to images
            # print(f"Extracting images from PDF (ID: {pdf_id}). Max pages: {max_pages}")
            # images = convert_from_path(pdf_file)
            print(f"Extracting images from {pdf_file} to {output_dir}. Max pages: {max_pages}")
            # Convert PDF pages to images
            images = convert_from_path(pdf_file)
            
            # Counter for processed pages
            saved_image_paths = []

            for i, image in enumerate(images):
                if max_pages and len(saved_image_paths) >= max_pages:
                    break

                if page_numbers and i not in page_numbers:
                    continue

                save_path = output_dir / f"image_{i + 1}.png"
                image.save(save_path, "PNG")
                saved_image_paths.append(str(save_path))

            return saved_image_paths
        
        except Exception as e:
            # Log the error for debugging
            raise Exception(f"Error during image extraction: {e}")