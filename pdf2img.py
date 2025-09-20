import fitz
import os
from pathlib import Path


def pdf_to_images(pdf_path: str, output_dir: str = None, dpi: int = 200):
    """Convert PDF pages to JPG images.

    Args:
        pdf_path (str): Path to the PDF file.
        output_dir (str, optional): Output directory. If None, creates a directory
            based on PDF filename.
        dpi (int, optional): Resolution for the output images. Defaults to 200.
    """
    # Open the PDF
    pdf_document = fitz.open(pdf_path)

    # Create output directory if not specified
    if output_dir is None:
        pdf_name = Path(pdf_path).stem
        output_dir = f"examples/{pdf_name}_pdf"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get page count before processing
    page_count = len(pdf_document)

    # Convert each page to image
    for page_num in range(page_count):
        page = pdf_document[page_num]

        # Create a matrix for the desired resolution
        mat = fitz.Matrix(dpi / 72, dpi / 72)

        # Render page to image
        pix = page.get_pixmap(matrix=mat)

        # Save as JPG
        output_path = os.path.join(output_dir, f"{page_num + 1}.jpg")
        pix.save(output_path)
        print(f"Saved page {page_num + 1} to {output_path}")

    pdf_document.close()
    print(f"PDF conversion completed. {page_count} pages saved to {output_dir}")


if __name__ == "__main__":
    pdf_path = "examples/example.pdf"
    pdf_to_images(pdf_path)