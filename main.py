from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
import fitz  # PyMuPDF
import os
import tempfile
import shutil
import logging
import time
from PIL import Image
import io

app = FastAPI()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def compress_image(input_path: str, output_path: str, target_size: int, quality_steps: list[int] = [95, 75, 50, 30]) -> bool:
    """Compress an image with iterative quality reduction to reach target size."""
    try:
        img = Image.open(input_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        temp_output = tempfile.mktemp(suffix='.jpg')
        tolerance = target_size * 0.1  # ±10% tolerance
        target_min = target_size - tolerance
        target_max = target_size + tolerance

        for quality in quality_steps:
            img.save(temp_output, format='JPEG', quality=quality, optimize=True)
            compressed_size = await get_file_size(temp_output)
            logger.info(f"Image size after compression (quality={quality}): {compressed_size / 1024 / 1024:.2f} MB")
            if target_min <= compressed_size <= target_max:
                shutil.move(temp_output, output_path)
                return True

        # If quality reduction isn't enough, try resizing
        scale_factor = 0.9  # Reduce dimensions by 10% each iteration
        for _ in range(3):  # Try up to 3 resizing iterations
            compressed_size = await get_file_size(temp_output)  # Check size from last quality attempt
            if compressed_size <= target_max:
                break
            width, height = img.size
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            for quality in quality_steps:
                img.save(temp_output, format='JPEG', quality=quality, optimize=True)
                compressed_size = await get_file_size(temp_output)
                logger.info(f"Image size after resizing and compression (quality={quality}, scale={scale_factor}): {compressed_size / 1024 / 1024:.2f} MB")
                if target_min <= compressed_size <= target_max:
                    shutil.move(temp_output, output_path)
                    return True
            scale_factor *= 0.9  # Reduce further if needed

        # If all attempts fail, use the last attempt
        if os.path.exists(temp_output):
            shutil.move(temp_output, output_path)
        return True

    except Exception as e:
        logger.error(f"Error compressing image: {str(e)}")
        return False

async def compress_images_in_pdf(input_path: str, output_path: str, target_size: int, quality_steps: list[int] = [95, 75, 50, 30]) -> bool:
    """Compress images in PDF while preserving text, with iterative quality reduction."""
    temp_output = tempfile.mktemp(suffix='.pdf')
    tolerance = target_size * 0.1
    target_min = target_size - tolerance
    target_max = target_size + tolerance
    
    for quality in quality_steps:
        try:
            pdf = fitz.open(input_path)
            output_pdf = fitz.open()

            for page_num in range(len(pdf)):
                try:
                    page = pdf[page_num]
                    new_page = output_pdf.new_page(width=page.rect.width, height=page.rect.height)

                    image_list = page.get_images(full=True)
                    for img_index, img in enumerate(image_list):
                        xref = img[0]
                        try:
                            base_image = pdf.extract_image(xref)
                            image_bytes = base_image["image"]

                            img = Image.open(io.BytesIO(image_bytes))
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                            output = io.BytesIO()
                            img.save(output, format='JPEG', quality=quality, optimize=True)
                            img_data = output.getvalue()

                            rect = fitz.Rect(0, 0, img.width, img.height)
                            new_page.insert_image(rect, stream=img_data)
                        except Exception as e:
                            logger.warning(f"Skipping image on page {page_num}, index {img_index}: {str(e)}")
                            blank_img = Image.new('RGB', (50, 50), color='white')
                            output = io.BytesIO()
                            blank_img.save(output, format='JPEG', quality=quality)
                            blank_img_data = output.getvalue()
                            new_page.insert_image(fitz.Rect(0, 0, 50, 50), stream=blank_img_data)

                    try:
                        new_page.show_pdf_page(new_page.rect, pdf, page_num)
                    except Exception as e:
                        logger.warning(f"Failed to copy text on page {page_num}: {str(e)}")
                        new_page.insert_text((50, 50), "Content skipped due to error", fontsize=12)

                except Exception as e:
                    logger.warning(f"Skipping page {page_num} due to error: {str(e)}")
                    new_page = output_pdf.new_page(width=595, height=842)
                    new_page.insert_text((50, 50), f"Page {page_num} skipped due to error", fontsize=12)

            output_pdf.save(temp_output, deflate=True, garbage=4, clean=True, linear=True)
            pdf.close()
            output_pdf.close()

            compressed_size = await get_file_size(temp_output)
            logger.info(f"PDF size after text-preserving compression (quality={quality}): {compressed_size / 1024 / 1024:.2f} MB")
            if target_min <= compressed_size <= target_max:
                shutil.move(temp_output, output_path)
                return True

        except Exception as e:
            logger.error(f"Error compressing images in PDF at quality={quality}: {str(e)}")
            return False

    if os.path.exists(temp_output):
        shutil.move(temp_output, output_path)
    return False

async def convert_pages_to_images(input_path: str, temp_dir: str, output_path: str, target_size: int, quality: int = 95, dpi: int = 700, page_count: int = 1) -> list[str]:
    """Convert each PDF page to a JPEG image (fallback), adjust based on target size."""
    try:
        quality_steps = [quality, max(60, quality - 20), max(30, quality - 40)]
        dpi_steps = [dpi, max(600, dpi - 300), max(300, dpi - 600)]
        if page_count > 15:
            quality_steps = [max(60, q - (page_count // 10) * 10) for q in quality_steps]
            dpi_steps = [max(300, d - (page_count // 20) * 100) for d in dpi_steps]

        tolerance = target_size * 0.1
        target_min = target_size - tolerance
        target_max = target_size + tolerance

        pdf = fitz.open(input_path)
        image_paths = []

        for quality, dpi in zip(quality_steps, dpi_steps):
            logger.info(f"Trying quality={quality}, dpi={dpi} for {page_count} pages")
            image_paths = []
            for page_num in range(len(pdf)):
                try:
                    page = pdf[page_num]
                    pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    
                    temp_image_path = os.path.join(temp_dir, f"page_{page_num}.jpg")
                    img.save(temp_image_path, format='JPEG', quality=quality, optimize=True)
                    image_paths.append(temp_image_path)
                except Exception as e:
                    logger.warning(f"Skipping page {page_num} during image conversion: {str(e)}")

            # Check size after converting all pages
            temp_pdf = tempfile.mktemp(suffix='.pdf')
            if await create_pdf_from_images(image_paths, temp_pdf):
                compressed_size = await get_file_size(temp_pdf)
                logger.info(f"PDF size after image-based compression attempt: {compressed_size / 1024 / 1024:.2f} MB")
                if target_min <= compressed_size <= target_max:
                    pdf.close()
                    shutil.move(temp_pdf, output_path)
                    return image_paths
        # safe_unlink(temp_pdf)

        pdf.close()
        return image_paths  # Return the last attempt's image paths
    except Exception as e:
        logger.error(f"Error converting pages to images: {str(e)}")
        return []

async def create_pdf_from_images(image_paths: list[str], output_path: str) -> bool:
    """Merge images in sequence to create a PDF (fallback)."""
    try:
        pdf = fitz.open()
        for image_path in image_paths:
            try:
                img = Image.open(image_path)
                img_pdf = fitz.open()
                rect = fitz.Rect(0, 0, img.width, img.height)
                page = img_pdf.new_page(width=img.width, height=img.height)
                page.insert_image(rect, filename=image_path)
                pdf.insert_pdf(img_pdf)
                img_pdf.close()
                img.close()
            except Exception as e:
                logger.warning(f"Skipping image {image_path} during PDF creation: {str(e)}")
                page = pdf.new_page(width=595, height=842)
                page.insert_text((50, 50), "Image skipped due to error", fontsize=12)

        if len(pdf) == 0:
            page = pdf.new_page(width=595, height=842)
            page.insert_text((50, 50), "All content skipped due to errors", fontsize=12)

        pdf.save(output_path, deflate=True, garbage=4, clean=True, linear=True)
        pdf.close()
        return True
    except Exception as e:
        logger.error(f"Error creating PDF from images: {str(e)}")
        return False

async def get_file_size(file_path: str) -> int:
    """Get file size in bytes."""
    return os.path.getsize(file_path)

def safe_cleanup(file_path: str, max_attempts: int = 3, delay: float = 0.5):
    """Safely delete a file or directory with retry mechanism."""
    for attempt in range(max_attempts):
        try:
            if os.path.exists(file_path):
                if os.path.isdir(file_path):
                    shutil.rmtree(file_path, ignore_errors=True)
                else:
                    os.unlink(file_path)
            return
        except PermissionError as e:
            logger.warning(f"Attempt {attempt + 1}/{max_attempts} to delete {file_path} failed: {str(e)}")
            if attempt < max_attempts - 1:
                time.sleep(delay)
        except Exception as e:
            logger.error(f"Error deleting {file_path}: {str(e)}")
            raise
    logger.error(f"Failed to delete {file_path} after {max_attempts} attempts")

def cleanup_files(file_paths: list[str], background_tasks: BackgroundTasks):
    """Schedule file cleanup using BackgroundTasks."""
    for file_path in file_paths:
        background_tasks.add_task(safe_cleanup, file_path)

@app.post("/compress-file/")
async def compress_file_endpoint(file: UploadFile, background_tasks: BackgroundTasks, quality: int = 95):
    """Endpoint to upload and compress a PDF or image file to ~50% of its original size."""
    # Determine file type
    filename_lower = file.filename.lower()
    is_pdf = filename_lower.endswith('.pdf')
    is_image = filename_lower.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'))

    if not (is_pdf or is_image):
        raise HTTPException(status_code=400, detail="Only PDF, JPEG, PNG, BMP, GIF, and TIFF files are allowed")

    if quality < 1 or quality > 100:
        raise HTTPException(status_code=400, detail="Quality must be between 1 and 100")

    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.tmp')
    input_path = temp_input.name
    output_suffix = '.pdf' if is_pdf else '.jpg'
    output_path = tempfile.mktemp(suffix=output_suffix)
    temp_dir = tempfile.mkdtemp()

    try:
        # Write uploaded content
        content = await file.read()
        temp_input.write(content)
        temp_input.flush()
        temp_input.close()

        initial_size = await get_file_size(input_path)
        logger.info(f"Initial file size: {initial_size / 1024 / 1024:.2f} MB")

        # Calculate dynamic target size (~50% of original size)
        target_size = int(initial_size * 0.5)
        logger.info(f"Target size (50% of original): {target_size / 1024 / 1024:.2f} MB")

        success = False
        compressed_size = 0

        if is_pdf:
            # Check page count for PDFs
            pdf = fitz.open(input_path)
            page_count = len(pdf)
            pdf.close()
            logger.info(f"PDF has {page_count} pages")

            # Step 1: Try preserving text and compressing images
            logger.info("Attempting PDF compression while preserving text")
            quality_steps = [min(quality, q) for q in [95, 75, 50, 30]]
            success = await compress_images_in_pdf(input_path, output_path, target_size, quality_steps=quality_steps)
            if success:
                compressed_size = await get_file_size(output_path)
                logger.info(f"Final size after text-preserving compression: {compressed_size / 1024 / 1024:.2f} MB")

            # Step 2: If text-preserving fails or size exceeds target, fall back to image-based approach
            if not success or compressed_size > target_size * 1.1:  # Allow 10% tolerance
                if not success:
                    logger.info("Text-preserving compression failed, falling back to image-based compression")
                else:
                    logger.info("Size exceeds target, falling back to image-based compression")
                
                temp_output = tempfile.mktemp(suffix='.pdf')
                image_paths = await convert_pages_to_images(input_path, temp_dir, temp_output, target_size, quality=quality, dpi=700, page_count=page_count)
                if not image_paths:
                    raise HTTPException(status_code=500, detail="Failed to convert PDF pages to images")

                success = os.path.exists(temp_output)  # Check if temp_output was created successfully
                if not success:
                    success = await create_pdf_from_images(image_paths, temp_output)
                    if not success:
                        cleanup_files(image_paths, background_tasks)
                        raise HTTPException(status_code=500, detail="Failed to create compressed PDF")

                compressed_size = await get_file_size(temp_output)
                logger.info(f"Size after image-based compression: {compressed_size / 1024 / 1024:.2f} MB")
                if compressed_size > target_size * 1.1:
                    cleanup_files(image_paths, background_tasks)
                    logger.warning(f"Could not compress PDF to within 10% of target size ({target_size / 1024 / 1024:.2f} MB)")

                shutil.move(temp_output, output_path)
                cleanup_files(image_paths, background_tasks)

        else:  # Image file
            logger.info("Compressing image file")
            quality_steps = [min(quality, q) for q in [95, 75, 50, 30]]
            success = await compress_image(input_path, output_path, target_size, quality_steps=quality_steps)
            if not success:
                raise HTTPException(status_code=500, detail="Failed to compress image")

            compressed_size = await get_file_size(output_path)
            logger.info(f"Final compressed image size: {compressed_size / 1024 / 1024:.2f} MB")
            if compressed_size > target_size * 1.1:
                logger.warning(f"Could not compress image to within 10% of target size ({target_size / 1024 / 1024:.2f} MB)")

        # Schedule cleanup
        cleanup_files([input_path, output_path, temp_dir], background_tasks)

        # Determine response media type and filename
        media_type = 'application/pdf' if is_pdf else 'image/jpeg'
        output_filename = "compressed_" + file.filename.rsplit('.', 1)[0] + (".pdf" if is_pdf else ".jpg")

        return FileResponse(
            path=output_path,
            filename=output_filename,
            media_type=media_type
        )

    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        cleanup_files([input_path, output_path, temp_dir], background_tasks)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "File Compression API is running. Use /compress-file/ to upload and compress a PDF or image."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)