from transformers import AutoProcessor, AutoModelForCausalLM
from diffusers import AutoPipelineForInpainting, AutoPipelineForText2Image
from diffusers.utils import load_image, make_image_grid
from PIL import Image, ImageDraw
import torch
import matplotlib.pyplot as plt
import google.generativeai as genai
from prompt_text import get_gemini_response


def initialize_models():
    # Florence model setup
    model_id = 'microsoft/Florence-2-large'
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        trust_remote_code=True, 
        torch_dtype='auto'
    ).to('cuda:0').eval()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    # Setup for inpainting
    pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to("cuda:1")
    pipeline_inpaint = AutoPipelineForInpainting.from_pipe(pipeline_text2image).to("cuda")

    genai.configure(api_key="AIzaSyBCgIkI2WP7JE1Qh5IEqkXcxp9r4wbTH70")
    gemini = genai.GenerativeModel('gemini-1.5-flash')
    
    return model, processor, pipeline_inpaint, gemini


def run_florence(image, task_prompt, text_input=None, model=None, processor=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
        
    inputs = processor(text=prompt, images=image, return_tensors="pt").to('cuda', torch.float16)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"].cuda(),
        pixel_values=inputs["pixel_values"].cuda(),
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )
    
    return parsed_answer

def create_mask_from_polygons(polygons, image_size):
    mask = Image.new('L', image_size, 0)
    draw = ImageDraw.Draw(mask)
    
    if polygons and polygons[0]:
        polygon_points = polygons[0][0]
        
        polygon_tuples = []
        for i in range(0, len(polygon_points), 2):
            if i + 1 < len(polygon_points):
                x = polygon_points[i]
                y = polygon_points[i + 1]
                polygon_tuples.append((x, y))
        
        if len(polygon_tuples) >= 2:
            draw.polygon(polygon_tuples, fill=255)
    
    return mask


def draw_bounding_boxes(image, bboxes, labels=None):
    """
    Draw bounding boxes on the image with optional labels
    """
    # Create a copy of the image to draw on
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    
    # Define colors for different boxes (you can modify these)
    colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF']
    
    # Draw each bounding box
    for idx, bbox in enumerate(bboxes):
        color = colors[idx % len(colors)]  # Cycle through colors
        
        # Draw rectangle
        draw.rectangle(
            [(bbox[0], bbox[1]), (bbox[2], bbox[3])],
            outline=color,
            width=3
        )
        
        # Draw label if provided
        if labels and idx < len(labels):
            draw.text(
                (bbox[0], bbox[1] - 20),
                labels[idx],
                fill=color
            )
    
    return draw_image


def process_image(image_path, task_prompt, segmentation_text=None, inpainting_prompt=None):
    # Initialize models
    # model, processor, pipeline_inpaint, gemini = initialize_models()
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    resize_methods = {
            'NEAREST': Image.Resampling.NEAREST,
            'BOX': Image.Resampling.BOX,
            'BILINEAR': Image.Resampling.BILINEAR,
            'HAMMING': Image.Resampling.HAMMING,
            'BICUBIC': Image.Resampling.BICUBIC,
            'LANCZOS': Image.Resampling.LANCZOS
        }
    image = image.resize((1024,1024), resize_methods.get('BILINEAR', Image.Resampling.BILINEAR))
    
    # Get Florence results based on task
    results = run_florence(image, task_prompt, text_input=segmentation_text, 
                         model=model, processor=processor)
    
    # Handle segmentation and inpainting if needed
    if task_prompt == '<REFERRING_EXPRESSION_SEGMENTATION>' and segmentation_text and inpainting_prompt:
        # Create mask from segmentation results
        polygons = results['<REFERRING_EXPRESSION_SEGMENTATION>']['polygons']
        mask = create_mask_from_polygons(polygons, image.size)
        original_width, original_height = image.size

        negative_prompt = "bad anatomy, deformed, ugly, disfigured"
        # Perform inpainting
        inpainted_image = pipeline_inpaint(
            prompt=inpainting_prompt+'high resolution',
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask,
            strength=0.85,
            guidance_scale=12.5,
            height=original_height,
            width=original_width
        ).images[0]
        
        # Display results
        grid = make_image_grid([image, mask, inpainted_image], rows=1, cols=3)
        plt.figure(figsize=(15, 5))
        plt.imshow(grid)
        plt.axis('off')
        plt.show()
        
        return {
            'florence_results': results,
            'inpainted_image': inpainted_image,
            'mask': mask
        }
    elif task_prompt == '<OD>' or task_prompt == '<CAPTION_TO_PHRASE_GROUNDING>':
        # Get bounding boxes and labels
        bboxes = results[task_prompt]['bboxes']
        labels = results[task_prompt]['labels'] if 'labels' in results[task_prompt] else None
        
        # Draw bounding boxes on image
        annotated_image = draw_bounding_boxes(image, bboxes, labels)
        
        # Display result
        plt.figure(figsize=(15, 10))
        plt.imshow(annotated_image)
        plt.axis('off')
        plt.show()
        
        return {
            'florence_results': results,
            'annotated_image': annotated_image,
            'bboxes': bboxes,
            'labels': labels
        }
    

    else:
        # For other tasks, just return Florence results
        return {'florence_results': results[task_prompt]}

# Example usage
if __name__ == "__main__":
    model, processor, pipeline_inpaint, gemini = initialize_models()

    image_path = "./images/dog.png"
    prompt = ["""
          you are about to solve a semantic question, and you need to classify the following question to a certain task type
          \n
          there are three different task categories: image caption, object detection, and image segmentation.
          \n
          if you think the question is referred to image captioning, your response should be '<DETAILED_CAPTION>' without quotation marks;
          \n
          and if you think the question is referred to object detection, your response should be '<CAPTION_TO_PHRASE_GROUNDING>' without quotation marks;
          \n
          and if you think the question is referred to image segmentation, your response should be '<REFERRING_EXPRESSION_SEGMENTATION>' without quotation marks.
          \n
          for example, if the question is 'describe what is the man doing' or 'what color is the car', you response should be '<DETAILED_CAPTION>' without quotation marks.
          \n
          and if the question is 'locate the man', you response should be '<CAPTION_TO_PHRASE_GROUNDING>' without quotation marks.
          \n
          here is the question:
          """]

    question = 'inpaint the dog on the bench with a purple capybara'
    input = get_gemini_response(model=gemini, prompt=prompt, question=question)
    print(input)


    # Example 1: Segmentation with inpainting
    results = process_image(
        image_path=image_path,
        task_prompt=input[0],
        segmentation_text=input[1],
        inpainting_prompt=input[2]
    )
    if 'inpainted_image' in results:
        results['inpainted_image'].save("inpainted_result.png")
        results['mask'].save("generated_mask.png")
    elif 'annotated_image' in results:
        results['annotated_image'].save("annotated_image.png")
    else:
        print(results['florence_results'])

    
    # # Example 2: Caption
    # results2 = process_image(
    #     image_path=image_path,
    #     task_prompt='<DETAILED_CAPTION>'
    # )
    # print("Caption results:", results2['florence_results'])
    
    # # # Example 3: Object Detection
    # results3 = process_image(
    #     image_path=image_path,
    #     task_prompt='<OD>'
    # )
    # print("Object Detection results:", results3['florence_results'])
    
    # # # Example 4: Dense Region Caption
    # results4 = process_image(
    #     image_path=image_path,
    #     task_prompt='<DENSE_REGION_CAPTION>'
    # )
    # print("Dense Region Caption results:", results4['florence_results'])
    