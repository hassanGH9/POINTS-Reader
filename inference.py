from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2VLImageProcessor
import torch
import sys
import argparse


# We recommend using the following prompt to better performance,
# since it is used throughout the training process.
# Available prompts for selection
prompts = [
    """Please extract all the text from the image with the following requirements:
1. Return tables in HTML format.
2. Return all other text in Markdown format.""",

    """You are an AI assistant specialized in converting PDF images to Markdown format. Please follow these instructions for the conversion:

1. Text Processing:
- Accurately recognize all text content in the PDF image without guessing or inferring.
- Convert the recognized text into Markdown format.
- Maintain the original document structure, including headings, paragraphs, lists, etc.

2. Mathematical Formula Processing:
- Convert all mathematical formulas to LaTeX format.
- Enclose inline formulas with $.
- Enclose block formulas with $$.

3. Table Processing:
- Convert tables to HTML format.
- Wrap the entire table with <table> and </table>.

4. Figure Handling:
- Ignore figures content in the PDF image. Do not attempt to describe or convert images.

5. Output Format:
- Ensure the output Markdown document has a clear structure with appropriate line breaks between elements.
- For complex layouts, try to maintain the original document's structure and format as closely as possible.

Please strictly follow these guidelines to ensure accuracy and consistency in the conversion. Your task is to accurately convert the content of the PDF image into Markdown format without adding any extra explanations or comments."""
]
parser = argparse.ArgumentParser(description='POINTS-Reader inference')
parser.add_argument('image_path', nargs='?', default='examples/6.jpg',
                    help='Path to the image file')
parser.add_argument('--ref_image', type=str,
                    help='Reference image for one-shot learning')
parser.add_argument('--ref_text', type=str,
                    help='Reference text file for one-shot learning')
parser.add_argument('--prompt', type=int, default=0,
                    help='Select prompt by index (0 for basic, 1 for detailed PDF conversion)')

args = parser.parse_args()
image_path = args.image_path

# Select prompt based on --prompt argument
if args.prompt < 0 or args.prompt >= len(prompts):
    print(f"Error: Invalid prompt index {args.prompt}. Available indices: 0-{len(prompts)-1}")
    sys.exit(1)

selected_prompt = prompts[args.prompt]

model_path = 'models/POINTS-Reader'
model = AutoModelForCausalLM.from_pretrained(model_path,
                                             trust_remote_code=True,
                                             torch_dtype=torch.bfloat16,
                                             device_map='cuda')
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
image_processor = Qwen2VLImageProcessor.from_pretrained(model_path)

# Build messages with one-shot example if provided
messages = []

# Add one-shot example if both ref_image and ref_text are provided
if args.ref_image and args.ref_text:
    with open(args.ref_text, 'r', encoding='utf-8') as f:
        ref_text_content = f.read()

    ref_content = [
        dict(type='image', image=args.ref_image),
        dict(type='text', text=selected_prompt)
    ]

    messages.append({
        'role': 'user',
        'content': ref_content
    })
    messages.append({
        'role': 'assistant',
        'content': [dict(type='text', text=ref_text_content)]
    })

# Add the actual query
content = [
    dict(type='image', image=image_path),
    dict(type='text', text=selected_prompt)
]


messages.append({
    'role': 'user',
    'content': content
})
generation_config = {
        'max_new_tokens': 2048,
        'repetition_penalty': 1.05,
        'temperature': 0.7,
        'top_p': 0.8,
        'top_k': 20,
        'do_sample': True
    }
response = model.chat(
    messages,
    tokenizer,
    image_processor,
    generation_config
)
print("=" * 50)
print(response)