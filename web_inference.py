
from typing import List
import requests
import json
import base64


def encode_image_to_base64(image_path: str) -> str:
    """Convert an image file to base64 encoding.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Base64 encoded image data URL.
    """
    with open(image_path, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/jpeg;base64,{encoded_string}"


def call_wepoints(messages: List[dict],
                 temperature: float = 0.7,
                 max_new_tokens: int = 2048,
                 repetition_penalty: float = 1.05,
                 top_p: float = 0.8,
                 top_k: int = 20,
                 do_sample: bool = True,
                 url: str = 'http://127.0.0.1:8081/v1/chat/completions') -> str:
    """Query WePOINTS model to generate a response.

    Args:
        messages (List[dict]): A list of messages to be sent to WePOINTS. The
            messages should be the standard OpenAI messages, like:
            [
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'text',
                            'text': 'Please describe this image in short'
                        },
                        {
                            'type': 'image_url',
                            'image_url': {'url': 'data:image/jpeg;base64,<base64_encoded_image>'}
                        }
                    ]
                }
            ]
        temperature (float, optional): The temperature of the model.
            Defaults to 0.0.
        max_new_tokens (int, optional): The maximum number of new tokens to generate.
            Defaults to 2048.
        repetition_penalty (float, optional): The penalty for repetition.
            Defaults to 1.05.
        top_p (float, optional): The top-p probability threshold.
            Defaults to 0.8.
        top_k (int, optional): The top-k sampling vocabulary size.
            Defaults to 20.
        do_sample (bool, optional): Whether to use sampling or greedy decoding.
            Defaults to True.
        url (str, optional): The URL of the WePOINTS model.
            Defaults to 'http://127.0.0.1:8081/v1/chat/completions'.

    Returns:
        str: The generated response from WePOINTS.
    """
    data = {
        'model': 'POINTS-Reader',
        'messages': messages,
        'max_new_tokens': max_new_tokens,
        'temperature': temperature,
        'repetition_penalty': repetition_penalty,
        'top_p': top_p,
        'top_k': top_k,
        'do_sample': do_sample,
    }
    response = requests.post(url,
                             json=data)
    response = json.loads(response.text)
    response = response['choices'][0]['message']['content']
    return response

prompt = """Please extract all the text from the image with the following requirements:
1. Return tables in HTML format.
2. Return all other text in Markdown format."""

image_path = 'examples/6.jpg'
messages = [{
              'role': 'user',
              'content': [
                  {
                      'type': 'text',
                      'text': prompt
                  },
                  {
                      'type': 'image_url',
                      'image_url': {'url': encode_image_to_base64(image_path)}
                  }
              ]
            }]
response = call_wepoints(messages)
print(response)
