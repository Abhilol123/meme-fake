import gradio as gr
from PIL import Image  
import re
import io
import os
import requests
import base64
from io import BytesIO
from Inference import inference


def infer(character: str):
    images = inference.inference(character)
    result_images = []
    for image in images:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        image_b64 = (f"data:image/jpeg;base64,{img_str}")
        result_images.append(image_b64)
    return result_images

def get_sample_images():
    images = []
    for (dirpath, dirnames, filenames) in os.walk("sample_images"):
        for filename in filenames: 
            path = dirpath + "/" + filename
            images.append(Image.open(path))
    return images
    
    
def get_characters():
    characters = []
    for (dirpath, dirnames, filenames) in os.walk("lora"):
        for filename in filenames: 
            characters.append(filename.split(".")[0])
    return characters      

def mirror(x):
    return x

css = """
        .gradio-container {
            font-family: 'IBM Plex Sans', sans-serif;
        }
        .gr-button {
            color: white;
            border-color: black;
            background: black;
        }
        input[type='range'] {
            accent-color: black;
        }
        .dark input[type='range'] {
            accent-color: #dfdfdf;
        }
        .container {
            max-width: 730px;
            margin: auto;
            padding-top: 1.5rem;
        }
        #gallery {
            min-height: 22rem;
            margin-bottom: 15px;
            margin-left: auto;
            margin-right: auto;
            border-bottom-right-radius: .5rem !important;
            border-bottom-left-radius: .5rem !important;
        }
        #gallery>div>.h-full {
            min-height: 20rem;
        }
        .details:hover {
            text-decoration: underline;
        }
        .gr-button {
            white-space: nowrap;
        }
        .gr-button:focus {
            border-color: rgb(147 197 253 / var(--tw-border-opacity));
            outline: none;
            box-shadow: var(--tw-ring-offset-shadow), var(--tw-ring-shadow), var(--tw-shadow, 0 0 #0000);
            --tw-border-opacity: 1;
            --tw-ring-offset-shadow: var(--tw-ring-inset) 0 0 0 var(--tw-ring-offset-width) var(--tw-ring-offset-color);
            --tw-ring-shadow: var(--tw-ring-inset) 0 0 0 calc(3px var(--tw-ring-offset-width)) var(--tw-ring-color);
            --tw-ring-color: rgb(191 219 254 / var(--tw-ring-opacity));
            --tw-ring-opacity: .5;
        }
        #advanced-btn {
            font-size: .7rem !important;
            line-height: 19px;
            margin-top: 12px;
            margin-bottom: 12px;
            padding: 2px 8px;
            border-radius: 14px !important;
        }
        #advanced-options {
            display: none;
            margin-bottom: 20px;
        }
        .footer {
            margin-bottom: 45px;
            margin-top: 35px;
            text-align: center;
            border-bottom: 1px solid #e5e5e5;
        }
        .footer>p {
            font-size: .8rem;
            display: inline-block;
            padding: 0 10px;
            transform: translateY(10px);
            background: white;
        }
        .dark .footer {
            border-color: #303030;
        }
        .dark .footer>p {
            background: #0b0f19;
        }
        .acknowledgments h4{
            margin: 1.25em 0 .25em 0;
            font-weight: bold;
            font-size: 115%;
        }
        #container-advanced-btns{
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: center;
        }
        .animate-spin {
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            from {
                transform: rotate(0deg);
            }
            to {
                transform: rotate(360deg);
            }
        }
        #share-btn-container {
            display: flex; padding-left: 0.5rem !important; padding-right: 0.5rem !important; background-color: #000000; justify-content: center; align-items: center; border-radius: 9999px !important; width: 13rem;
        }
        #share-btn {
            all: initial; color: #ffffff;font-weight: 600; cursor:pointer; font-family: 'IBM Plex Sans', sans-serif; margin-left: 0.5rem !important; padding-top: 0.25rem !important; padding-bottom: 0.25rem !important;
        }
        #share-btn * {
            all: unset;
        }
        .gr-form{
            flex: 1 1 50%; border-top-right-radius: 0; border-bottom-right-radius: 0;
        }
        #prompt-container{
            gap: 0;
        }
        #share-btn-container div:nth-child(-n+2){
        width: auto !important;
        min-height: 0px !important;
        } 
"""

block = gr.Blocks(css=css)

examples = [
    [
        'Anmol'
    ],
    [
        'Abhinav'
    ],
    [
        'Billy',
    ],
    [
        'Zaryab',
    ]
]


with block:
    gr.HTML(
        """
            <div style="text-align: center; max-width: 650px; margin: 0 auto; padding-top: 7px;">
              <div
                style="
                  display: inline-flex;
                  align-items: center;
                  gap: 0.8rem;
                  font-size: 1.75rem;
                "
              >
                <h1 style="font-weight: 900; margin-bottom: 7px;">
                  Meme Fake Demo
                </h1>
              </div>
              <p style="text-align: center;"">
              This is a demo of a face swapping (deepfaking) platform which is used to get memes with
              the faces swapped for the character who is chosen.
              </p>
            </div>
        """
    )
    with gr.Group():
        with gr.Box():
            with gr.Row(elem_id="prompt-container").style(mobile_collapse=False, equal_height=True):
                options=get_characters()
                text = gr.Dropdown(
                    options,
                    label="Character",
                    show_label=True,
                    elem_id="character-input",
                ).style(
                    border=(True, False, True, True),
                    rounded=(True, False, False, True),
                    container=False,
                )
                btn = gr.Button("Generate images").style(
                    margin=False,
                    rounded=(False, True, True, False),
                    full_width=False,
                )

        gallery = gr.Gallery(
            label="Generated images",
            show_label=False, 
            elem_id="gallery"
        ).style(grid=[2], height="auto")

        
        gr.Gallery(value=get_sample_images).style(grid=4)
        
        btn.click(infer, inputs=text, outputs=[gallery], postprocess=False)
        
        
#         share_button.click(
#             None,
#             [],
#             [],
#             _js=share_js,
#         )

block.queue(concurrency_count=40, max_size=20).launch(max_threads=150, server_name="0.0.0.0")