from diffusers import StableDiffusionInpaintPipeline
from lora_diffusion import patch_pipe, tune_lora_scale
from PIL import Image
import torch
import os


class Inference:
    def __init__(self) -> None:
        self.data = {
            " 00003.jpg": "a man holding up money and saying shut up and take my money, in the john wick movie, with futuristic gear and helmet, futurama, with a beard and a black shirt, backed on kickstarter, mirrorless camera, light brown hair blue eyes, instagram filter, detailed characters, reddit",
            " 00005.jpg": "a man in a blue suit sitting at a table, brutal fight, trending on teemill, an escalating violent firefight, threads, whoa, wearing shorts and t shirt, precisionism, funniest meme ever, perfect coherence, show-accurate, with text",
            " 00006.jpg": "a man that is standing in front of a fire, destroying houses, smirking at the camera, internet art, actual photo, taken in the late 2010s, internet meme, southern gothic art, storyboard, art direction, slay, winning photograph, devious, trollface, high - resolution photograph, this is fine, life. america 2 0 9 8",
            " 00008.jpg": "a man sitting in a chair next to a bottle of beer, internet meme, explorer, chrome reflect, witty, sneer, someone lost job, foil, bold black lines, high - rated, lit windows, expert, thumb up, safari, an elderly, helpful, bronze, speed grapher, in danger, windows",
            " 00010.jpg": "a man with a cigarette in his mouth saying sleep on your couch makes breakfast, internet meme, muscular bald man, aww, hero, very grainy image, subject is smiling, wearing a white gi, industrial saliva ooze, sausage, with text, with a bunk bed, hashing, food, creative commons attribution, trollface, fine dining, meme",
            " 00011.jpg": "a close up of a person wearing a hat, “the ultimate gigachad, wearing pajamas, nas, 2005 blog, no watermark, crystal ruff, punchable expression, canada, dressed as a scavenger, bad boy, classified, no cropping, pictured from the shoulders up, eeire mood, mobeus",
            " 00012.jpg": "a man holding a sign with a drawing of a cat on it, in oval office, exploitable image, praised, corrected, perfect animal, inspired by Dong Yuan, childrens book, kitt, leaking, say, cool hair, edited, sfw, hilarious, kind, 1 st winner, signature, meow",
            " 00013.jpg": "a man with long hair talking on a cell phone, mordor as a bustling city, funniest meme ever, walking to the right, pulp, sly, with a white background, simplified, the ring is horizontal, narrow footpath, one panel, slightly turned to the right, modular",
            " 00014.png": "a man in a fur coat holding a microphone, boromir in an 80\'s anime world, climate change, smirking deviously, 9gag, ux, tn, warning, maintenance photo, seasonal, wearing latex, change, new jersey, people are panicking, complaints, yelling furiously",
            " 00015.jpg": "a picture of a guy with a caption that reads friend posts his post the web does, trollface, by Adam Szentpétery, funniest meme ever, brian miller, loss, rule of thrids, in pain, blonde guy, ( ( dr sues ) ), 9gag, not cropped, tilted frame, sitting on edge of bed, webs, he is holding a smartphone",
            " 00016.jpg": "a man standing in an office with a cup of coffee, goddamn! plus, language, heavily upvoted, basil, ball, hi - rez, i think, the funniest meme ever, bass, bad photo, corrected, facepalm, helvetica, welcoming attitude, ballistic, callouts, english text, grandma",
            "_00020.jpg": "someone tells me their name me after 3 seconds i am once again asking for your name, reddit meme, bernie, by Sargent Johnson, real trending on instagram image, iso-250, hexagonal, iso: 200, most popular, speech, hugh quality, iso : 2 0 0",
            "_00021.jpg": "a man in a green shirt holding a bird, ifunny impact font bottom text, male physician, frodo, negative self-talk, secret <, facepalm, beautiful plans, drinking and smoking, with a creepy secret temple, loosely cropped, elves, six-pack, food focus, cure, obese",
            "_00024.jpg": "a man sitting at a desk in front of a computer, crying one single tear, clockface, accurate image, michael_jackson, coworkers, title - shift, :fire: :sunglasses: :joystick: :eyes: :2, bronze, people at night, rips, relatable, nine-dimensional, alarm clock, devastated",
            "_00026.jpg": "a man with a surprised look on her face, the computer gods ascend, versatile, aquarius, relatable, joke, dystopian bad vibes, perfect!!!, mouth half open, screengrab, inspired by Elsie Few, witty, traders, man is sitting, incredibly realistic, comedic, virus, endless loop, people at work",
            "_00027.jpg": "a man in a tuxedo holding a mouse, meme, catscatscats, inspired by Charles Fremont Conner, amused, real-life accurate, no humans, yah, template, microscopic picture, gamer",
            "_00028.jpg": "a picture of a man , dark forest in background, nasus, unsanitary, awkward and anxious, huge surprised eyes, sideways glance",
        }
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16
        ).to("cuda")
        self.pipe.enable_xformers_memory_efficient_attention()

        def dummy(images, **kwargs):
            return images, False
        self.pipe.safety_checker = dummy
        return None

    def inference(self, character_name) -> list:
        os.makedirs(f"./output3/{character_name}/", exist_ok=True)
        # Load lora
        patch_pipe(self.pipe, f"./lora/{character_name}.safetensors")
        tune_lora_scale(self.pipe.unet, 0.6)
        images_generated = []
        for file_name in self.data:
            image_input = Image.open(
                f"input images/{file_name}").convert('RGB').resize((512, 512))
            mask_image = Image.open(
                f"mask images/{file_name}").convert('RGB').resize((512, 512))
            image = self.pipe(
                prompt=f"<s1><s2> {self.data[file_name]}",
                image=image_input,
                mask_image=mask_image,
                negative_prompt="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
            ).images[0]
            images_generated.append(image)
        return images_generated
