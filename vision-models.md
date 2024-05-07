# Vision models · Ollama Blog
![Vision Models](https://ollama.com/public/blog/vision.png)

New LLaVA models
----------------

The LLaVA (Large Language-and-Vision Assistant) model collection has been updated to version `1.6` supporting:

*   **Higher image resolution:** support for up to 4x more pixels, allowing the model to grasp more details.
*   **Improved text recognition and reasoning capabilities:** trained on additional document, chart and diagram data sets.
*   **More permissive licenses:** distributed via the Apache 2.0 license or the LLaMA 2 Community License.

These models are available in three parameter sizes. 7B, 13B and a new 34B model:

*   `ollama run llava:7b`
*   `ollama run llava:13b`
*   `ollama run llava:34b`

Usage
-----

### CLI

![Art](https://ollama.com/public/blog/jmb.jpg)

To use a vision model with `ollama run`, reference `.jpg` or `.png` files using file paths:

```
% ollama run llava "describe this image: ./art.jpg"
The image shows a colorful poster featuring an illustration of a cartoon character with spiky hair. Above the character's head is a crown, suggesting royalty or high status. On the right side of the poster, there is text in French that reads "Jean-Michel Basquiat,"

```


### Python

```
import ollama

res = ollama.chat(
	model="llava",
	messages=[
		{
			'role': 'user',
			'content': 'Describe this image:',
			'images': ['./art.jpg']
		}
	]
)

print(res['message']['content'])

```


### JavaScript

```
import ollama from 'ollama'

const res = await ollama.chat({
	model: 'llava',
	messages: [{
		role: 'user',
		content: 'Describe this image:'
		images: ['./art.jpg']
	}]
})

console.log(res.message.content)

```


> Note: in the Ollama [Python](https://github.com/ollama/ollama-python) and [JavaScript](https://github.com/ollama/ollama-js) libraries and the [REST API](https://github.com/jmorganca/ollama/blob/main/docs/api.md), base64-encoded files can be provided in the `images` parameter. See the full [API docs](https://github.com/jmorganca/ollama/blob/main/docs/api.md) for more examples on providing images to vision models.

Examples
--------

### Object detection

![Object detection](https://ollama.com/public/blog/objectdetection.jpg)

```
% ollama run llava "tell me what do you see in this picture? ./pic.jpg"
In the image, a man wearing blue and white is holding video game controllers in his hands while smiling. His arms are raised above his head as he plays with the virtual reality gaming devices. The man appears to be enjoying himself and immersed in his gaming experience.

```


### Text recognition

![Word Art](https://ollama.com/public/blog/wordart.jpg)

```
% ollama run llava "what does the text say? ./wordart.png"
This is an image with a capitalized, bolded text of "ollama." The background is black and the letters are yellow, which create a contrasting appearance. Although the words "ollama" make up the majority of the image, it could also be seen as part of a sign or advertisement due to its brightness and prominent font style.

```


Thank you LLaVA team
--------------------

The LLaVA 1.6 model family is published by an incredible [team](https://twitter.com/LLaVAAI). Information on how the models were trained is available [on the LLaVA website](https://llava-vl.github.io/blog/2024-01-30-llava-1-6/), as well as benchmark results comparing LLaVA 1.6 to leading open-source and proprietary models.

![Until next time](https://ollama.com/public/blog/nexttime.png)
