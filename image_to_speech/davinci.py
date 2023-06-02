# Tested the word detection on DaVinci as well
# Don't use this file since it's broken
import openai
from PIL import Image

# Set up OpenAI API credentials
openai.api_key = "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"

# Load the image
image = Image.open('C:/WPI/lego/unblurred.jpg')

# Convert the image to bytes
image_bytes = image.tobytes()

# Use the OpenAI API to extract text from the image
response = openai.Completion.create(
    engine="davinci",
    prompt=f"Extract the text from this image: {image_bytes}",
    max_tokens=2048,
    n=1,
    stop=None,
    temperature=0.5,
)

# Extract the text from the response
text = response.choices[0].text.strip()

# Print the extracted text
print(text)
