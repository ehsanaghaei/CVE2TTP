import openai

from lib.functions import read_txt
KEY = read_txt(r"./configs/openai_key.key")

openai.api_key = "<YOUR_KEY>"

prompt = "Hello, how are you today?"
model_engine = "ada"

# Generate a response
completion = openai.Completion.create(
    engine=model_engine,
    prompt=prompt,
    max_tokens=1024,
    n=1,
    stop=None,
    temperature=0.5,
)

response = completion.choices[0].text
print(response)
