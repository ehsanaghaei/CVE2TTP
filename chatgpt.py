import openai

from lib.functions import read_txt
KEY = read_txt(r"/media/ea/SSD2/Projects/CVE2TTP/configs/openai_key.key")

openai.api_key = "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"

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