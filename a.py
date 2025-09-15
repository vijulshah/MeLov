import pathlib

from google import genai
from google.genai import types

client = genai.Client(api_key="AIzaSyCJaITZX82UvKA6F3sTuIJIwB1SeynamXs")

doc_url = "./data/raw/my_biodata/Bio Data - Vijul Shah.pdf"
filepath = pathlib.Path(doc_url)
prompt = "Extract information from this bio-data and show in json format."
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        types.Part.from_bytes(
            data=filepath.read_bytes(),
            mime_type="application/pdf",
        ),
        prompt,
    ],
)
print(response.text)
