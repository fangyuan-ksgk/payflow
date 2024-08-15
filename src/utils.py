from pdf2image import convert_from_path
import base64
from openai import OpenAI 
import os
import io 
import re 
import json
import glob
from pypdf import PdfReader
from .prompt import PARSE_AOR_PROMPT
from .aor import AOR


oai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_oai_response(prompt, system_prompt="You are a helpful assistant", img=None, img_type=None):
    
    if isinstance(prompt, str):
        msg = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    else:
        msg = [
            {"role": "system", "content": system_prompt},
        ]
        msg.extend(prompt)
    
    if img is not None and img_type is not None:
        if isinstance(img, str):
            img = [img]
        image_content = []
        for _img in img:
            image_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_img}"}})
            
        text = msg[-1]["content"]
        text_content = [{"type": "text", "text": text}]
        
        msg.append({
            "role": "user",
            "content": text_content + image_content,
        })
        
    response = oai_client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=msg,
    )
    
    # print("Response: ", response.choices[0].message.content)
    
    return response.choices[0].message.content


def get_pdf_contents(pdf_file, first_page=1, last_page=1):
    # Convert the first page of the PDF to an image
    images = convert_from_path(pdf_file, first_page=first_page, last_page=last_page)

    pdf_base64_images = []
    for pdf_image in images:
        # Convert PIL Image to base64 string
        buffered = io.BytesIO()
        pdf_image.save(buffered, format="PNG")
        pdf_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        pdf_base64_images.append(pdf_image_base64)
        
    return pdf_base64_images

def get_pdf_text(pdf_path: str) -> str:
    pdf_text = ""
    reader = PdfReader(pdf_path)
    number_of_pages = len(reader.pages)
    for i in range(number_of_pages):
        page = reader.pages[i]
        pdf_text += page.extract_text()
    return pdf_text


import ast 

def load_json_with_ast(json_str):
    json_str_cleaned = json_str.strip()
    papers = ast.literal_eval(json_str_cleaned)
    return papers


def parse_json_response(content):
    try:
        # Try to parse the entire content as JSON
        json_data = json.loads(content)
    except json.JSONDecodeError:
        # If that fails, try to extract JSON object using regex
        match = re.search(r'\{.*?\}', content, re.DOTALL)
        if match:
            json_str = match.group(0)
            try:
                json_data = json.loads(json_str)
            except json.JSONDecodeError:
                try:
                    json_data = load_json_with_ast(json_str)
                except:
                    return {}
        else:
            return {}
    
    return json_data


def preprocess_aor(aor_dir: str = "database/aor"):
    pdf_paths = glob.glob(f"{aor_dir}/*.pdf")

    print("Preprocessing AORs...")
    for pdf_path in pdf_paths:
        # TBD: Skip preprocessing if things are already preprocessed
        pdf_txt = get_pdf_text(pdf_path)
        prompt = PARSE_AOR_PROMPT.format(pdf_txt=pdf_txt)
        response = get_oai_response(prompt)
        parsed_aor_dict = parse_json_response(response)
        aor = AOR(**parsed_aor_dict)
        aor.pdf_text = pdf_txt
        aor.pdf_path = pdf_path
        aor.save(aor_dir)