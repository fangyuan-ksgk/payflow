import dataclasses
from dataclasses import dataclass
from typing import List, Dict
from datetime import date
from dataclasses import field
import glob 
from PIL import Image, ImageDraw, ImageFont
from pdf2image import convert_from_path
import io

def pdf_to_img(pdf_file, first_page=1, last_page=1):
    # Convert the first page of the PDF to an image
    images = convert_from_path(pdf_file, first_page=first_page, last_page=last_page)        
    return images[0]

def file_to_img(file_path):
    if file_path.endswith(".pdf"):
        img = pdf_to_img(file_path)
    elif file_path.endswith(".eml"):
        img = None 
    elif file_path.endswith((".png", ".jpg", ".jpeg")):
        with Image.open(file_path) as img_raw:
            buffered = io.BytesIO()
            img_raw.save(buffered, format="PNG")
            img = buffered.getvalue()
    else:
        raise ValueError("Unknown file format")
    return img

def render_aor_image(aor):
    # Create a new image with a white background
    img = Image.new('RGB', (800, 600), color='white')

    # Create a drawing object
    draw = ImageDraw.Draw(img)

    # Define font
    font = ImageFont.load_default().font_variant(size=20)

    # Draw AOR details
    y_offset = 50
    draw.text((50, y_offset), f"AOR No: {aor.no}", fill='black', font=font)
    y_offset += 30
    # Split description into multiple lines if it's too long
    description_words = aor.description.split()
    description_lines = []
    current_line = ""
    for word in description_words:
        if draw.textlength(current_line + " " + word, font=font) < 500:
            current_line += " " + word if current_line else word
        else:
            description_lines.append(current_line)
            current_line = word
    if current_line:
        description_lines.append(current_line)

    # Draw description lines
    for line in description_lines:
        draw.text((50, y_offset), f"Description: {line}" if line == description_lines[0] else line, fill='black', font=font)
        y_offset += 30
    draw.text((50, y_offset), f"Expiry Date: {aor.expiry_date}", fill='black', font=font)
    y_offset += 50

    # Draw items and budgets
    for item, budget in zip(aor.items, aor.budgets):
        item_text = f"{item}"
        budget_text = f"${budget:.2f}"
        
        # Draw item (left-aligned)
        draw.text((50, y_offset), item_text, fill='black', font=font, align="left")
        
        # Draw budget (right-aligned)
        budget_width = draw.textlength(budget_text, font=font)
        draw.text((750 - budget_width, y_offset), budget_text, fill='black', font=font, align="right")
        
        y_offset += 30

    # Save the image to a bytes buffer
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    return Image.open(img_buffer)


@dataclass
class AOR:
    items: list[str]
    budgets: List[float]
    no: str
    description: str # this is the place where we would perform RAG-based chat
    expiry_date: date
    pdf_text: str = ""
    pdf_path: str = ""
    cached_dict: Dict[str, str] = field(default_factory=dict) # TBD: Cached retrieval results in dictionary format (description : content)

    @property 
    def remaining_budgets(self):
        return self.budgets
    
    @property
    def image(self):
        if self.pdf_path.endswith(".eml"):
            return render_aor_image(self)
        else:
            return file_to_img(self.pdf_path)
    
    def save(self, aor_dir: str = "database/aor"):
        import os
        import json
        
        # Create the directory if it doesn't exist
        os.makedirs(aor_dir, exist_ok=True)
        
        self.no = self.no.replace("/","-")

        # Create the file path
        file_path = os.path.join(aor_dir, f"{self.no}.json")
        
        # Convert the AOR object to a dictionary
        aor_dict = dataclasses.asdict(self)
        
        # Write the dictionary to a JSON file
        with open(file_path, 'w') as f:
            json.dump(aor_dict, f, indent=4)
  

    @classmethod
    def load(self, aor_path: str):
        import json
        with open(aor_path, "r") as f:
            aor_dict = json.load(f)
        aor_dict['no']= aor_dict['no'].replace("-","/")
        return AOR(**aor_dict)
    
    @property
    def narrative(self): 
        # This is the light-rag, which always provide a simple narrative on the AOR pipeline, the more detailed RAG is over the pdf_text itself
        # TBD: a frequency-based caching mechanism to store (query, rag-result) pairs based on accepted responses (filtering mechanism)
        item_budget_str = ""
        for (item, budget) in zip(self.items, self.budgets):
            item_budget_str += f"Budget for {item} is {budget}\n"
        narrative_str = f"AOR no. {self.no}\n{item_budget_str}Expiry date: {self.expiry_date}\nDescription: {self.description}"
        return narrative_str
    

def load_aors(aor_dir: str = "database/aor"):
    aor_files = glob.glob(f"{aor_dir}/*.json")
    aor_list = []
    for aor_file in aor_files:
        aor = AOR.load(aor_file)
        aor_list.append(aor)
    return aor_list 


@dataclass 
class Invoice:
    no: str
    date: str
    currency: str  # ["Yuan", "Thai Baht", "Singapore Dollar", "US Dollar"]
    seller: str
    items: List[str]
    amounts: List[float]
    description: str = ""
    invoice_text: str = ""
    invoice_path: str = ""
    
    @property
    def narrative(self):
        item_amount_str = ""
        for (item, amount) in zip(self.items, self.amounts):
            item_amount_str += f"{item}: {amount} {self.currency}\n"
        narrative_str = f"Invoice no. {self.no}\nDate: {self.date}\nSeller: {self.seller}\nCurrency: {self.currency}\n{item_amount_str}Total Amount: {self.total_amount} {self.currency}"
        if self.invoice_path:
            narrative_str += f"\nInvoice Path: {self.invoice_path}"
        return narrative_str
    
    @property
    def image(self):
        return file_to_img(self.invoice_path)

    @property 
    def total_amount(self):
        return sum(self.amounts)
    
    def save(self, invoice_dir: str = "database/invoice"):
        import os
        import json
        
        # Create the directory if it doesn't exist
        os.makedirs(invoice_dir, exist_ok=True)
        
        self.no = self.no.replace("/","-")

        # Create the file path
        file_path = os.path.join(invoice_dir, f"{self.no}.json")
        
        # Convert the Invoice object to a dictionary
        invoice_dict = dataclasses.asdict(self)
        
        # Write the dictionary to a JSON file
        with open(file_path, 'w') as f:
            json.dump(invoice_dict, f, indent=4)

    @classmethod
    def load(cls, invoice_path: str):
        import json
        with open(invoice_path, "r") as f:
            invoice_dict = json.load(f)
        invoice_dict['no']= invoice_dict['no'].replace("-","/")
        return Invoice(**invoice_dict)
    
    
def load_invoices(invoice_dir: str = "database/invoice"):
    invoice_files = glob.glob(f"{invoice_dir}/*.json")
    invoice_list = []
    for invoice_file in invoice_files:
        invoice = Invoice.load(invoice_file)
        invoice_list.append(invoice)
    return invoice_list 