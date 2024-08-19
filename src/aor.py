import dataclasses
from dataclasses import dataclass
from typing import List, Dict
from datetime import date
from dataclasses import field
import glob 

# Use case 1: how much left in the AOR xxx, what items are allowed, I want to claim xxx, can I do that?
# Use case 2: I have this invoice, can you help me check whether it fits into AOR xxx ?

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