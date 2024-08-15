import dataclasses
from dataclasses import dataclass
from typing import List
from datetime import date

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
  

    @property
    def load(self, aor_path: str):
        import json
        with open(aor_path, "r") as f:
            aor_dict = json.load(f)
        aor_dict['no']= aor_dict['no'].replace("-","/")
        return AOR(**aor_dict)