PARSE_AOR_PROMPT = """Please analyze the AOR (Authority of Requirement) file and provide the following information in JSON format:

{{
    "items": ["list of items covered by the AOR"],
    "budgets": [0.0, 0.0, ...],  // list of corresponding budgets for each item
    "no": "AOR number",
    "description": "brief description of the AOR",
    "expiry_date": "YYYY-MM-DD"
}}

Here is the AOR file: {pdf_txt}"""
