# Specific Function to enhance performance on single pdf file here
import re

def parse_aor_text(text):
    sections = {
        "BACKGROUND": "",
        "AIM": "",
        "SCOPE OF WORK AND CONTRACT VALUE": "",
        "EVALUATION METHODOLOGY": "",
        "RECOMMENDATION": "",
        "GENERAL REQUIREMENT": "",
        "TECHNICAL REQUIREMENTS FOR TRIAL OF CMDS-LITE": "",
        "DETAILS OF POST-TRIAL REPORT": "",
        "FAMILIARISATION": "",
        "PUBLICATIONS": "",
        "QUALITY, INSPECTION, AND ACCEPTANCE": "",
        "WARRANTY": "",
        "OPTION FOR SCALE-UP": "",
        "CLARIFICATIONS": ""
    }
    
    current_section = None
    for line in text.split('\n'):
        line = line.strip()
        if line in sections:
            current_section = line
        elif current_section:
            sections[current_section] += line + "\n"
    
    parsed_aor = {}
    for section, content in sections.items():
        if content.strip():
            parsed_aor[section] = {
                "description": section,
                "original_text": content.strip()
            }
    
    return parsed_aor