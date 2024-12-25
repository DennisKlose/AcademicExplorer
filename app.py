import base64
import google.generativeai as genai
import pydantic
from typing import List
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ConfigDict
from typing_extensions import Annotated
import re
import json
from pathlib import Path
from datetime import date
import datetime
from pydantic.functional_validators import BeforeValidator

# Load environment variables
load_dotenv()

# Custom date validator function
def parse_date_string(date_str: str) -> date:
    """Convert DD-MM-YY string to date object."""
    try:
        return datetime.datetime.strptime(date_str, "%d-%m-%y").date()
    except ValueError:
        # Try alternative formats if the primary format fails
        try:
            return datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            raise ValueError(f"Date must be in DD-MM-YY format, got: {date_str}")

# Type for date fields with validation
DateField = Annotated[date, BeforeValidator(parse_date_string)]

# Define Pydantic model for structured output
class PaperAnalysis(BaseModel):
    title: str
    authors: List[str]
    affiliations: List[str]
    background: str
    hypotheses: List[str]
    major_findings: List[str]
    sample_types: List[str]
    methods: List[str]
    journal: str
    date_received: DateField
    date_revised: DateField
    date_accepted: DateField
    date_published: DateField
    
    model_config = ConfigDict(
        json_encoders={
            date: lambda d: d.strftime("%d-%m-%y")  # Format dates as DD-MM-YY in JSON output
        }
    )

def clean_json_response(response_text: str) -> str:
    """Remove markdown code blocks and any other non-JSON formatting from the response."""
    # Remove markdown code blocks
    clean_text = re.sub(r'```json\s*|\s*```', '', response_text)
    # Remove any leading/trailing whitespace
    clean_text = clean_text.strip()
    return clean_text

def save_json_response(json_data: str, input_pdf_path: str):
    """Save JSON response to a file with the same name as the PDF but .json extension."""
    # Convert PDF path to Path object for easier manipulation
    pdf_path = Path(input_pdf_path)
    
    # Create index directory if it doesn't exist
    index_dir = Path("index")
    index_dir.mkdir(exist_ok=True)
    
    # Create output path with same name but .json extension in index directory
    json_filename = pdf_path.stem + ".json"
    json_path = index_dir / json_filename
    
    # Save JSON to file
    with open(json_path, 'w', encoding='utf-8') as f:
        f.write(json_data)
    
    print(f"JSON saved to: {json_path}")

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

# Construct the prompt for structured analysis
analysis_prompt = """
Analyze this scientific paper and provide ONLY a JSON response with no additional text or markdown formatting. The response should contain the following fields:
{
    "title": "paper title",
    "authors": ["list of authors"],
    "affiliations": ["list of universities and other author affiliations"],
    "background": "one sentence describing background/relevance",
    "hypotheses": ["list of major hypotheses/questions"],
    "major_findings": ["list of key findings, 1-2 sentences each"],
    "sample_types": ["types of samples used (human/mouse/other)"],
    "methods": ["list of methods and computational techniques used"],
    "journal": "journal name",
    "date_received": "DD-MM-YY",
    "date_revised": "DD-MM-YY",
    "date_accepted": "DD-MM-YY",
    "date_published": "DD-MM-YY"
}
Return ONLY the JSON with the actual values, with no markdown formatting or code blocks.
Ensure all dates are in DD-MM-YY format (e.g., "01-03-23" for March 1, 2023).
"""

# Read and process the PDF
doc_path = "data/s41398-020-01147-z.pdf"
with open(doc_path, "rb") as doc_file:
    doc_data = base64.standard_b64encode(doc_file.read()).decode("utf-8")

# Generate response with JSON format specified
response = model.generate_content(
    [
        {'mime_type': 'application/pdf', 'data': doc_data},
        analysis_prompt
    ],
    generation_config={
        'temperature': 0.3,  # Lower temperature for more structured output
    },
    stream=False
)

try:
    # Clean the response text
    cleaned_response = clean_json_response(response.text)
    
    # Parse the response into Pydantic model using the newer validation method
    paper_analysis = PaperAnalysis.model_validate_json(cleaned_response)
    
    # Convert to JSON for output
    json_output = paper_analysis.model_dump_json(indent=2)
    
    # Print to console
    print(json_output)
    
    # Save to file
    save_json_response(json_output, doc_path)
    
except pydantic.ValidationError as e:
    print(f"Error parsing response: {e}")
    print("Validation error details:")
    for error in e.errors():
        print(f"- {error['loc']}: {error['msg']}")
except Exception as e:
    print(f"Unexpected error: {e}")
    print(f"Raw response: {response.text}")  # Added for debugging