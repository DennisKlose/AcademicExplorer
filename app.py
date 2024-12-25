import base64
import json
import os
import re
from datetime import date
from pathlib import Path
from typing import List

import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import google.generativeai as genai
from pydantic import BaseModel, ConfigDict
from typing_extensions import Annotated
from pydantic.functional_validators import BeforeValidator
import datetime


# Initialize Dash app with a modern theme
app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True
)

# Reuse your existing models and functions
def parse_date_string(date_str: str) -> date:
    try:
        return datetime.datetime.strptime(date_str, "%d-%m-%y").date()
    except ValueError:
        try:
            return datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            raise ValueError(f"Date must be in DD-MM-YY format, got: {date_str}")

DateField = Annotated[date, BeforeValidator(parse_date_string)]

class PaperAnalysis(BaseModel):
    title: str = "NA"
    authors: List[str] = []
    affiliations: List[str] = []
    background: str = "NA"
    hypotheses: List[str] = []
    major_findings: List[str] = []
    sample_types: List[str] = []
    methods: List[str] = []
    journal: str = "NA"
    date_received: DateField | str
    date_revised: DateField | str
    date_accepted: DateField | str
    date_published: DateField | str
    
    model_config = ConfigDict(
        json_encoders={
            date: lambda d: d.strftime("%d-%m-%y") if d else "NA"
        }
    )

    def model_dump(self, *args, **kwargs):
        data = super().model_dump(*args, **kwargs)
        for key, value in data.items():
            if isinstance(value, list) and not value:
                data[key] = ["NA"]
        return data


def get_pdf_files():
    """Get list of PDF files in the data directory."""
    data_dir = Path("data")
    return sorted([f for f in data_dir.glob("*.pdf")])

def get_analyzed_files():
    """Get list of analyzed JSON files in the index directory."""
    index_dir = Path("index")
    index_dir.mkdir(exist_ok=True)
    return sorted([f for f in index_dir.glob("*.json")])

def clean_json_response(response_text: str) -> str:
    """Remove markdown code blocks and any other non-JSON formatting from the response."""
    clean_text = re.sub(r'```json\s*|\s*```', '', response_text)
    clean_text = clean_text.strip()
    return clean_text

def get_pdf_status():
    """Get lists of analyzed and pending PDF files."""
    pdf_files = get_pdf_files()
    analyzed_files = {f.stem for f in get_analyzed_files()}
    
    analyzed = []
    pending = []
    
    for pdf in pdf_files:
        if pdf.stem in analyzed_files:
            analyzed.append(pdf)
        else:
            pending.append(pdf)
            
    return analyzed, pending

def create_analysis_display():
    """Create a card to display analysis results."""
    return dbc.Card([
        dbc.CardHeader("Analysis Results"),
        dbc.CardBody(
            id="analysis-display",
            children=[
                html.P("Select a file to view its analysis.")
            ]
        )
    ])

def create_file_list():
    analyzed_pdfs, pending_pdfs = get_pdf_status()
    
    dropdown_options = [{"label": f"📊 {pdf.name}", "value": pdf.stem} for pdf in analyzed_pdfs]
    
    return dbc.Card([
        dbc.CardHeader("PDF Files"),
        dbc.CardBody([
            dbc.Label("Select an analyzed PDF to view:"),
            dcc.Dropdown(
                id='file-dropdown',
                options=dropdown_options,
                value='',
                clearable=True,
                className="mb-4"
            ),
            dbc.Button(
                ["Analyze All Pending Files"],
                id="batch-analyze-btn",
                color="primary",
                className="w-100",
                style={"display": "none"} if not pending_pdfs else {},
            ),
            dbc.Progress(
                id="analysis-progress",
                value=0,
                label="",
                striped=True,
                animated=True,
                className="mt-3",
                style={"display": "none"}
            ),
            html.Div("All files have been analyzed!", 
                    className="text-success mt-3",
                    style={"display": "block" if not pending_pdfs else "none"})
        ])
    ])

app.layout = dbc.Container([
    html.H1("Academic Explorer", className="my-4"),
    dbc.Row([
        dbc.Col(create_file_list(), width=4),
        dbc.Col(create_analysis_display(), width=8),
    ]),
    dcc.Store(id="current-analysis"),
    dcc.Loading(
        id="loading",
        children=html.Div(id="loading-output"),
        type="circle",
    )
], fluid=True)

# Layout
app.layout = dbc.Container([
    html.H1("Academic Explorer", className="my-4"),
    
    dbc.Row([
        dbc.Col(create_file_list(), width=4),
        dbc.Col(create_analysis_display(), width=8),
    ]),
    
    dcc.Store(id="current-analysis"),
    
    # Loading spinner
    dcc.Loading(
        id="loading",
        children=html.Div(id="loading-output"),
        type="circle",
    )
], fluid=True)

@callback(
    [Output("analysis-progress", "value"),
     Output("analysis-progress", "style"),
     Output("analysis-progress", "label"),
     Output("batch-analyze-btn", "disabled"),
     Output("file-dropdown", "options"),
     Output("current-analysis", "data")],
    Input("batch-analyze-btn", "n_clicks"),
    prevent_initial_call=True
)
def batch_analyze_pdfs(n_clicks):
    if not n_clicks:
        return dash.no_update

    _, pending_pdfs = get_pdf_status()
    total_files = len(pending_pdfs)
    
    if total_files == 0:
        return 0, {"display": "none"}, "", True, dash.no_update, dash.no_update

    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    current_analysis = None
    for i, pdf in enumerate(pending_pdfs, 1):
        progress = (i / total_files) * 100
        progress_label = f"Processing {i}/{total_files}"
        
        try:
            # Existing PDF analysis code...
            doc_path = pdf.absolute()
            with open(doc_path, "rb") as doc_file:
                doc_data = base64.standard_b64encode(doc_file.read()).decode("utf-8")
            
            response = model.generate_content(
                [{'mime_type': 'application/pdf', 'data': doc_data}, ANALYSIS_PROMPT],
                generation_config={'temperature': 0},
                stream=False
            )
            
            cleaned_response = clean_json_response(response.text)
            paper_analysis = PaperAnalysis.model_validate_json(cleaned_response)
            json_output = paper_analysis.model_dump_json(indent=2)
            
            index_dir = Path("index")
            index_dir.mkdir(exist_ok=True)
            json_path = index_dir / f"{pdf.stem}.json"
            
            with open(json_path, 'w', encoding='utf-8') as f:
                f.write(json_output)
            
            current_analysis = json.loads(json_output)
            
        except Exception as e:
            print(f"Error analyzing {pdf.name}: {str(e)}")
            continue

    analyzed_pdfs, _ = get_pdf_status()
    new_options = [{"label": f"📊 {pdf.name}", "value": pdf.stem} for pdf in analyzed_pdfs]
    
    return (
        100,  # Progress value
        {"display": "block"},  # Progress style
        f"Completed {total_files} files",  # Progress label
        True,  # Disable button
        new_options,  # Updated dropdown options
        current_analysis  # Last analyzed file
    )

# Analysis prompt as a constant
ANALYSIS_PROMPT = """
Analyze this scientific paper and provide ONLY a JSON response with no additional text or markdown formatting. The response should contain the following fields:
{
    "title": "paper title - delete all quotes from it",
    "authors": ["list of authors"],
    "affiliations": ["list of universities and other author affiliations"],
    "background": "one sentence describing background/relevance",
    "hypotheses": ["list of major hypotheses/questions"],
    "major_findings": ["list of key findings, 1-2 sentences each"],
    "sample_types": ["types of samples used (human/mouse/other)"],
    "methods": ["list of methods and computational techniques used"],
    "journal": "journal name",
    "date_received": "DD-MM-YY or NA",
    "date_revised": "DD-MM-YY or NA",
    "date_accepted": "DD-MM-YY or NA",
    "date_published": "DD-MM-YY or NA"
}
Return ONLY the JSON with the actual values, with no markdown formatting or code blocks.
For any field where data cannot be found, use "NA" for text fields or ["NA"] for list fields.
For dates, use DD-MM-YY format (e.g., "01-03-23") or "NA" if not found.
"""

@callback(
    Output("analysis-display", "children"),
    [Input("file-dropdown", "value"),
     Input("current-analysis", "data")],
    prevent_initial_call=True
)
def update_analysis_display(selected_file, current_analysis):
    """Update the analysis display based on dropdown selection or new analysis."""
    ctx = dash.callback_context
    trigger = ctx.triggered[0]["prop_id"]
    
    if "file-dropdown" in trigger and selected_file:
        json_path = Path("index") / f"{selected_file}.json"
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                analysis_data = json.load(f)
                return create_analysis_content(analysis_data)
    elif current_analysis:
        return create_analysis_content(current_analysis)
    
    return "Select a file to view its analysis."

def create_analysis_content(analysis_data):
    """Create the analysis content display."""
    return [
        html.H3(analysis_data["title"], className="mb-4"),
        html.H5("Authors"),
        html.P(", ".join(analysis_data["authors"])),
        html.H5("Affiliations"),
        html.Ul([html.Li(aff) for aff in analysis_data["affiliations"]]),
        html.H5("Background"),
        html.P(analysis_data["background"]),
        html.H5("Hypotheses"),
        html.Ul([html.Li(hyp) for hyp in analysis_data["hypotheses"]]),
        html.H5("Major Findings"),
        html.Ul([html.Li(finding) for finding in analysis_data["major_findings"]]),
        html.H5("Sample Types"),
        html.P(", ".join(analysis_data["sample_types"])),
        html.H5("Methods"),
        html.Ul([html.Li(method) for method in analysis_data["methods"]]),
        dbc.Row([
            dbc.Col([
                html.H5("Journal"),
                html.P(analysis_data["journal"])
            ], width=6),
            dbc.Col([
                html.H5("Important Dates"),
                html.P([
                    f"Received: {analysis_data['date_received']}", html.Br(),
                    f"Revised: {analysis_data['date_revised']}", html.Br(),
                    f"Accepted: {analysis_data['date_accepted']}", html.Br(),
                    f"Published: {analysis_data['date_published']}"
                ])
            ], width=6)
        ])
    ]

if __name__ == '__main__':
    app.run_server(debug=True)