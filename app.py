import base64
import json
import os
import re
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
import threading
import time

import dash
from dash import html, dcc, callback, Input, Output, State
from dash.dependencies import Input, Output, State
import dash.exceptions
import dash_bootstrap_components as dbc
import google.generativeai as genai
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Annotated
from pydantic.functional_validators import BeforeValidator

# Progress tracking class
@dataclass
class ProgressState:
    current: int = 0
    total: int = 0
    is_processing: bool = False
    current_file: Optional[str] = None
    needs_refresh: bool = False

# Global progress state
progress_state = ProgressState()

# Initialize Dash app with a modern theme
app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True
)

def parse_date_string(date_str: str) -> date:
    try:
        return datetime.strptime(date_str, "%d-%m-%y").date()
    except ValueError:
        try:
            return datetime.strptime(date_str, "%Y-%m-%d").date()
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
    analyzed: datetime = Field(default_factory=datetime.now)
    
    model_config = ConfigDict(
        json_encoders={
            date: lambda d: d.strftime("%d-%m-%y") if d else "NA",
            datetime: lambda dt: dt.strftime("%Y-%m-%d %H:%M:%S")
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
    data_dir.mkdir(exist_ok=True)  # Create data directory if it doesn't exist
    return sorted([f for f in data_dir.glob("*.pdf")])

def get_analyzed_files():
    """Get list of analyzed JSON files in the index directory."""
    index_dir = Path("index")
    index_dir.mkdir(exist_ok=True)
    return sorted([f for f in index_dir.glob("*.json")])

def clean_json_response(response_text: str) -> str:
    """Remove markdown code blocks and any other non-JSON formatting from the response."""
    # Remove markdown code blocks
    clean_text = re.sub(r'```json\s*|\s*```', '', response_text)
    # Remove any leading/trailing whitespace
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
    """Create the file list component with dropdown."""
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
                disabled=len(pending_pdfs) == 0,
                style={"display": "none" if len(pending_pdfs) == 0 else "block"},
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
            html.Div(
                "All files have been analyzed!", 
                id="analysis-complete-message",
                className="text-success mt-3",
                style={"display": "block" if len(pending_pdfs) == 0 else "none"}
            )
        ])
    ])

# Layout
app.layout = dbc.Container([
    html.H1("Academic Explorer", className="my-4"),
    dbc.Row([
        dbc.Col(create_file_list(), width=4),
        dbc.Col(create_analysis_display(), width=8),
    ]),
    dcc.Store(id="current-analysis"),
    dcc.Store(id="refresh-trigger", data=0),
    dcc.Loading(
        id="loading",
        children=html.Div(id="loading-output"),
        type="circle",
    ),
    dcc.Interval(
        id='progress-interval',
        interval=500,
        n_intervals=0,
        disabled=True
    )
], fluid=True)

# Analysis prompt
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

def process_pdfs_thread():
    """Function to process PDFs in a separate thread"""
    global progress_state
    
    _, pending_pdfs = get_pdf_status()
    progress_state.total = len(pending_pdfs)
    progress_state.current = 0
    progress_state.is_processing = True
    progress_state.needs_refresh = False
    
    try:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        for pdf in pending_pdfs:
            progress_state.current_file = pdf.name
            try:
                # Read and encode PDF file
                doc_path = pdf.absolute()
                with open(doc_path, "rb") as doc_file:
                    doc_data = base64.b64encode(doc_file.read()).decode("utf-8")
                
                # Generate analysis using Gemini
                response = model.generate_content(
                    [
                        {'mime_type': 'application/pdf', 'data': doc_data},
                        ANALYSIS_PROMPT
                    ],
                    generation_config={'temperature': 0},
                    stream=False
                )
                
                # Process response
                cleaned_response = clean_json_response(response.text)
                paper_analysis = PaperAnalysis.model_validate_json(cleaned_response)
                paper_analysis.analyzed = datetime.now()
                
                # Save analysis to JSON file
                json_output = paper_analysis.model_dump_json(indent=2)
                index_dir = Path("index")
                index_dir.mkdir(exist_ok=True)
                json_path = index_dir / f"{pdf.stem}.json"
                
                with open(json_path, 'w', encoding='utf-8') as f:
                    f.write(json_output)
                
                progress_state.current += 1
                progress_state.needs_refresh = True
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error analyzing {pdf.name}: {str(e)}")
                continue
                
    except Exception as e:
        print(f"Error initializing Gemini API: {str(e)}")
    
    finally:
        progress_state.is_processing = False
        progress_state.current_file = None

@callback(
    [Output("analysis-progress", "value"),
     Output("analysis-progress", "style"),
     Output("analysis-progress", "label"),
     Output("batch-analyze-btn", "disabled"),
     Output("batch-analyze-btn", "style"),
     Output("file-dropdown", "options"),
     Output("progress-interval", "disabled"),
     Output("refresh-trigger", "data"),
     Output("analysis-complete-message", "style")],
    [Input("batch-analyze-btn", "n_clicks"),
     Input("progress-interval", "n_intervals")],
    [State("refresh-trigger", "data")],
    prevent_initial_call=True
)
def handle_analysis_and_progress(n_clicks, n_intervals, refresh_count):
    """Handle both analysis initialization and progress updates"""
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    analyzed_pdfs, pending_pdfs = get_pdf_status()
    new_options = [{"label": f"📊 {pdf.name}", "value": pdf.stem} for pdf in analyzed_pdfs]
    
    if trigger_id == "batch-analyze-btn" and n_clicks:
        if not pending_pdfs:
            return (0, {"display": "none"}, "", True, {"display": "none"}, 
                    new_options, True, refresh_count, {"display": "block"})
        
        thread = threading.Thread(target=process_pdfs_thread)
        thread.start()
        
        return (0, {"display": "block"}, "Starting analysis...", True, 
                {"display": "block"}, new_options, False, refresh_count, 
                {"display": "none"})
    
    elif trigger_id == "progress-interval":
        if not progress_state.is_processing and progress_state.current == progress_state.total:
            analyzed_pdfs, pending_pdfs = get_pdf_status()
            final_options = [{"label": f"📊 {pdf.name}", "value": pdf.stem} for pdf in analyzed_pdfs]
            button_style = {"display": "none"} if not pending_pdfs else {"display": "block"}
            
            new_refresh_count = refresh_count + 1 if progress_state.needs_refresh else refresh_count
            progress_state.needs_refresh = False
            
            return (100, {"display": "block"}, f"Completed {progress_state.total} files",
                    not pending_pdfs, button_style, final_options, True, 
                    new_refresh_count, {"display": "block"})
        
        if progress_state.total > 0:
            progress = (progress_state.current / progress_state.total) * 100
            current_file = progress_state.current_file or "Preparing..."
            label = f"Processing {progress_state.current}/{progress_state.total} - Current file: {current_file}"
            
            analyzed_pdfs, _ = get_pdf_status()
            current_options = [{"label": f"📊 {pdf.name}", "value": pdf.stem} for pdf in analyzed_pdfs]
            
            return (progress, {"display": "block"}, label, True, 
                    {"display": "block"}, current_options, False, refresh_count,
                    {"display": "none"})
    
    return dash.no_update

@callback(
    Output("analysis-display", "children"),
    [Input("file-dropdown", "value")],
    prevent_initial_call=True
)
def update_analysis_display(selected_file):
    """Update the analysis display based on dropdown selection."""
    if not selected_file:
        return "Select a file to view its analysis."
    
    json_path = Path("index") / f"{selected_file}.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            analysis_data = json.load(f)
            return create_analysis_content(analysis_data)
    
    return "Analysis file not found."

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