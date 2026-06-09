import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from PIL import Image
from pdf2image import convert_from_path
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer, util

# Set page layout to wide
st.set_page_config(page_title="Petition Routing Dashboard", layout="wide")

# CACHED SYSTEM INITIALIZATION
# ==========================================
@st.cache_resource
def load_resources():
    """Load the Excel sheet, pre-calculated embeddings, and the ML model."""
    excel_filename = "taxonomy.xlsx"
    embeddings_filename = "taxonomy_embeddings.npy"
    
    # Load DataFrame
    df = pd.read_excel(excel_filename)
    df.columns = df.columns.str.strip()

    # Load the pre-calculated embeddings (takes less than a second)
    import torch
    embeddings_raw = np.load(embeddings_filename)
    embeddings = torch.from_numpy(embeddings_raw) # Convert to PyTorch tensor for cosine similarity

    # Loading model (only used now to encode the single incoming user query)
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    return df, model, embeddings

# Initialize Gemini Client using environment variable
ai_client = None
gemini_key = os.environ.get("GEMINI_API_KEY")

if gemini_key:
    ai_client = genai.Client(api_key=gemini_key)

# Load cached resources
resources_loaded = False
try:
    df, similarity_model, taxonomy_embeddings = load_resources()
    resources_loaded = True
except Exception as e:
    st.error(f"Failed to load Excel taxonomy or Model: {e}")

# ==========================================
# PROCESSING FUNCTIONS
# ==========================================

def preprocess_pdf_to_images(pdf_path):
    pages = convert_from_path(pdf_path, dpi=150) # Balanced DPI for speed and clarity
    return [page.convert('RGB') for page in pages]

def extract_from_multiple_images(images_list):
    prompt = """
    You are an expert assistant reading a multi-page handwritten Tamil petition.
    Look at all the attached page images in sequence to understand the entire petition.

    Task 1: Document Classification
    - Identify which pages represent the handwritten or typed petition text.
    - Identify which pages are supporting attachments (such as ID cards like Aadhaar, certificates, or receipts).

    Task 2: Full OCR Extraction
    - For the identified petition content, perform a detailed transcription of the text.
    - DO NOT transcribe the text inside the supporting attachments.
    - Save this full transcription under the key 'full_petition_ocr'.

    Task 3: Metadata Extraction
    - Extract the standard petitioner metadata from the entire document.

    Provide the output in English, strictly in JSON format with these exact keys:
    1. petitioner_name: (Name of the Petitioner)
    2. address_details: (Full address details)
    3. phone_number: (10-digit phone number if present, otherwise null)
    4. is_differently_abled: (boolean: true if petition mentions disability, otherwise false)
    5. full_petition_ocr: (The full Tamil transcription of ONLY the petition pages, preserving paragraph breaks)
    6. grievance_description: (A clear, precise and consolidated English summary (Don't mention name, address here) optimized for administrative routing. Emphasize key action words)
    7. attachments: (List of names/types of attachments identified, e.g., ["Aadhaar Card", "Receipt"])

    Do not include markdown formatting or backticks around the JSON output.
    """
    contents = []
    contents.extend(images_list)
    contents.append(prompt)

    response = ai_client.models.generate_content(
        model='gemini-2.5-flash',  # Updated to standard stable version
        contents=contents,
        config=types.GenerateContentConfig(response_mime_type="application/json")
    )
    return json.loads(response.text)

def get_top_10_candidates(english_grievance):
    query_embedding = similarity_model.encode(english_grievance, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, taxonomy_embeddings)[0]
    top_results_indices = np.argsort(cosine_scores.cpu().numpy())[-10:][::-1]

    candidates = []
    for idx in top_results_indices:
        row = df.iloc[idx]
        candidates.append({
            "excel_row_index": int(idx),
            "Department Name": row['Department Name'],
            "Grievance Type": row['Grievance Type'],
            "Grievance Sub Type": row['Grievance Sub Type'],
            "Sub Department": row['Sub Department'],
            "Responsible officer": row['Responsible officer'],
            "similarity_score": f"{round(float(cosine_scores[idx]) * 100, 2)}%"
        })
    return candidates

def select_best_candidate_with_gemini(english_grievance, candidates):
    prompt = f"""
    You are an administrative routing expert. Your task is to analyze a citizen's grievance summary and choose the most appropriate classification row from a list of 10 candidate options.

    Citizen Grievance Summary:
    "{english_grievance}"

    Candidate Options (from our Excel database):
    {json.dumps(candidates, indent=2)}

    Task:
    1. Compare the Citizen Grievance Summary with the 'Grievance Type' and 'Grievance Sub Type' of the candidates.
    2. Choose the single row that represents the most logical and accurate destination for this specific petitioner.
    3. Estimate your routing confidence as a percentage score between 0% and 100%.

    Provide the final decision strictly in JSON format with these exact keys:
    1. Department Name: (Selected candidate value)
    2. Sub Department: (Selected candidate value)
    3. Grievance Type: (Selected candidate value)
    4. Grievance Sub Type: (Selected candidate value)
    5. Responsible officer: (Selected candidate value)
    6. similarity_confidence: (Confidence percentage, e.g., "92%")
    7. selection_reason: (A very brief explanation of why this candidate was selected)

    Do not include markdown formatting or backticks around the JSON.
    """
    response = ai_client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json")
    )
    return json.loads(response.text)

# ==========================================
# STREAMLIT UI LAYOUT
# ==========================================

st.title("📋 OCR & Routing System of CM cell petitions")
st.markdown("Upload the petition (PDF) to perform OCR, extract the metadata and department routing.")

# Check layout feasibility
if not resources_loaded:
    st.info("Please upload your 'taxonomy.xlsx' file to your workspace to activate the application.")
elif not ai_client:
    st.warning("Please configure your GEMINI_API_KEY in the Hugging Face Space Secrets to proceed.")
else:
    # File Uploader
    uploaded_file = st.file_uploader("Upload Handwritten Petition PDF", type=["pdf"])

    if uploaded_file is not None:
        temp_pdf_path = "temp_uploaded_petition.pdf"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("🚀 Process and Route Petition", type="primary"):
            with st.spinner("Step 1/4: Converting PDF pages to images..."):
                images = preprocess_pdf_to_images(temp_pdf_path)
                st.success(f"Successfully loaded {len(images)} document page(s).")

            with st.spinner("Step 2/4: Processing handwriting and metadata..."):
                extraction_result = extract_from_multiple_images(images)
                english_summary = extraction_result.get("grievance_description")

            with st.spinner("Step 3/4: Fetching top candidates from taxonomy..."):
                candidates = get_top_10_candidates(english_summary)

            with st.spinner("Step 4/4: Confirming final routing assignment..."):
                final_routing = select_best_candidate_with_gemini(english_summary, candidates)

            st.balloons()
            st.success("Routing Complete!")

            st.subheader("Full OCR Transcription of the Petition")
            st.text_area(
                    label=" ",
                    value=extraction_result.get("full_petition_ocr"),
                    height=450
                )

            st.subheader("👤 Petitioner Profile")
            st.write(f"**Name:** {extraction_result.get('petitioner_name')}")
            st.write(f"**Phone Number:** {extraction_result.get('phone_number')}")
            st.write(f"**Differently Abled Status:** {'Yes' if extraction_result.get('is_differently_abled') else 'No'}")
            st.write(f"**Address:** {extraction_result.get('address_details')}")
            st.write(f"**Mentioned Attachments:** {', '.join(extraction_result.get('attachments', [])) if extraction_result.get('attachments') else 'None'}")
            st.subheader("📝 Summarized Grievance")
            st.text_area("English Summary: ", english_summary, height=200)

            st.subheader("🎯 Department & Category Routing")
            st.write(f"**Assigned Department:** {final_routing.get('Department Name')}")
            st.write(f"**Sub Department:** {final_routing.get('Sub Department')}")
            st.write(f"**Grievance Type:** {final_routing.get('Grievance Type')}")
            st.write(f"**Grievance Sub Type:** {final_routing.get('Grievance Sub Type')}")
            st.write(f"**Responsible Officer:** {final_routing.get('Responsible officer')}")
            st.write(f"**Mapping Confidence Score:** {final_routing.get('similarity_confidence')}")
            st.write(f"**Routing Reason:** *{final_routing.get('selection_reason')}*")

            with st.expander("🔍 View Pre-Reranking Top 10 Candidates (Embedding Scores)"):
                st.write(pd.DataFrame(candidates)[["excel_row_index", "Department Name", "Grievance Type", "Grievance Sub Type", "similarity_score"]])

            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)
