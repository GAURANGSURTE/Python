# the plain Code 
# Pdf Extraction using PyPDF

# Installing the Library
!pip install pymupdf
Importing Libraries
import fitz
import pandas as pd
import re

#Extracting data from the pdf
pdf_path = 'datasets\\a1988-59.pdf'

doc = fitz.open(pdf_path)

text = ""
for page in doc:
    text += page.get_text()

print("Text Length: ", len(text))

#Cleaning the text by adding spaces and sub
text = text.replace("\n", " ")
text = re.sub(r'\s+', ' ', text)
# Detecting the chapters and sections, subsections as our dataset consists of the part such as 
# (no): what the section is 
# (no): subsections explaining about the terms that also false under the section 
rows = []

current_chapter = None
current_section = None

tokens = re.split(r'(?=\bCHAPTER\b|\d+\.)', text)

for token in tokens:
    
    # Detect chapter
    chap = re.search(r'CHAPTER\s+([IVXLC]+)', token)
    if chap:
        current_chapter = chap.group(1)
    
    # Detect section
    sec = re.match(r'(\d+)\.\s*(.*)', token)
    if sec:
        current_section = sec.group(1)
        rows.append({
            "chapter": current_chapter,
            "section": current_section,
            "subsection": "main",
            "text": sec.group(2).strip()
        })
    
    # Detect subsections (i), (ii), (iii)
    subs = re.findall(r'\((i+|v|x+)\)\s*(.*?)(?=\(\w+\)|Explanation|$)', token)
    for s in subs:
        rows.append({
            "chapter": current_chapter,
            "section": current_section,
            "subsection": s[0],
            "text": s[1].strip()
        })
    
    # Detect explanation
    exp = re.search(r'Explanation\.\—(.*)', token)
    if exp:
        rows.append({
            "chapter": current_chapter,
            "section": current_section,
            "subsection": "explanation",
            "text": exp.group(1).strip()
        })

# Converting into the DataFrame for better redabolity and visualise
df = pd.DataFrame(rows)
df.head(20)
df.to_csv("Motor_Vehicle_Act_Dataset.csv", index= False)
print("Dataset saved successfully")

#Chunking the Data If you require to chunck it for using it in the ML model normally or in the RAG
def chunk_text(text, size= 200):
    words = text.split()
    for i in range(0, len(words), size):
        yield " ".join(words[i:i+size])

#Checking the Saved file Results to see for and Anomaly present in the data
df1 = pd.read_csv('datasets\\Motor_Vehicle_Act_Dataset.csv')
df1.head()
