import os, json, re, shutil
import pandas as pd
from ftfy import fix_text
from langchain.tools import tool
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from nlp_dedup import Deduper

load_dotenv()

def text_cleaning(text):
    if "24*" in text:
        return text  
    try:
        text = text.encode('latin1').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    text = re.sub(r'_x[0-9A-Fa-f]{4}_', '', text)
    text = text.replace('â€™', '')
    text = re.sub(r'[^A-Za-z0-9\s\'.,-]', '', text)
    text = fix_text(str(text))
    text = re.sub(r"<.*?>", " ", text)          # remove HTML
    text = re.sub(r"\?{2,}", " ", text)         # remove ??, ???, etc.
    text = re.sub(r"[âà¸]+", " ", text)         # remove common broken encoding leftovers
    text = re.sub(r"\s+", " ", text).strip()    # normalize spaces
    return text

def remove_duplicates(input_df_path, text_col, output_csv_path="Data/customer_messages_deduplicated.csv", output_dir="deduplicated"):

    df = pd.read_csv(input_df_path)
    df[text_col] = df[text_col].apply(text_cleaning)
    corpus = df[text_col].dropna().tolist()

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    deduper = Deduper(
        similarity_threshold=0.95,
        ngram_size=3
    )

    deduper.deduplicate(corpus=corpus, overwrite=True)

    jsonl_path = os.path.join(output_dir, "deduplicated_corpus.jsonl")
    dedup_df = pd.read_json(jsonl_path, lines=True)
    dedup_df.to_csv(output_csv_path, index=False)
    return dedup_df

@tool
def llm_customer_messages_analysis(input_json: str) -> str:
    """ Analyze customer messages with an LLM, classify category and sentiment, and save the results to CSV files.
    Accepts either a plain CSV path or a JSON string with input_csv/output_csv/column/batch_size
    """

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

    input_json = input_json.strip()

    # Accept plain CSV path
    if input_json.lower().endswith(".csv") and not input_json.startswith("{"):
        payload = {
            "input_csv": input_json,
            "output_csv": "results/customer_messages_analysis.csv",
            "column": "message",
            "batch_size": 10
        }
    else:
        payload = json.loads(input_json)

    input_csv = payload.get("input_csv", "Data/customer_messages.csv")
    output_csv = payload.get("output_csv", "results/customer_messages_analysis.csv")
    column = payload.get("column", "message")
    batch_size = payload.get("batch_size", 10)

    os.makedirs("results", exist_ok=True)

    deduplicated_data = "results/customer_messages_deduplicated.csv"

    # Step 1: deduplicate inside the tool
    remove_duplicates(
        input_df_path=input_csv,
        text_col=column,
        output_csv_path=deduplicated_data,
        output_dir="deduplicated"
    )

    # Step 2: read deduplicated data
    df = pd.read_csv(deduplicated_data)

    if "text" not in df.columns:
        return json.dumps({
            "status": "error",
            "message": "Deduplicated output does not contain 'text' column.",
            "available_columns": list(df.columns)
        }, ensure_ascii=False, indent=2)

    df["text"] = df["text"].fillna("").astype(str)
    df["clean_message"] = df["text"].apply(text_cleaning)

    all_results = []

    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i + batch_size].copy()
        messages = batch_df["clean_message"].tolist()

        formatted_messages = [
            {"id": idx, "message": msg}
            for idx, msg in zip(batch_df.index.tolist(), messages)
        ]

        prompt = f"""
You are an expert customer support analyst.

Your task is to classify customer messages.

Use the following categories carefully:

Inquiry
Customer is asking for information, clarification, or status.

Complaint
Customer is reporting a problem or expressing dissatisfaction with a product or service.
No action is explicitly requested.

Suggestion
Customer proposes an improvement or idea.

Transactional Request
Customer explicitly asks for an action to be performed such as:
refund, cancellation, exchange, replacement, update, change, return, or order modification.

After categorizing the message, return the product it talks about, choose one of the following:
- Slim Fit T-shirt
- Coffee Press
- 3pc Cook Set - Steel
- Foldable Table
- Portable Blender
- Wireless Earbud Pro
- Kids Sneakers
- null, if user is not mentioning a specific product

Important Decision Rule:
If a message contains BOTH a complaint AND a request for action,
classify it as Transactional Request.

Products do not need to match exactly the defined title names.
If the customer message is not mentioning a specific product, return null.

For each message return:
- category: one of [Inquiry, Complaint, Suggestion, Transactional Request]
- sentiment: one of [Positive, Neutral, Negative]
- product

Critical output rules:
- Return exactly one JSON object for every input message.
- Preserve the same id for each message.
- Do not omit any id.
- Return ONLY a valid JSON array.
- Do not include markdown fences.
- Do not include explanations.

Return ONLY a valid JSON array in this format:
[
  {{
    "id": 1,
    "category": "Complaint",
    "sentiment": "Negative",
    "product": "Foldable Table"
  }}
]

Messages:
{json.dumps(formatted_messages, ensure_ascii=False, indent=2)}
""".strip()

        response = llm.invoke(prompt).content.strip()
        response = response.replace("```json", "").replace("```", "").strip()

        try:
            batch_results = json.loads(response)
            if not isinstance(batch_results, list):
                batch_results = []
        except json.JSONDecodeError:
            batch_results = []

        returned_ids = {
            item.get("id") for item in batch_results
            if isinstance(item, dict) and "id" in item
        }

        for idx in batch_df.index.tolist():
            if idx not in returned_ids:
                batch_results.append({
                    "id": idx,
                    "category": "Inquiry",
                    "sentiment": "Neutral",
                    "product": None
                })

        all_results.extend(batch_results)

    results_df = pd.DataFrame(all_results)

    if results_df.empty:
        df["category"] = "Inquiry"
        df["sentiment"] = "Neutral"
        df["product"] = None
    else:
        results_df["id"] = pd.to_numeric(results_df["id"], errors="coerce")
        results_df = (
            results_df
            .dropna(subset=["id"])
            .drop_duplicates(subset=["id"], keep="first")
        )
        results_df["id"] = results_df["id"].astype(int)

        df = df.reset_index().rename(columns={"index": "row_id"})
        df["row_id"] = df["row_id"].astype(int)

        df = df.merge(results_df, left_on="row_id", right_on="id", how="left")

        df["category"] = df["category"].fillna("Inquiry")
        df["sentiment"] = df["sentiment"].fillna("Neutral")
        if "product" in df.columns:
            df["product"] = df["product"].where(df["product"].notna(), None)

        df = df.drop(columns=["row_id", "id"], errors="ignore")

    df.to_csv(output_csv, index=False)

    example_rows = df[["text", "category", "sentiment", "product"]].head(5).to_dict(orient="records")

    return json.dumps({
        "status": "success",
        "deduplicated_csv": deduplicated_data,
        "analyzed_csv": output_csv,
        "rows_processed": len(df),
        "text_column": "text",
        "examples": example_rows
    }, ensure_ascii=False, indent=2)