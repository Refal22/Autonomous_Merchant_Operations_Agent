import pandas as pd
import numpy as np
from word2number import w2n
import re, os, json
from langchain.tools import tool
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from tools.customer_support_analysis import text_cleaning

load_dotenv()

def convert_price(value):
    """Convert price values to numeric (handles currency symbols and text numbers)."""
    value = str(value).strip().lower()
    # Remove currency symbols and keep digits and decimals
    cleaned = re.sub(r"[^\d.]", "", value.replace(",", ""))
    # Try numeric conversion
    try:
        return float(cleaned)
    except ValueError:
        pass
    # Try converting written numbers
    try:
        return float(w2n.word_to_num(value))
    except Exception:
        return np.nan

def parse_llm_mapping(text: str) -> dict:
    text = text.strip().replace("```json", "").replace("```", "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in LLM output:\n{text}")
    return json.loads(text[start:end + 1])

def apply_llm_mapping(df: pd.DataFrame, column: str, rules: list[str]) -> pd.DataFrame:
    if column not in df.columns:
        return df

    out = df.copy()
    original_col = f"{column}_original"
    out[original_col] = out[column]

    unique_values = out[column].dropna().astype(str).str.strip().unique().tolist()
    if not unique_values:
        return out

    payload = json.dumps({
        "column": column,
        "rules": rules,
        "values": unique_values
    }, ensure_ascii=False)

    raw = llm_normalize_values.invoke(payload)
    mapping = parse_llm_mapping(raw)

    out[column] = out[column].apply(
        lambda x: mapping.get(str(x).strip(), x) if pd.notna(x) else x
    )

    return out

@tool
def parse_catalog(file_path: str) -> str:
    """Inspect the raw CSV catalog and return a diagnostic summary."""

    file_path = file_path.strip()

    if not file_path.lower().endswith(".csv"):
        return "Input must be a CSV file."

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.lower()

    summary = {
        "rows": len(df),
        "columns": list(df.columns),
        "missing_values": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum())
    }

    if "product_id" in df.columns:
        summary["duplicate_product_ids"] = int(df["product_id"].duplicated().sum())

    if "title" in df.columns:
        titles = df["title"].dropna().astype(str).str.strip()
        summary["unique_titles"] = int(titles.nunique())
        summary["title_examples"] = titles.drop_duplicates().head(10).tolist()
        summary["suspicious_title_examples"] = titles[
            titles.str.contains(r"[^A-Za-z0-9\s&/\-()]", regex=True, na=False)
        ].drop_duplicates().head(10).tolist()

    if "category" in df.columns:
        categories = df["category"].dropna().astype(str).str.strip()
        summary["unique_categories"] = int(categories.nunique())
        summary["category_examples"] = categories.drop_duplicates().head(10).tolist()

    if "attributes" in df.columns:
        attrs = df["attributes"].dropna().astype(str).str.strip()
        summary["unique_attributes"] = int(attrs.nunique())
        summary["attribute_examples"] = attrs.drop_duplicates().head(10).tolist()
        summary["malformed_attribute_examples"] = attrs[
            ~attrs.str.contains("=", regex=False, na=False)
        ].drop_duplicates().head(10).tolist()

    if "description" in df.columns:
        desc = df["description"].dropna().astype(str).str.strip()
        summary["description_examples"] = desc.head(10).tolist()
        summary["noisy_description_examples"] = desc[
            desc.str.contains(r"[âà¸]|\?{2,}", regex=True, na=False)
        ].drop_duplicates().head(10).tolist()

    if "price" in df.columns:
        price_num = pd.to_numeric(df["price"], errors="coerce")
        summary["invalid_price_count"] = int(price_num.isna().sum())
        summary["non_positive_price_count"] = int((price_num <= 0).sum())

    if "cost" in df.columns:
        cost_num = pd.to_numeric(df["cost"], errors="coerce")
        summary["invalid_cost_count"] = int(cost_num.isna().sum())
        summary["non_positive_cost_count"] = int((cost_num <= 0).sum())

    if "price" in df.columns and "cost" in df.columns:
        price_num = pd.to_numeric(df["price"], errors="coerce")
        cost_num = pd.to_numeric(df["cost"], errors="coerce")
        summary["price_below_cost_count"] = int((price_num < cost_num).sum())

    return json.dumps(summary, ensure_ascii=False, indent=2)

@tool
def llm_normalize_values(input_json: str) -> str:
    """Normalize unique values for one column and return a JSON mapping."""
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

    try:
        payload = json.loads(input_json)
    except json.JSONDecodeError:
        return "{}"

    column = payload.get("column", "")
    rules = payload.get("rules", [])
    values = payload.get("values", [])

    if not isinstance(values, list) or not values:
        return "{}"

    values = [str(v).strip() for v in values if str(v).strip()]
    values = list(dict.fromkeys(values))[:100]

    prompt = f"""
    You clean ecommerce catalog values.

    Column: {column}

    Rules:
    {chr(10).join(f"- {r}" for r in rules)}

    Return ONLY ONE valid JSON object.
    No markdown.
    No explanations.
    No text before or after JSON.

    Format:
    {{"original value": "cleaned value"}}

    Requirements:
    - preserve meaning
    - keep the same number of unique inputs
    - if uncertain, keep the original

    Input values:
    {json.dumps(values, ensure_ascii=False)}
    """.strip()

    try:
        response = llm.invoke(prompt).content.strip()
        return response.replace("```json", "").replace("```", "").strip()
    except Exception:
        return "{}"
    

@tool
def clean_catalog(file_path: str) -> str:
    """ Clean catalog data, normalize categories, and save the cleaned CSV into the results folder. """
    file_path = file_path.strip()
    df = pd.read_csv(file_path)
    cleaned_df = df.copy()

    # Normalize column names
    cleaned_df.columns = cleaned_df.columns.str.strip().str.lower()

    # Strip whitespace safely
    for col in cleaned_df.select_dtypes(include="object").columns:
        cleaned_df[col] = cleaned_df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    # Clean price
    if "price" in cleaned_df.columns:
        cleaned_df["price"] = cleaned_df["price"].apply(convert_price).astype("float64")

    # Clean cost
    if "cost" in cleaned_df.columns:
        cleaned_df["cost"] = cleaned_df["cost"].apply(convert_price).astype("float64")

    if "title" in cleaned_df.columns:
        cleaned_df = apply_llm_mapping(
            cleaned_df,
            "title",
            [
                "fix obvious spelling mistakes and rewrite it correctly",
                "merge values with the same or similar meanin INTO ONE TITLE",
                "remove unnecessary symbols",
                "improve formatting",
                "preserve product meaning"
            ]
        )

    if "attributes" in cleaned_df.columns:
        cleaned_df = apply_llm_mapping(
            cleaned_df,
            "attributes",
            [
                "standardize attribute formatting as key=value pairs separated by semicolons",
                "expand obvious abbreviations when clear",
                "remove noisy symbols like ??",
                "normalize spaces around units such as 1L to 1 L",
                "preserve meaning",
                "if uncertain, keep the original"
            ]
        )

    original_unique_categories = 0
    normalized_unique_categories = 0

    if "category" in cleaned_df.columns:
        original_unique_categories = cleaned_df["category"].nunique(dropna=True)

        cleaned_df = apply_llm_mapping(
            cleaned_df,
            "category",
            [
                "merge values with the same or similar meaning into ONE category without duplication",
                "keep names lowercase",
                "do not invent unrelated categories"
            ]
        )

        normalized_unique_categories = cleaned_df["category"].nunique(dropna=True)

    original_unique_descriptions = 0
    normalized_unique_descriptions = 0

    if "description" in cleaned_df.columns:
        cleaned_df["description"] = cleaned_df["description"].apply(
            lambda x: text_cleaning(x) if pd.notna(x) else x
        )

        # mark invalid descriptions as missing
        cleaned_df["description"] = cleaned_df["description"].apply(
            lambda x: pd.NA
            if pd.notna(x) and (
                len(str(x).strip()) < 8 or
                str(x).strip().isdigit() or
                not re.search(r"[A-Za-z]", str(x))
            )
            else x
        )

        original_unique_descriptions = cleaned_df["description"].nunique(dropna=True)

        cleaned_df = apply_llm_mapping(
            cleaned_df,
            "description",
            [
                "clean and improve the product description text",
                "fix obvious spelling mistakes",
                "remove noisy symbols and broken encoded text",
                "remove uncertainty or review-like wording when it does not belong in a product description",
                "preserve product meaning",
                "do not invent product specifications or features",
                "if the description is too ambiguous, keep it close to the original cleaned text"
            ]
        )

        cleaned_df["description"] = cleaned_df["description"].apply(
            lambda x: pd.NA
            if pd.notna(x) and (
                len(str(x).strip()) < 8 or
                str(x).strip().isdigit() or
                not re.search(r"[A-Za-z]", str(x))
            )
            else x
        )

        normalized_unique_descriptions = cleaned_df["description"].nunique(dropna=True)

    # Create results directory
    os.makedirs("results", exist_ok=True)

    for col in ["title", "attributes", "category", "description"]:
        original_col = f"{col}_original"
        if original_col in cleaned_df.columns:
            mapping_df = cleaned_df[[original_col, col]].drop_duplicates()
            mapping_df.to_csv(f"results/{col}_mapping.csv", index=False)
            cleaned_df = cleaned_df.drop(columns=[original_col])

    texts = ["attributes", "category", "description"]

    for col in texts:
        if col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].apply(
                lambda x: pd.NA if pd.isna(x) or str(x).strip() == "" else x
            )

    cleaned_df = cleaned_df.drop_duplicates(subset=["title", "category", "price", "cost"], keep="first")

    output_path = f"results/cleaned_catalog.csv"
    cleaned_df.to_csv(output_path, index=False)

    missing_prices = cleaned_df["price"].isna().sum() if "price" in cleaned_df.columns else 0
    missing_costs = cleaned_df["cost"].isna().sum() if "cost" in cleaned_df.columns else 0
    missing_categories = cleaned_df["category"].isna().sum() if "category" in cleaned_df.columns else 0
    missing_attributes = cleaned_df["attributes"].isna().sum() if "attributes" in cleaned_df.columns else 0
    missing_descriptions = cleaned_df["description"].isna().sum() if "description" in cleaned_df.columns else 0

    return (
        f"Catalog cleaned successfully.\n"
        f"Missing prices after cleaning: {missing_prices}\n"
        f"Missing costs after cleaning: {missing_costs}\n"
        f"Missing categories after cleaning: {missing_categories}\n"
        f"Missing attributes after cleaning: {missing_attributes}\n"
        f"Missing descriptions after cleaning: {missing_descriptions}\n"
        f"Original unique categories: {original_unique_categories}\n"
        f"Normalized unique categories: {normalized_unique_categories}\n"
        f"Cleaned file saved to: {output_path}\n"
        f"Final row count: {len(cleaned_df)}"
    )