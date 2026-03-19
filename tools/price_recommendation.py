import pandas as pd
import numpy as np
import json, re, os
from langchain_groq import ChatGroq
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()

def build_product_signals(products_df: pd.DataFrame, messages_df: pd.DataFrame) -> pd.DataFrame:
    products_df = products_df.copy()
    messages_df = messages_df.copy()

    def norm(x):
        if pd.isna(x):
            return ""
        return str(x).strip().lower()
    # products
    products_df["title_norm"] = products_df["title"].apply(norm)
    products_df["price"] = pd.to_numeric(products_df["price"], errors="coerce")
    products_df["cost"] = pd.to_numeric(products_df["cost"], errors="coerce")
    products_df["margin"] = products_df["price"] - products_df["cost"]
    products_df["margin_pct"] = np.where(
        products_df["price"] > 0,
        (products_df["price"] - products_df["cost"]) / products_df["price"],
        np.nan
    )

    # messages
    messages_df["product_norm"] = messages_df["product"].apply(norm)
    messages_df["category_norm"] = messages_df["category"].apply(norm)

    if "sentiment" in messages_df.columns:
        messages_df["sentiment_norm"] = messages_df["sentiment"].apply(norm)
    else:
        messages_df["sentiment_norm"] = ""

    # if you have deduplicated counts, use them; otherwise each row = 1
    if "count" in messages_df.columns:
        messages_df["weight"] = pd.to_numeric(messages_df["count"], errors="coerce").fillna(1)
    elif "repeat_count" in messages_df.columns:
        messages_df["weight"] = pd.to_numeric(messages_df["repeat_count"], errors="coerce").fillna(1)
    else:
        messages_df["weight"] = 1

    messages_df = messages_df[messages_df["product_norm"] != ""].copy()

    # aggregate message signals
    agg = (
        messages_df.groupby("product_norm")
        .agg(
            total_messages=("weight", "sum"),
            negative_count=("weight", lambda x: x[messages_df.loc[x.index, "sentiment_norm"] == "negative"].sum()),
            positive_count=("weight", lambda x: x[messages_df.loc[x.index, "sentiment_norm"] == "positive"].sum()),
            neutral_count=("weight", lambda x: x[messages_df.loc[x.index, "sentiment_norm"] == "neutral"].sum()),
            complaint_count=("weight", lambda x: x[messages_df.loc[x.index, "category_norm"] == "complaint"].sum()),
            suggestion_count=("weight", lambda x: x[messages_df.loc[x.index, "category_norm"] == "suggestion"].sum()),
        )
        .reset_index()
    )

    # merge with products
    result = products_df.merge(
        agg,
        left_on="title_norm",
        right_on="product_norm",
        how="left"
    )

    fill_cols = [
        "total_messages",
        "negative_count",
        "positive_count",
        "neutral_count",
        "complaint_count",
        "suggestion_count"
    ]
    result[fill_cols] = result[fill_cols].fillna(0)

    return result[
        [
            "title",
            "category",
            "price",
            "cost",
            "margin",
            "margin_pct",
            "total_messages",
            "negative_count",
            "positive_count",
            "neutral_count",
            "complaint_count",
            "suggestion_count"
        ]
    ]

@tool
def recommend_price_tool(input_json: str) -> str:
    """ Generate price recommendations from cleaned catalog data and analyzed customer messages.
    Accepts either:
    - a plain products CSV path
    - or a JSON string
    """

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

    input_json = input_json.strip()

    if input_json.lower().endswith(".csv") and not input_json.startswith("{"):
        payload = {
            "products_csv": input_json,
            "messages_csv": "results/customer_messages_analysis.csv",
            "output_csv": "results/price_results.csv"
        }
    else:
        payload = json.loads(input_json)

    products_csv = payload.get("products_csv", "results/cleaned_catalog.csv")
    messages_csv = payload.get("messages_csv", "results/customer_messages_analysis.csv")
    output_csv = payload.get("output_csv", "results/price_results.csv")

    os.makedirs("results", exist_ok=True)

    products_df = pd.read_csv(products_csv)
    messages_df = pd.read_csv(messages_csv)

    signals_df = build_product_signals(products_df, messages_df)

    results = []

    for _, row in signals_df.iterrows():
        print("processing output:", row["title"])

        data = {
            "product": row["title"],
            "price": None if pd.isna(row["price"]) else float(row["price"]),
            "cost": None if pd.isna(row["cost"]) else float(row["cost"]),
            "total_messages": int(row["total_messages"]),
            "complaint_count": int(row["complaint_count"]),
            "suggestion_count": int(row["suggestion_count"]),
            "negative_count": int(row["negative_count"]),
            "neutral_count": int(row["neutral_count"]),
            "positive_count": int(row["positive_count"]),
            "margin": None if pd.isna(row["margin"]) else float(row["margin"]),
            "margin_pct": None if pd.isna(row["margin_pct"]) else float(row["margin_pct"])
        }

        prompt = f"""
Decide the recommened pricing action based on the provided data.

For each product, you should choose one of the following actions:

Increase: if suggestions and positive sentiment are increasing, recommend price increasing with mentioing a new price.
Decrease: if complaints and negative sentiment are increasing, recommend price decrease mentioing a new price.
Hold: if natural sentiment are exsist.
No Action/Invalid Data: if there are missing information like cost or price (were not entered by the merchant)

You have to justify the signals you used for choosing this action.

Reules:
- Do NOT recommend a price lower than the cost.
- Do NOT recommend a price increase when negative sentiment is rising.

Data:
{json.dumps(data, ensure_ascii=False, indent=2)}

Return ONLY valid JSON:
{{
    "title": "....",
    "action": "...",
    "reason": "...",
    "new_price": "...."
}}
""".strip()

        response = llm.invoke(prompt).content.strip()
        print(response)
        response = response.replace("```json", "").replace("```", "").strip()

        start = response.find("{")
        end = response.rfind("}")
        if start != -1 and end != -1:
            clear_json = response[start:end + 1]
        else:
            clear_json = ""

        try:
            result = json.loads(clear_json)
        except Exception as e:
            print("error:", row["title"], e)
            result = {
                "title": row["title"],
                "action": "hold",
                "reason": f"LLM parsing failed: {str(e)}",
                "new_price": row["price"]
            }

        results.append({
            "product": row["title"],
            "action": result.get("action"),
            "reason": result.get("reason"),
            "new_price": result.get("new_price")
        })

    results_df = pd.DataFrame(results)
    print("Final pricing rows:", len(results_df))
    print(results_df[["product", "action", "reason"]])

    results_df.to_csv(output_csv, index=False)

    return json.dumps({
        "status": "success",
        "products_csv": products_csv,
        "messages_csv": messages_csv,
        "output_csv": output_csv,
        "rows_processed": len(results_df)
    }, ensure_ascii=False, indent=2)

