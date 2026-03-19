import os
import time
import tempfile
import pandas as pd
import streamlit as st
from Agent.agent import create_agent

def save_markdown_from_agent(response, path="results/daily_report.md"):
    os.makedirs("results", exist_ok=True)
    text = response.get("output", "") if isinstance(response, dict) else str(response)
    marker = "# Daily Merchant Final Report"
    # extract markdown part only
    if marker in text:
        text = text[text.index(marker):]
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path

st.set_page_config(page_title="Agent Interface", layout="wide")

# -----------------------
# Session state
# -----------------------
if "page" not in st.session_state:
    st.session_state.page = "Catalog Analysis"

if "outputs" not in st.session_state:
    st.session_state.outputs = {
        "Catalog Analysis": "No output yet.",
        "Customers": "No output yet.",
        "Recommendations": "No output yet.",
        "Report": "No output yet."
    }

if "step_status" not in st.session_state:
    st.session_state.step_status = {
        "Load Data": "idle",
        "Catalog Analysis": "idle",
        "Customers": "idle",
        "Recommendations": "idle",
        "Report": "idle"
    }

if "catalog_df" not in st.session_state:
    st.session_state.catalog_df = None

if "customer_df" not in st.session_state:
    st.session_state.customer_df = None

if "catalog_path" not in st.session_state:
    st.session_state.catalog_path = None

if "customer_path" not in st.session_state:
    st.session_state.customer_path = None

# -----------------------
# Sidebar
# -----------------------
with st.sidebar:
    st.markdown("## Agent Interface")
    st.markdown("### Pages")

    if st.button("📦 Catalog Analysis", use_container_width=True):
        st.session_state.page = "Catalog Analysis"
    if st.button("👥 Customers Messages Analysis", use_container_width=True):
        st.session_state.page = "Customers Messages Analysis"
    if st.button("🤖 Recommendations", use_container_width=True):
        st.session_state.page = "Recommendations"
    if st.button("📑 Daily Report", use_container_width=True):
        st.session_state.page = "Daily Report"

page = st.session_state.page

# -----------------------
# Helpers
# -----------------------
def save_uploaded_file(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1] or ".csv"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name

def safe_for_streamlit(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str)
    return df

def show_steps():
    icons = {"idle": "⚪", "running": "🔄", "done": "✅"}
    with step_box:
        st.markdown("### Process")
        for step, status in st.session_state.step_status.items():
            st.write(f"{icons[status]} {step}")

def run_step(step, output=None):
    st.session_state.step_status[step] = "running"
    show_steps()
    time.sleep(0.3)
    st.session_state.step_status[step] = "done"
    if output is not None:
        st.session_state.outputs[step] = output
    show_steps()

# -----------------------
# Main interface
# -----------------------
st.title("Autonomous Merchant Operations Agent")

catalog_file = st.file_uploader("Upload Catalog CSV", type="csv", key="catalog")
customer_file = st.file_uploader("Upload Customer Messages CSV", type="csv", key="customers")

step_box = st.container()
show_steps()

# -----------------------
# Load previews
# -----------------------
if catalog_file is not None:
    try:
        catalog_file.seek(0)
        st.session_state.catalog_df = pd.read_csv(catalog_file)
        catalog_file.seek(0)
    except Exception as e:
        st.error(f"Could not read catalog CSV: {e}")

if customer_file is not None:
    try:
        customer_file.seek(0)
        st.session_state.customer_df = pd.read_csv(customer_file)
        customer_file.seek(0)
    except Exception as e:
        st.error(f"Could not read customer CSV: {e}")

if catalog_file is None and customer_file is None:
    st.info("Please upload at least one CSV file to begin.")

else:
    if st.button("Run Agent"):
        for k in st.session_state.step_status:
            st.session_state.step_status[k] = "idle"

        run_step("Load Data", "Uploaded files loaded successfully.")

        catalog_path = None
        customer_path = None

        if catalog_file is not None:
            catalog_file.seek(0)
            catalog_path = save_uploaded_file(catalog_file)
            st.session_state.catalog_path = catalog_path

        if customer_file is not None:
            customer_file.seek(0)
            customer_path = save_uploaded_file(customer_file)
            st.session_state.customer_path = customer_path

        agent = create_agent()

        prompt = f"""
You are an autonomous merchant operations agent.

    Use the available tools to inspect and clean this catalog file: {catalog_path if catalog_path else "Not provided"}.

    You may only use these exact tool names:
    - parse_catalog
    - clean_catalog
    - llm_normalize_values
    - llm_customer_messages_analysis
    - recommend_price_tool

    Do not translate tool names.
    Do not invent tool names.

    When using tools, you MUST follow exactly this format:

    Thought: your reasoning
    Action: one of [parse_catalog, clean_catalog, llm_normalize_values, llm_customer_messages_analysis, recommend_price_tool]
    Action Input: the exact tool input

    After a tool returns, continue with:

    Observation: the tool result

    Important formatting rules:
    - Do not write anything between Thought and Action.
    - Do not skip Action after Thought.
    - Do not use backticks around tool names.
    - Do not use markdown code blocks for tool calls.
    - If no more tool calls are needed, write:
    Final Answer: ...

    WORKFLOW ORDER IS MANDATORY

    1. Start with parse_catalog on the original catalog CSV file only.
    2. Use the parse_catalog output to detect and list concrete catalog problems before doing any cleaning (you MUST mention all wrong examples)
    3. If you detect inconsistent text values in title, category, or attributes:
    - list the inconsistent values and wrong examples
    - use llm_normalize_values on unique values only
    4- If you detect invalid numeric values in price or cost you should mention it.
    4. After identifying needed fixes, you MUST call clean_catalog on the original catalog CSV file.
    5. Immediately after clean_catalog is completed, you MUST write the Catalog Analysis Report.
    6. Do not move to customer support analysis until the Catalog Analysis Report is complete.
    7. After the Catalog Analysis Report is written, you MUST call llm_customer_messages_analysis on {customer_path if customer_path else "Not provided"}.
    8. Do not skip llm_customer_messages_analysis.
    9. After customer message analysis is completed, you MUST call recommend_price_tool using the cleaned catalog file and the analyzed customer messages file.
    10. After catalog analysis, customer support analysis, and price recommendations are all completed, produce ONE final daily report in Markdown.

    CATALOG PROBLEMS TO DETECT

    Problems may include:
    - missing values
    - duplicate rows
    - duplicate product IDs
    - suspicious or repeated titles that have similar or one meaning and should be treated as one
    - inconsistent or repeated categories that should be treated as one
    - malformed, unclear, or misspelled attributes
    - noisy, missing, unclear, or inappropriate descriptions
    - descriptions containing customer opinion
    - invalid, missing, or text-based prices/costs
    - price lower than cost

    DESCRIPTION QUALITY CHECK

    You must carefully inspect the "description" column and detect ALL description-related problems.

    Description problems include:
    - missing descriptions
    - descriptions that are empty after cleaning
    - descriptions shorter than a meaningful sentence
    - descriptions containing only numbers or very few characters
    - descriptions containing uncertainty words such as:
    "maybe", "might", "possibly", "could be"
    - descriptions containing customer feedback or complaints
    - descriptions that do not describe the product itself
    - descriptions that appear corrupted or partially removed during cleaning

    For every detected issue in the description column, you MUST:
    - list the product_id
    - show the problematic description value
    - explain why the description is problematic
    - state whether the issue was automatically cleaned or requires merchant review

    CATALOG ANALYSIS REPORT INSTRUCTIONS

    Immediately after clean_catalog is completed, produce a merchant-facing Catalog Analysis Report.

    This report must be factual, structured, and based only on the tool outputs.

    IMPORTANT RULES:
    - Mention ALL detected catalog problems with all cases found.
    - Do not ask the merchant to fix issues that were already corrected automatically.
    - Clearly separate:
    1. all problems detected
    2. fixes performed automatically
    3. unresolved issues requiring merchant review
    - Always include missing descriptions, missing categories, missing attributes, missing prices, and missing costs if they exist.
    - Always include ALL problematic descriptions detected in the "description" column.
    - If duplicate or near-duplicate products are found, report them as possible duplicate listings caused by inconsistent data entry.
    - Do not invent product information or recommended replacements unless clearly supported by the data.
    - If a value cannot be safely inferred, say that merchant review is required.

    Use EXACTLY this structure for the Catalog Analysis Report:

    SECTION 1 — Catalog Problems Detected
    List every detected issue grouped by type.
    For each issue include:
    - affected column
    - explanation
    - concrete example values
    - why this issue matters

    SECTION 2 — Automatically Fixed by the Agent

    List only the issues that were explicitly fixed by the cleaning process according to the actual tool outputs.

    Rules:
    - Do NOT claim that missing values were removed unless the tool output explicitly says rows with missing values were removed.
    - Do NOT say that rows were removed because of missing or invalid values unless this is directly supported by the clean_catalog output.
    - If duplicates were removed, say that duplicates were removed.
    - If categories were normalized, say that categories were normalized.
    - If attributes, titles, or descriptions were cleaned, mention only those fixes that were actually performed.

    Use only facts directly supported by the clean_catalog result.

    SECTION 3 — Missing Critical Information

    Before writing this section:
    - You MUST inspect the cleaned catalog data after clean_catalog.
    - You MUST use the cleaned dataset only, not the raw dataset.
    - Note that missings might be representes as "null", "NaN" or " " 

    Rules:
    - Only list fields that are STILL missing in the cleaned dataset.
    - You MUST include ALL remaining missing critical fields.
    - You MUST explicitly check:
    - category
    - attributes
    - description
    - price
    - cost
    - Empty strings must be treated as missing.
    - Very short, numeric-only, or unusable descriptions must be treated as missing.
    - If a field has no remaining missing values, do NOT mention it.

    For each missing field, include:
    - field name
    - exact number of affected records

    Important:
    - Do not skip missing textual fields if they still exist.

    SECTION 4 — Data Entry Guidance
    Give short practical recommendations to prevent these issues in future.
    Examples:
    - use one consistent category naming format
    - avoid speculative wording in product descriptions
    - do not include customer complaints in product descriptions
    - ensure every product has category, attributes, description, price, and cost
    - avoid duplicate listings for the same product

    CUSTOMER SUPPORT ANALYSIS RULES

    After the Catalog Analysis Report is written, you MUST analyze customer support messages from the CSV file {customer_path if customer_path else "Not provided"} using llm_customer_messages_analysis.

    Do not skip this tool.
    You MUST use it before recommend_price_tool.
    Use only the exact tool name.

    PRICING RULES

    After customer message analysis is completed, you MUST call recommend_price_tool using:
    - the cleaned catalog file
    - the analyzed customer messages file

    FINAL DAILY REPORT INSTRUCTIONS

    After completing catalog analysis, customer support analysis, and price recommendations, produce ONE final daily report in Markdown.

    The daily report is different from the Catalog Analysis Report.
    The daily report must combine all of the above into a single structured daily report for the merchant.

    Use EXACTLY this structure:

    # Daily Merchant Final Report

    ## Executive Summary
    Briefly summarize the most important findings from the tools you used. WITHOUT mentioning tha action.
    Just summarize usefull insights

    ## Red Alerts
    List only the most critical issues that require immediate attention (with mentioning at least ONE example)

    ## Yellow Alerts
    List medium-priority issues that should be reviewed soon (with mentioning at least ONE example)

    ## Catalog Issues
    Summarize the most important catalog data problems still affecting the business (with mentioning at least ONE example)

    ## Customer Support Insights
    Summarize the main patterns of the analysis and list usefull insights.

    ## Pricing Recommendations
    List the recommended pricing actions for EACH product with clear AND short explanations.
    You MUST mention examples

    ## Recommended Merchant Actions
    List the most important next actions the merchant should take.

    FINAL RULES
    - Keep noise low.
    - Mention only the most important issues.
    - Be specific and actionable.
    - Use only the tool outputs.
    - Write the final answer in Markdown.
    - Never invent product information.
    - Only correct obvious errors and formatting issues.
    - Do NOT mention the used tools. this is a merchant report.

    STRICT TOOL FORMAT RULE:

    When using a tool, you MUST follow exactly this format:

    Thought: your reasoning
    Action: one of [parse_catalog, clean_catalog, llm_normalize_values, llm_customer_messages_analysis, recommend_price_tool]
    Action Input: the exact tool input

    Do not write anything between Thought and Action.
    Do not skip Action after Thought.
    Do not write Observation yourself.

    If no tool is needed, write only:
    Final Answer: ...


"""
        result = agent.invoke({"input": prompt})
        save_markdown_from_agent(result)

        try:
            result = agent.invoke({"input": prompt})
            final_output = result["output"]

            if catalog_path:
                run_step("Catalog Analysis", final_output)
            if customer_path:
                run_step("Customers Messages Analysis", final_output)
            if catalog_path and customer_path:
                run_step("Recommendations", final_output)
            else:
                st.session_state.step_status["Recommendations"] = "done"
            run_step("Report", final_output)

            st.session_state.outputs["Catalog Analysis"] = final_output
            st.session_state.outputs["Customers Messages Analysis"] = final_output
            st.session_state.outputs["Recommendations"] = final_output
            st.session_state.outputs["Report"] = final_output

            st.success("Agent completed successfully.")

        except Exception as e:
            st.error(f"Agent failed: {e}")

# -----------------------
# Selected page output
# -----------------------
st.markdown("---")
st.header(page)

catalog_df = st.session_state.catalog_df
customer_df = st.session_state.customer_df

if page == "Catalog Analysis":
    results_path22 = "results/cleaned_catalog.csv"
    if os.path.exists(results_path22):
        try:
            rec_df22 = pd.read_csv(results_path22)
            st.subheader("Cleaned Catalog Data")
            st.dataframe(safe_for_streamlit(rec_df22))
        except Exception as e:
            st.warning(f"Could not load recommendations file: {e}")

    if catalog_df is not None:
        st.subheader("Catalog Preview (Before Cleaning)")
        st.dataframe(safe_for_streamlit(catalog_df.head()))


elif page == "Customers Messages Analysis":
    results_path2 = "results/customer_messages_analysis.csv"
    if os.path.exists(results_path2):
        try:
            rec_df = pd.read_csv(results_path2)
            st.subheader("Customers Messages Analysis")
            st.dataframe(safe_for_streamlit(rec_df))
        except Exception as e:
            st.warning(f"Could not load recommendations file: {e}")

    if customer_df is not None:
        st.subheader("Customer Messages Preview (Before cleaning)")
        st.dataframe(safe_for_streamlit(customer_df.head()))

        if "category" in customer_df.columns:
            st.subheader("Message Categories")
            counts_df = customer_df["category"].value_counts(dropna=False).reset_index()
            counts_df.columns = ["category", "count"]
            st.dataframe(safe_for_streamlit(counts_df))

elif page == "Recommendations":
    #st.write(st.session_state.outputs["Recommendations"])

    price_results_path = "results/recommended_prices.csv"
    if os.path.exists(price_results_path):
        try:
            rec_df = pd.read_csv(price_results_path)
            rec_df = rec_df.drop(columns=['id_x', 'id_y','clean_message'], errors='ignore')
            rec_df = rec_df.loc[:, ~rec_df.columns.str.contains('id_|clean_message')]
            st.subheader("Price Recommendations")
            st.dataframe(safe_for_streamlit(rec_df))
        except Exception as e:
            st.warning(f"Could not load recommendations file: {e}")


elif page == "Report":
    report = st.session_state.outputs["Report"]

    # Colorize sections
    report = report.replace(
        "## Red Alerts",
        "## <span style='color:red'>Red Alerts</span>"
    ).replace(
        "## Yellow Alerts",
        "## <span style='color:orange'>Yellow Alerts</span>"
    )

    st.markdown(report, unsafe_allow_html=True)

    daily_report_path = "results/daily_report.md"
    if os.path.exists(daily_report_path):
        try:
            with open(daily_report_path, "r", encoding="utf-8") as f:
                saved_report = f.read()

                saved_report = saved_report.replace(
                    "## Red Alerts",
                    "## <span style='color:red'>Red Alerts</span>"
                ).replace(
                    "## Yellow Alerts",
                    "## <span style='color:orange'>Yellow Alerts</span>"
                )

                st.subheader("Saved Daily Report")
                st.markdown(saved_report, unsafe_allow_html=True)

        except Exception as e:
            st.warning(f"Could not load daily report: {e}")