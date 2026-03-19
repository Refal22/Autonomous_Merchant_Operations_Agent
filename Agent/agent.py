import os
from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
from langchain_groq import ChatGroq
from tools.catalog_analysis import parse_catalog, clean_catalog, llm_normalize_values
from tools.customer_support_analysis import llm_customer_messages_analysis
from tools.price_recommendation import recommend_price_tool

load_dotenv()

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

def create_agent():
    llm = ChatGroq(
        # model="llama-3.1-8b-instant" , "llama-3.3-70b-versatile", "openai/gpt-oss-120b"
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

    tools = [parse_catalog, clean_catalog, llm_normalize_values, llm_customer_messages_analysis, recommend_price_tool]

    agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True)

    return agent

if __name__ == "__main__":
    agent = create_agent()

    products_data = "Data/products_raw.csv"
    customers_data = "Data/customer_messages.csv"

    prompt = f"""

    You are an autonomous merchant operations agent.

    Use the available tools to inspect and clean this catalog file: {products_data}.

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
    7. After the Catalog Analysis Report is written, you MUST call llm_customer_messages_analysis on {customers_data}.
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

    After the Catalog Analysis Report is written, you MUST analyze customer support messages from the CSV file {customers_data} using llm_customer_messages_analysis.

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
    final_output = result["output"]

    print("\n=== AGENT RESULT ===")

    save_markdown_from_agent(result)

    print(final_output)
    print(result)
