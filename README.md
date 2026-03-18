# Autonomous_Merchant_Operations_Agent
This project aims to build an **Autonomous Merchant Operations Agent** that helps online merchants manage their stores with less manual effort.

## 📌 Table of Contents
- [Project Structure](#project-structure)
- [Setup](#setup)
- [How to Run](#how-to-run)
- [Outputs](#outputs)
- [Assumptions & Notes](#assumptions--notes)

## Project Structure
```
project/
│
├── Agent/
│ └── agent.py # Main agent (entry point)
│
├── tools/ # The core (all the tools used by the agent)
│ ├── catalog_analysis.py # Catalog parsing, cleaning and analyzing
│ ├── customer_support_analysis.py # Analyzing customers messsages 
│ └── price_recommendation.py # Pricing adjustments
│
├── Data/ # Input datasets
│
├── results/ # Generated outputs
│
├── App.py # Streamlit UI to trigger the agent and view the report/results
├── requirements.txt # All the required packages for the project
├── .env # To set API keys
```

## Setup
1. Create a virtual environment and activate it
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Modify the `.env` file with your API keys:
```env
GROQ_API_KEY='your_groq_api_key'
LANGSMITH_API_KEY='your_langsmith_api_key'
```

## How to Run

### Option 1 — Run the Agent (CLI)
```bash
python -m Agent.agent
```
---
### Option 2 — Run the Streamlit App
```bash
streamlit run App.py
```
Then:
- Upload catalog + customer CSV files
- Click **Run Agent**
- View results in the UI

## Outputs
After running the agent, a `results/` folder will be created containing:

- `cleaned_catalog_*.csv`
- `customer_messages_analysis.csv`
- `price_results.csv`
- `*_mapping_*.csv` (normalization mappings for monitoring and further processing)
- Final **Daily Report (Markdown)**

## Sample Output

A sample daily report generated from the agent is included in the repository under:
```
sample_output/daily_report.md
```

---
