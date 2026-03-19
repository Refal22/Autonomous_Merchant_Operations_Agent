# Report

## Explanation of Design Decisions
The project was designed as a **modular agent-based system** to ensure flexibility, maintainability, and ease of debugging. 
Each core business function: 
* Catalog analysis
* Customer support analysis
* Pricing recommendations.

Was all implemented as a separate tool. This separation allowed each component to be developed, tested, and improved independently without affecting the rest of the system

**LangChain** was used to orchestrate the agent workflow and manage tool interactions. This choice enables structured reasoning, controlled execution orders, and easy integration of multiple tools within a single agent.

For the LLM, Groq (Llama-3.3-70b-versatile) was selected after testing multiple models. The decision was based on its strong performance in:
- Structured reasoning
- Optimized and consistent results
- Ability to follow strict instructions and formatting rules

Additionally, Groq provides fast inference, which improves the responsiveness of the system.

**Streamlit** was added as a lightweight interface to make the system accessible to non-technical users.
This allows the agent to be executed by both:
- Via command line (for development and debugging)
- Via a web interface (for usability and demonstration).

**LangSmith** was used for monitoring and debugging. It provides visibility into:
- Tool calls
- Intermediate reasoning steps
- failure points

This significantly improved the ability to diagnose issues such as formatting errors and incorrect tool usage.

Finally, **LLM-based** processing was selectively used in some tools for tasks that require semantic understanding, such as:
- Normalizing inconsistent text values
- Classifying and analyzing customer messages
- Generate explanations and reports

This hybrid approach (rule-based + LLM) ensures both reliability and flexibility.

## Debugging Examples

