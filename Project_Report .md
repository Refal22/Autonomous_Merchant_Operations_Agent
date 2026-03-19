# Project Report 

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
During development, several challenging issues occurred that required iterative debugging and careful investigation. Below are the most critical examples and how they were resolved

The first challenge was **prompt engineering**. It was difficult to find the right set of instructions that would consistently guide the agent to behave correctly and act as expected. An example of the key issues I encountered was that the agent would either skip required tool calls or enter loops with errors such as *"Invalid format: Missing 'Action:' after 'Thought:'"*.  
To debug this, I analyzed the agent’s intermediate reasoning steps and identified where formatting was broken. I then iteratively refined the prompt by:
- Enforced a strict tool usage format  
- Removing conflicting instructions  
- Clearly separating when to use tools vs when to produce the final answer

After multiple iterations, I achieved stable behavior where the agent followed the workflow reliably without looping.

The second major challenge was **designing a robust data cleaning tool**. The catalog data contained complex issues such as duplicates, inconsistent text, unclear values, and hidden missing/duplicated data. During debugging, I noticed that some records appeared “clean” but still caused incorrect reporting. For example, invalid descriptions or empty strings were not being treated as missing.  
To fix this, I improved the cleaning logic by:
- Normalize all missing representations (NaN, empty strings, invalid text)  
- Added validation rules to detect weak or unusable descriptions  
- Ensuring consistency before and after LLM normalization

This required multiple test runs and manual inspection of outputs until the cleaning process became reliable.

The third challenge was **deciding when to use deterministic logic versus LLM-based reasoning**. In early versions, over-reliance on LLM caused unstable outputs and inconsistencies, while purely rule-based logic failed to handle complex text variations.  
Through experimentation, I debugged this by splitting the responsibilities:
- Deterministic logic for structured operations (counts, validation, constraints)  
- LLM usage only for tasks requiring semantic understanding (normalization, classification, explanation).  
This balance improved both stability and performance while keeping the system efficient.

Overall, debugging required iterative testing, careful inspection of intermediate outputs, and continuous refinement of both prompts and tool logic until the system behaved consistently.


