# crewAI-trip-planner

A smart, agent-based trip planner leveraging AI and OpenAI's GPT-4 for generating and managing travel itineraries. Built with Python and Streamlit, this project automates and optimizes trip planning with advanced reasoning, real-time data, and modular agents.

---

## 🌐 Try the Live App

Access the deployed Streamlit app here:  
**[Trip Planner on Streamlit](https://appapppy-dlyihw7zvuupplcvgwnf2b.streamlit.app)**

---

## Features

- **Agent-based planning:** Modular agents (city selection, itinerary, budget, transport, etc.) coordinate to create and refine trip plans.
- **Integration with OpenAI:** Utilizes GPT-4 for reasoning and retrieval-augmented generation (RAG).
- **Real-time data:** Integrates APIs for weather, events, currency, transport, and more.
- **Guardrails:** Input/output validation and business logic enforcement for robust plans.
- **Streamlit Web App:** User-friendly, interactive interface for planning trips.
- **Extensible tooling:** Easily add custom tasks, agents, and tools.
- **Observability:** Built-in telemetry and tracing support with Phoenix integration.

---

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/Hrishitcodethis/crewAI-trip-planner.git
   cd crewAI-trip-planner
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Set up API keys:**
   - Add your API keys in a `.env` file (for local use) or via Streamlit secrets (for deployment).
   - Required: `OPENAI_API_KEY`, and keys for OpenTripMap, WeatherAPI, Currency API, Eventbrite, AviationStack, TransitLand, etc.

4. **Configure Telemetry (Optional):**
   - For Phoenix tracing:
     ```sh
     export PHOENIX_ENABLED=true
     export PHOENIX_CLIENT_HEADERS="api_key=your_phoenix_api_key"
     ```
   - If not configured, the app will use console tracing by default.

---

## Usage

- **Web App (Recommended):**  
  Visit the live app: [Trip Planner on Streamlit](https://appapppy-dlyihw7zvuupplcvgwnf2b.streamlit.app)

- **Local Development:**  
  ```sh
  streamlit run streamlit_app.py
  ```
  or
  ```sh
  streamlit run trip_planner/app.py
  ```

- **Command-line (Advanced/Legacy):**  
  ```sh
  python main.py
  ```
  (For CLI-based planning, not recommended for most users.)

---

## Project Structure

- `streamlit_app.py` — Minimal entry point for Streamlit, imports and runs the main app.
- `trip_planner/app.py` — Main Streamlit app logic and UI.
- `trip_planner/agents.py` — Agent definitions and orchestration (city selection, itinerary, etc.).
- `trip_planner/tasks.py` — Task prompt templates for agents.
- `trip_planner/guardrails.py` — Input/output/business rule validation.
- `trip_planner/tools/` — Modular tools for travel, search, and calculations.
- `trip_planner/telemetry.py` — Telemetry and tracing configuration.
- `requirements.txt` — Dependency management.
- `.streamlit/secrets.toml` — (Not committed) for API keys on Streamlit Cloud.
- `main.py` — CLI-based planning (legacy/advanced use).
- `langgraph_rag_agent_llama3_local (1).ipynb` — (Optional) Notebook for prototyping.

---

## Requirements

- Python 3.11+
- pip
- API keys for OpenAI and other integrated services
- (Optional) Phoenix API key for advanced tracing

---

## Contributing

Pull requests and issues are welcome! Please open an issue for bug reports or feature suggestions.

---

## License

[MIT](LICENSE)

---

> **Note:** For a full list of files, visit the [project directory](https://github.com/Hrishitcodethis/crewAI-trip-planner/tree/main/).
