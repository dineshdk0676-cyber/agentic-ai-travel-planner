# ✈️ Agentic AI Travel Planning Assistant (LangChain + Streamlit)

An AI travel agent that understands a trip request, decides which tools to call
(flights, hotels, attractions, live weather, budget), and produces a day-wise itinerary
with a cost breakdown, reasoning, and structured JSON.

## Features
- **LangChain tool-calling agent** (`AgentExecutor`) with 5 tools, multi-step reasoning, and a visible tool-call trace.
- **Tools:** Flight search (cheapest / fastest / earliest), Hotel recommendation (stars, price, best value),
  Places discovery (type, rating), Weather (free [Open-Meteo](https://open-meteo.com), no key), Budget estimation.
- **Offline planner:** the same tools with fixed rules, so the app works with **no API key**.
- Weather-aware itinerary: outdoor attractions go on the driest days.
- Outputs: readable Markdown + JSON, both downloadable.
- LLM providers: OpenAI, Google Gemini (has a free tier), Anthropic.

## Project structure
```
app.py                     Streamlit UI
travel_agent/
  config.py                city coordinates, weather codes, defaults
  data_loader.py           loads + validates the JSON datasets
  services.py              flight / hotel / place / budget logic (pure Python)
  weather.py               Open-Meteo client (forecast + historical fallback)
  tools.py                 LangChain @tool wrappers
  agent.py                 LLM factory, prompt, AgentExecutor
  planner.py               deterministic planner + Markdown renderer
data/                      flights.json, hotels.json, places.json
tests/                     pytest unit tests
.streamlit/                config + secrets example
```

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Choose **Offline planner** for a no-key demo, or **AI Agent** and enter an API key in the sidebar
(or put it in `.streamlit/secrets.toml`, see the `.example` file).

Run tests: `pip install -r requirements-dev.txt && pytest`

## Deploy to GitHub + Streamlit Community Cloud
1. Create an empty GitHub repo, then from this folder:
   ```bash
   git init && git add . && git commit -m "AI travel planner"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<repo>.git
   git push -u origin main
   ```
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → pick the repo, branch `main`, main file `app.py`.
3. **Advanced settings → Secrets**: add `OPENAI_API_KEY = "..."` (or `GOOGLE_API_KEY` / `ANTHROPIC_API_KEY`).
   Never commit keys: `.streamlit/secrets.toml` is git-ignored.
4. Deploy. Without a secret, visitors can still use the offline planner or paste their own key in the sidebar.

## Data notes
- The sample flight data has only 30 flights, so many city pairs have **no direct flight**. The app lists valid routes
  in the *Explore data* tab and explains missing flights instead of inventing them.
- Flight dates in the sample are in 2025, so trip dates rarely match; the planner then shows route flights and says so.
  Return fare is estimated as the outbound fare when no return flight exists (flagged in the output).
- Place `name` and `type` in the sample data are randomly generated (e.g. "Beautiful Park" has type "fort"). The app uses the `type` field for filtering.
- Weather: live forecast for dates within ~15 days; otherwise typical conditions from the same dates last year (labelled).
- The hotel budget assumes 1 room per 2 travellers; trip days = nights + 1; default food/local travel is ₹1,000 per person per day.
