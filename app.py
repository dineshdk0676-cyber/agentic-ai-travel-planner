"""Streamlit front-end for the Agentic AI Travel Planning Assistant.

Run locally:  streamlit run app.py
"""

import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from travel_agent.config import DEFAULT_DAILY_EXPENSE_INR  # noqa: E402
from travel_agent.data_loader import (  # noqa: E402
    available_cities,
    available_routes,
    load_flights,
    load_hotels,
    load_places,
)
from travel_agent.planner import plan_trip, render_markdown  # noqa: E402

PROVIDER_DEFAULTS = {
    "OpenAI": ("gpt-4o-mini", "OPENAI_API_KEY"),
    "Google Gemini": ("gemini-2.0-flash", "GOOGLE_API_KEY"),
    "Anthropic": ("claude-3-5-haiku-latest", "ANTHROPIC_API_KEY"),
}
PLACE_TYPES = ["park", "beach", "lake", "fort", "temple", "museum", "market", "monument"]

st.set_page_config(page_title="AI Travel Planner", page_icon="✈️", layout="wide")


def get_secret_or_env(name: str) -> str:
    """Read a key from Streamlit secrets (deployment) or environment variables (local)."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:  # noqa: BLE001 - no secrets.toml present
        pass
    return os.environ.get(name, "")


def render_sidebar() -> dict:
    """Sidebar: planning mode and LLM credentials."""
    st.sidebar.title("⚙️ Settings")
    mode = st.sidebar.radio(
        "Planning mode",
        ["🤖 AI Agent (LangChain)", "⚡ Offline planner (no API key)"],
        help="The agent lets an LLM decide which tools to call. The offline planner "
        "runs the same tools with fixed rules, so it needs no API key.",
    )
    settings = {"use_agent": mode.startswith("🤖")}
    if settings["use_agent"]:
        provider = st.sidebar.selectbox("LLM provider", list(PROVIDER_DEFAULTS))
        default_model, env_name = PROVIDER_DEFAULTS[provider]
        settings["provider"] = provider
        settings["model"] = st.sidebar.text_input("Model", value=default_model)
        stored_key = get_secret_or_env(env_name)
        if stored_key:
            st.sidebar.success(f"Using {env_name} from secrets/environment.")
        settings["api_key"] = stored_key or st.sidebar.text_input(
            f"{provider} API key", type="password", help="Never committed to GitHub."
        )
    st.sidebar.caption("Weather: Open-Meteo (free, no key). Data: flights / hotels / places JSON.")
    return settings


def render_form() -> dict | None:
    """Main trip form. Returns the submitted values or None."""
    cities = available_cities()
    with st.form("trip_form"):
        col1, col2, col3, col4 = st.columns(4)
        origin = col1.selectbox("From", cities, index=cities.index("Delhi"))
        destination = col2.selectbox("To", cities, index=cities.index("Goa"))
        start_date = col3.date_input("Start date", value=date.today() + timedelta(days=14), min_value=date.today())
        days = col4.slider("Trip length (days)", 2, 7, 3)

        col5, col6, col7, col8 = st.columns(4)
        travelers = col5.number_input("Travellers", 1, 10, 1)
        min_stars = col6.slider("Minimum hotel stars", 1, 5, 3)
        flight_sort = col7.selectbox("Flight preference", ["cheapest", "fastest", "earliest"])
        hotel_sort = col8.selectbox("Hotel preference", ["best_value", "cheapest", "highest_rated"])

        col9, col10 = st.columns(2)
        place_types = col9.multiselect("Attraction types (optional)", PLACE_TYPES)
        budget_limit = col10.number_input("Total budget cap in ₹ (0 = none)", 0, 1_000_000, 0, step=1000)
        daily_expense = st.slider(
            "Food & local travel per person per day (₹)", 300, 5000, DEFAULT_DAILY_EXPENSE_INR, step=100
        )
        extra_notes = st.text_area("Anything else? (agent mode)", placeholder="e.g. I love beaches and prefer a hotel with a pool")
        submitted = st.form_submit_button("✨ Plan my trip", type="primary", use_container_width=True)

    if not submitted:
        return None
    return {
        "origin": origin,
        "destination": destination,
        "start_date": start_date,
        "days": days,
        "travelers": int(travelers),
        "min_stars": min_stars,
        "flight_sort": flight_sort,
        "hotel_sort": hotel_sort,
        "place_types": place_types,
        "budget_limit": budget_limit or None,
        "daily_expense": daily_expense,
        "extra_notes": extra_notes,
    }


def run_offline(form: dict) -> dict:
    """Plan without an LLM."""
    plan = plan_trip(
        origin=form["origin"],
        destination=form["destination"],
        start_date=form["start_date"],
        days=form["days"],
        travelers=form["travelers"],
        flight_sort=form["flight_sort"],
        hotel_sort=form["hotel_sort"],
        min_hotel_stars=form["min_stars"],
        place_types=form["place_types"],
        daily_expense=form["daily_expense"],
        budget_limit=form["budget_limit"],
    )
    return {"markdown": render_markdown(plan), "json": plan, "steps": []}


def run_with_agent(form: dict, settings: dict) -> dict:
    """Plan with the LangChain tool-calling agent."""
    from travel_agent.agent import build_agent_executor, build_llm, build_user_request, run_agent

    llm = build_llm(settings["provider"], settings["api_key"], settings["model"])
    executor = build_agent_executor(llm)
    request = build_user_request(
        form["origin"], form["destination"], form["start_date"], form["days"], form["travelers"],
        form["min_stars"], form["flight_sort"], form["place_types"], form["budget_limit"],
        form["daily_expense"], form["extra_notes"],
    )
    return run_agent(executor, request)


def render_result(result: dict) -> None:
    """Show itinerary, structured JSON and the agent's tool-call trace."""
    tab_plan, tab_json, tab_trace = st.tabs(["📋 Itinerary", "🧾 JSON", "🔍 Agent steps"])
    with tab_plan:
        st.markdown(result["markdown"])
        st.download_button("⬇️ Download itinerary (.md)", result["markdown"], "itinerary.md")
    with tab_json:
        if result["json"]:
            st.json(result["json"])
            st.download_button("⬇️ Download JSON", json.dumps(result["json"], indent=2, ensure_ascii=False), "itinerary.json")
        else:
            st.info("The agent did not return a JSON block this time.")
    with tab_trace:
        if not result["steps"]:
            st.info("No tool-call trace (offline planner runs tools in a fixed order).")
        for number, step in enumerate(result["steps"], start=1):
            with st.expander(f"{number}. {step['tool']}"):
                st.write("**Input**")
                st.json(step["input"])
                st.write("**Output**")
                st.code(step["output"], language="json")


def render_data_explorer() -> None:
    """Browse the raw datasets and valid flight routes."""
    st.subheader("📚 Datasets")
    st.caption("The sample flight data only covers some city pairs – pick routes from this list for flight results.")
    routes = pd.DataFrame(available_routes(), columns=["From", "To"])
    st.dataframe(routes, use_container_width=True, hide_index=True, height=200)
    flights_tab, hotels_tab, places_tab = st.tabs(["Flights", "Hotels", "Places"])
    flights_tab.dataframe(pd.DataFrame(load_flights()), use_container_width=True, hide_index=True)
    hotels_tab.dataframe(pd.DataFrame(load_hotels()), use_container_width=True, hide_index=True)
    places_tab.dataframe(pd.DataFrame(load_places()), use_container_width=True, hide_index=True)


def main() -> None:
    st.title("✈️ AI Travel Planning Assistant")
    st.caption("A LangChain agent that searches flights, hotels, attractions and live weather to build your itinerary.")
    settings = render_sidebar()

    planner_tab, data_tab = st.tabs(["Plan a trip", "Explore data"])
    with planner_tab:
        form = render_form()
        if form:
            if form["origin"] == form["destination"]:
                st.error("Please choose different origin and destination cities.")
            elif settings["use_agent"] and not settings.get("api_key"):
                st.error("Enter an API key in the sidebar, or switch to the offline planner.")
            else:
                with st.spinner("Planning your trip..."):
                    try:
                        st.session_state["result"] = (
                            run_with_agent(form, settings) if settings["use_agent"] else run_offline(form)
                        )
                    except Exception as exc:  # noqa: BLE001 - show any API/network error nicely
                        st.session_state.pop("result", None)
                        st.error(f"Could not plan the trip: {exc}")
        if "result" in st.session_state:
            render_result(st.session_state["result"])
    with data_tab:
        render_data_explorer()


main()
