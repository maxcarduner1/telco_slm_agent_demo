"""
WanderBricks Tools

Weather lookup via Open-Meteo (no API key required).
Defined with @tool for standard LangGraph ToolNode usage.
"""

import calendar
from collections import Counter

import requests
from typing import Any, Dict, Optional

from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# WMO weather codes used by Open-Meteo
# ---------------------------------------------------------------------------
_WEATHER_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
}

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"

_MONTH_NAMES = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _geocode(city: str) -> Optional[Dict[str, Any]]:
    """Resolve city name to coordinates via Open-Meteo Geocoding API."""
    resp = requests.get(GEOCODE_URL, params={"name": city, "count": 1}, timeout=10)
    resp.raise_for_status()
    results = resp.json().get("results", [])
    if not results:
        return None
    r = results[0]
    return {
        "name": r.get("name"),
        "latitude": r.get("latitude"),
        "longitude": r.get("longitude"),
        "country": r.get("country"),
    }


def _get_forecast(lat: float, lon: float, days: int = 14) -> Dict[str, Any]:
    """Fetch daily weather forecast from Open-Meteo."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join([
            "temperature_2m_max", "temperature_2m_min",
            "precipitation_sum", "weathercode",
        ]),
        "forecast_days": min(days, 16),
        "timezone": "auto",
    }
    resp = requests.get(FORECAST_URL, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def _get_historical_monthly(lat: float, lon: float, month: int) -> Dict[str, Any]:
    """Fetch daily data for a given month across 2021-2025 and return averages."""
    # Make one request per year, then combine -- avoids getting non-target months
    all_t_max, all_t_min, all_precip, all_codes = [], [], [], []
    for year in range(2021, 2026):
        last_day = calendar.monthrange(year, month)[1]
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": f"{year}-{month:02d}-01",
            "end_date": f"{year}-{month:02d}-{last_day:02d}",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
            "timezone": "auto",
        }
        resp = requests.get(HISTORICAL_URL, params=params, timeout=30)
        resp.raise_for_status()
        daily = resp.json().get("daily", {})
        all_t_max.extend(v for v in daily.get("temperature_2m_max", []) if v is not None)
        all_t_min.extend(v for v in daily.get("temperature_2m_min", []) if v is not None)
        all_precip.extend(v for v in daily.get("precipitation_sum", []) if v is not None)
        all_codes.extend(v for v in daily.get("weather_code", []) if v is not None)

    t_max, t_min, precip, codes = all_t_max, all_t_min, all_precip, all_codes

    avg_high = sum(t_max) / len(t_max) if t_max else 0
    avg_low = sum(t_min) / len(t_min) if t_min else 0
    avg_precip = sum(precip) / len(precip) if precip else 0
    typical_code = Counter(codes).most_common(1)[0][0] if codes else 0

    return {
        "avg_high": round(avg_high, 1),
        "avg_low": round(avg_low, 1),
        "avg_precip_mm": round(avg_precip, 1),
        "typical_weather": _WEATHER_CODES.get(typical_code, "Unknown"),
        "num_days": len(t_max),
    }


# ---------------------------------------------------------------------------
# LangChain @tool for ToolNode
# ---------------------------------------------------------------------------

@tool
def weather_lookup(city: str, month: Optional[int] = None) -> str:
    """Get weather information for a destination city.

    If month is provided (1-12), returns historical averages for that month
    based on 2021-2025 data. If month is omitted, returns a 14-day forecast.

    Args:
        city: Name of the city (e.g. "Paris", "Tokyo", "New York")
        month: Month number (1=January, 12=December). If provided, returns
               historical averages instead of a forecast.
    """
    location = _geocode(city)
    if not location:
        return f"Could not find location for '{city}'."

    name = f"{location['name']}, {location['country']}"

    # Historical monthly averages
    if month is not None:
        if not 1 <= month <= 12:
            return f"Invalid month: {month}. Must be 1-12."
        try:
            avg = _get_historical_monthly(location["latitude"], location["longitude"], month)
        except requests.RequestException as e:
            return f"Weather API error for {city}: {e}"
        return (
            f"Historical weather averages for {name} in {_MONTH_NAMES[month]} (2021-2025, {avg['num_days']} days):\n"
            f"  Avg high: {avg['avg_high']}C\n"
            f"  Avg low:  {avg['avg_low']}C\n"
            f"  Avg daily precipitation: {avg['avg_precip_mm']}mm\n"
            f"  Typical conditions: {avg['typical_weather']}"
        )

    # 14-day forecast (default)
    try:
        data = _get_forecast(location["latitude"], location["longitude"])
    except requests.RequestException as e:
        return f"Weather API error for {city}: {e}"

    daily = data.get("daily", {})
    dates = daily.get("time", [])
    t_max = daily.get("temperature_2m_max", [])
    t_min = daily.get("temperature_2m_min", [])
    precip = daily.get("precipitation_sum", [])
    codes = daily.get("weathercode", [])

    lines = [f"Weather forecast for {name}:"]
    for i, date in enumerate(dates):
        desc = _WEATHER_CODES.get(codes[i], "Unknown") if i < len(codes) else ""
        hi = f"{t_max[i]}°C" if i < len(t_max) else "?"
        lo = f"{t_min[i]}°C" if i < len(t_min) else "?"
        rain = f"{precip[i]}mm" if i < len(precip) else "?"
        lines.append(f"  {date}: {desc}, {lo}-{hi}, precip {rain}")

    return "\n".join(lines)


# All tools for binding to the enrichment LLM
all_tools = [weather_lookup]