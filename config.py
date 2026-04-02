"""
Configuration for the Epidemic Spread Prediction Data Pipeline.
All paths, URLs, and pipeline parameters in one place.
"""
from pathlib import Path

# ─── Project Paths ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_DIR = ROOT_DIR / "data" / "output"
COUNTRY_OUTPUT_DIR = OUTPUT_DIR / "countries"
LOGS_DIR = ROOT_DIR / "logs"

# Create directories
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR, COUNTRY_OUTPUT_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Data Source URLs ───────────────────────────────────────────────────────────
JHU_BASE_URL = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series"
JHU_URLS = {
    "confirmed": f"{JHU_BASE_URL}/time_series_covid19_confirmed_global.csv",
    "deaths":    f"{JHU_BASE_URL}/time_series_covid19_deaths_global.csv",
    "recovered": f"{JHU_BASE_URL}/time_series_covid19_recovered_global.csv",
}

OWID_URL = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"

GOOGLE_MOBILITY_URL = "https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv"

# ─── Pipeline Parameters ───────────────────────────────────────────────────────
ROLLING_WINDOW = 7           # 7-day rolling average window
ZSCORE_THRESHOLD = 3.0       # anomaly detection threshold
INTERPOLATION_LIMIT = 3      # max consecutive NaN days to interpolate
FLOAT_DTYPE = "float32"      # memory-optimized dtype

# ─── Target Countries for Per-Country Export ────────────────────────────────────
TARGET_COUNTRIES = ["United States", "India", "Brazil", "Germany", "South Africa"]

# ─── Country Name Mapping (harmonize across datasets) ──────────────────────────
COUNTRY_NAME_MAP = {
    # JHU names → standard
    "US": "United States",
    "Korea, South": "South Korea",
    "Taiwan*": "Taiwan",
    "Burma": "Myanmar",
    "Cote d'Ivoire": "Ivory Coast",
    "West Bank and Gaza": "Palestine",
    "Congo (Kinshasa)": "DR Congo",
    "Congo (Brazzaville)": "Congo",
    # OWID names → standard
    "United States": "United States",
    # Google Mobility names → standard
    "United States": "United States",
    "Republic of Korea": "South Korea",
}

# ─── Feature Columns ───────────────────────────────────────────────────────────
OWID_FEATURES = [
    "location",
    "date",
    "total_vaccinations",
    "people_vaccinated",
    "people_fully_vaccinated",
    "total_boosters",
    "new_vaccinations_smoothed",
    "total_tests",
    "new_tests",
    "new_tests_smoothed",
    "total_tests_per_thousand",
    "new_tests_per_thousand",
    "positive_rate",
    "tests_per_case",
    "stringency_index",
    "population_density",
    "median_age",
    "aged_65_older",
    "gdp_per_capita",
    "hospital_beds_per_thousand",
    "life_expectancy",
    "human_development_index",
    "population",
    "icu_patients",
    "hosp_patients",
    "reproduction_rate",
]

MOBILITY_FEATURES = [
    "country_region",
    "date",
    "retail_and_recreation_percent_change_from_baseline",
    "grocery_and_pharmacy_percent_change_from_baseline",
    "parks_percent_change_from_baseline",
    "transit_stations_percent_change_from_baseline",
    "workplaces_percent_change_from_baseline",
    "residential_percent_change_from_baseline",
]

MOBILITY_RENAME = {
    "retail_and_recreation_percent_change_from_baseline": "mobility_retail",
    "grocery_and_pharmacy_percent_change_from_baseline": "mobility_grocery",
    "parks_percent_change_from_baseline": "mobility_parks",
    "transit_stations_percent_change_from_baseline": "mobility_transit",
    "workplaces_percent_change_from_baseline": "mobility_workplaces",
    "residential_percent_change_from_baseline": "mobility_residential",
}
