"""
City definitions — 51 destinations.
All sentiment modules import from here.
"""

_PALETTE = [
    "#e11d48", "#f59e0b", "#3b82f6", "#22c55e", "#ef4444",
    "#8b5cf6", "#06b6d4", "#ec4899", "#14b8a6", "#f97316",
    "#6366f1", "#84cc16", "#d946ef", "#0ea5e9", "#a855f7",
    "#10b981", "#e879f9", "#facc15", "#38bdf8", "#fb923c",
    "#4ade80", "#c084fc", "#fbbf24", "#2dd4bf", "#f472b6",
    "#818cf8", "#34d399", "#fb7185", "#a3e635", "#67e8f9",
    "#c4b5fd", "#fca5a5", "#86efac", "#fcd34d", "#7dd3fc",
    "#d8b4fe", "#6ee7b7", "#fda4af", "#bef264", "#a5f3fc",
    "#e9d5ff", "#fecaca", "#bbf7d0", "#fde68a", "#bae6fd",
    "#f0abfc", "#a7f3d0", "#fecdd3", "#d9f99d", "#99f6e4",
    "#c7d2fe",
]

# (key, label, city_en, flag, country, airport_codes)
_CITY_DEFS = [
    ("istanbul",      "Istanbul",       "Istanbul",       "\U0001f1f9\U0001f1f7", "Turkey",              ["IST", "SAW"]),
    ("london",        "London",         "London",         "\U0001f1ec\U0001f1e7", "United Kingdom",      ["LHR", "LGW", "STN"]),
    ("paris",         "Paris",          "Paris",          "\U0001f1eb\U0001f1f7", "France",              ["CDG", "ORY"]),
    ("dubai",         "Dubai",          "Dubai",          "\U0001f1e6\U0001f1ea", "United Arab Emirates", ["DXB", "DWC"]),
    ("newyork",       "New York",       "New York",       "\U0001f1fa\U0001f1f8", "United States",       ["JFK"]),
    ("frankfurt",     "Frankfurt",      "Frankfurt",      "\U0001f1e9\U0001f1ea", "Germany",             ["FRA"]),
    ("barcelona",     "Barcelona",      "Barcelona",      "\U0001f1ea\U0001f1f8", "Spain",               ["BCN"]),
    ("rome",          "Rome",           "Rome",           "\U0001f1ee\U0001f1f9", "Italy",               ["FCO"]),
    ("milan",         "Milan",          "Milan",          "\U0001f1ee\U0001f1f9", "Italy",               ["MXP"]),
    ("madrid",        "Madrid",         "Madrid",         "\U0001f1ea\U0001f1f8", "Spain",               ["MAD"]),
    ("munich",        "Munich",         "Munich",         "\U0001f1e9\U0001f1ea", "Germany",             ["MUC"]),
    ("nice",          "Nice",           "Nice",           "\U0001f1eb\U0001f1f7", "France",              ["NCE"]),
    ("manchester",    "Manchester",     "Manchester",     "\U0001f1ec\U0001f1e7", "United Kingdom",      ["MAN"]),
    ("telaviv",       "Tel Aviv",       "Tel Aviv",       "\U0001f1ee\U0001f1f1", "Israel",              ["TLV"]),
    ("beirut",        "Beirut",         "Beirut",         "\U0001f1f1\U0001f1e7", "Lebanon",             ["BEY"]),
    ("amman",         "Amman",          "Amman",          "\U0001f1ef\U0001f1f4", "Jordan",              ["AMM"]),
    ("jeddah",        "Jeddah",         "Jeddah",         "\U0001f1f8\U0001f1e6", "Saudi Arabia",        ["JED"]),
    ("riyadh",        "Riyadh",         "Riyadh",         "\U0001f1f8\U0001f1e6", "Saudi Arabia",        ["RUH"]),
    ("doha",          "Doha",           "Doha",           "\U0001f1f6\U0001f1e6", "Qatar",               ["DOH"]),
    ("bahrain",       "Bahrain",        "Bahrain",        "\U0001f1e7\U0001f1ed", "Bahrain",             ["BAH"]),
    ("kuwait",        "Kuwait",         "Kuwait",         "\U0001f1f0\U0001f1fc", "Kuwait",              ["KWI"]),
    ("abudhabi",      "Abu Dhabi",      "Abu Dhabi",      "\U0001f1e6\U0001f1ea", "United Arab Emirates", ["AUH"]),
    ("cairo",         "Cairo",          "Cairo",          "\U0001f1ea\U0001f1ec", "Egypt",               ["CAI"]),
    ("hurghada",      "Hurghada",       "Hurghada",       "\U0001f1ea\U0001f1ec", "Egypt",               ["HRG"]),
    ("casablanca",    "Casablanca",     "Casablanca",     "\U0001f1f2\U0001f1e6", "Morocco",             ["CMN"]),
    ("marrakech",     "Marrakech",      "Marrakech",      "\U0001f1f2\U0001f1e6", "Morocco",             ["RAK"]),
    ("nairobi",       "Nairobi",        "Nairobi",        "\U0001f1f0\U0001f1ea", "Kenya",               ["NBO"]),
    ("mombasa",       "Mombasa",        "Mombasa",        "\U0001f1f0\U0001f1ea", "Kenya",               ["MBA"]),
    ("lagos",         "Lagos",          "Lagos",          "\U0001f1f3\U0001f1ec", "Nigeria",             ["LOS"]),
    ("abuja",         "Abuja",          "Abuja",          "\U0001f1f3\U0001f1ec", "Nigeria",             ["ABV"]),
    ("johannesburg",  "Johannesburg",   "Johannesburg",   "\U0001f1ff\U0001f1e6", "South Africa",        ["JNB"]),
    ("capetown",      "Cape Town",      "Cape Town",      "\U0001f1ff\U0001f1e6", "South Africa",        ["CPT"]),
    ("tokyo",         "Tokyo",          "Tokyo",          "\U0001f1ef\U0001f1f5", "Japan",               ["NRT", "HND"]),
    ("osaka",         "Osaka",          "Osaka",          "\U0001f1ef\U0001f1f5", "Japan",               ["KIX"]),
    ("seoul",         "Seoul",          "Seoul",          "\U0001f1f0\U0001f1f7", "South Korea",         ["ICN"]),
    ("beijing",       "Beijing",        "Beijing",        "\U0001f1e8\U0001f1f3", "China",               ["PEK"]),
    ("shanghai",      "Shanghai",       "Shanghai",       "\U0001f1e8\U0001f1f3", "China",               ["PVG"]),
    ("singapore",     "Singapore",      "Singapore",      "\U0001f1f8\U0001f1ec", "Singapore",           ["SIN"]),
    ("bangkok",       "Bangkok",        "Bangkok",        "\U0001f1f9\U0001f1ed", "Thailand",            ["BKK"]),
    ("phuket",        "Phuket",         "Phuket",         "\U0001f1f9\U0001f1ed", "Thailand",            ["HKT"]),
    ("delhi",         "Delhi",          "Delhi",          "\U0001f1ee\U0001f1f3", "India",               ["DEL"]),
    ("mumbai",        "Mumbai",         "Mumbai",         "\U0001f1ee\U0001f1f3", "India",               ["BOM"]),
    ("losangeles",    "Los Angeles",    "Los Angeles",    "\U0001f1fa\U0001f1f8", "United States",       ["LAX"]),
    ("chicago",       "Chicago",        "Chicago",        "\U0001f1fa\U0001f1f8", "United States",       ["ORD"]),
    ("miami",         "Miami",          "Miami",          "\U0001f1fa\U0001f1f8", "United States",       ["MIA"]),
    ("toronto",       "Toronto",        "Toronto",        "\U0001f1e8\U0001f1e6", "Canada",              ["YYZ"]),
    ("vancouver",     "Vancouver",      "Vancouver",      "\U0001f1e8\U0001f1e6", "Canada",              ["YVR"]),
    ("mexicocity",    "Mexico City",    "Mexico City",    "\U0001f1f2\U0001f1fd", "Mexico",              ["MEX"]),
    ("saopaulo",      "Sao Paulo",      "Sao Paulo",      "\U0001f1e7\U0001f1f7", "Brazil",              ["GRU"]),
    ("buenosaires",   "Buenos Aires",   "Buenos Aires",   "\U0001f1e6\U0001f1f7", "Argentina",           ["EZE"]),
    ("riodejaneiro",  "Rio de Janeiro", "Rio de Janeiro", "\U0001f1e7\U0001f1f7", "Brazil",              ["GIG"]),
]


def _build_cities():
    cities = {}
    for i, (key, label, city_en, flag, country, codes) in enumerate(_CITY_DEFS):
        cities[key] = {
            "label": label,
            "city_en": city_en,
            "flag": flag,
            "country": country,
            "codes": codes,
            "color": _PALETTE[i % len(_PALETTE)],
        }
    return cities


CITIES = _build_cities()

# Airport code -> city key lookup
AIRPORT_TO_CITY = {}
for _k, _c in CITIES.items():
    for _code in _c["codes"]:
        AIRPORT_TO_CITY[_code] = _k
