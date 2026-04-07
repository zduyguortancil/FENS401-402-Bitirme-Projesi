"""
Event classification — Enhanced keyword classifier.
No ML model, zero RAM, microsecond inference.
~416 keywords, 9 categories.
"""
import re

# ── Event definitions ───────────────────────────────────────────
EVENT_META = {
    "security_threat": {
        "tr": "Security Threat",
        "impact": -0.8,
        "icon": "\U0001f6a8",
        "keywords": [
            # Weapons / attacks
            "missile", "bomb", "bombing", "explosion", "explosive", "airstrike", "air strike",
            "drone attack", "drone strike", "rocket", "shelling", "artillery", "gunfire",
            "shooting", "gunman", "sniper", "IED", "landmine",
            # Terrorism
            "terror", "terrorism", "terrorist", "extremist", "militant", "insurgent", "jihad",
            "hostage", "kidnap", "abduction", "assassination", "car bomb", "suicide bomb",
            # War / conflict
            "war", "warfare", "conflict", "combat", "invasion", "military operation",
            "armed forces", "troops deployed", "military", "warzone", "ceasefire violation",
            "escalation", "offensive", "siege",
            # General security
            "security threat", "security alert", "security incident", "threat level",
            "no-fly zone", "airspace closed", "airspace closure", "travel ban",
            "embassy warning", "evacuation order", "state of emergency",
            "suspicious", "evacuated", "evacuation", "bomb threat", "bomb scare",
            "seized", "drug bust", "smuggling", "trafficking", "arrest",
            "stabbing", "shooting", "robbery", "crime", "murder",
            "fire", "blaze", "arson", "inferno",
            "crash", "accident", "collision",
        ],
    },
    "strike_protest": {
        "tr": "Strike / Protest",
        "impact": -0.6,
        "icon": "\u270a",
        "keywords": [
            # Strike
            "strike", "walkout", "work stoppage", "industrial action", "labor dispute",
            "labour dispute", "union action", "picket", "go-slow",
            # Protest
            "protest", "demonstration", "rally", "march", "riot", "civil unrest",
            "tear gas", "water cannon", "barricade", "road block", "blockade",
            "crackdown", "clashes", "detained", "opposition",
            # Workers
            "worker", "workers", "labor union", "trade union", "collective bargaining",
            "airline staff", "pilot strike", "cabin crew strike", "ground staff",
            "air traffic control strike", "ATC strike", "baggage handler",
        ],
    },
    "weather_disaster": {
        "tr": "Weather / Natural Disaster",
        "impact": -0.7,
        "icon": "\U0001f329\ufe0f",
        "keywords": [
            # Storm
            "storm", "thunderstorm", "severe weather", "extreme weather",
            "hurricane", "typhoon", "cyclone", "tropical storm",
            "tornado", "twister", "waterspout",
            # Winter
            "snow", "snowstorm", "blizzard", "ice storm", "freezing rain",
            "frost", "polar vortex", "cold snap", "avalanche",
            # Heat
            "heatwave", "heat wave", "extreme heat", "record temperature",
            # Other
            "fog", "dense fog", "visibility", "sandstorm", "dust storm",
            "flood", "flooding", "flash flood", "mudslide", "landslide",
            "earthquake", "tsunami", "volcanic", "volcano", "eruption",
            "wildfire", "bushfire", "forest fire", "drought",
        ],
    },
    "flight_disruption": {
        "tr": "Flight Disruption",
        "impact": -0.5,
        "icon": "\u2708\ufe0f\u26a0\ufe0f",
        "keywords": [
            # Cancellation / delay
            "cancel", "cancelled", "canceled", "cancellation",
            "delay", "delayed", "delays", "hours delay", "long delay",
            "disrupt", "disrupted", "disruption",
            # Operational
            "ground", "grounded", "ground stop", "divert", "diverted", "diversion",
            "suspend", "suspended", "suspension", "halt", "halted",
            "reroute", "rerouted", "stranded", "stuck at airport",
            "missed connection", "overbooking", "overbooked",
            "closed due to", "shut down", "not operating", "out of service",
            # Technical
            "technical fault", "mechanical issue", "engine failure",
            "bird strike", "emergency landing", "precautionary landing",
            "runway closure", "runway closed", "runway incursion",
            "airport closure", "airport closed", "terminal closed",
            "system outage", "IT failure", "booking system down",
            # Capacity
            "overcrowding", "overcrowded", "long queues", "queue chaos",
            "baggage delay", "lost luggage", "luggage chaos",
        ],
    },
    "tourism_growth": {
        "tr": "Tourism Growth",
        "impact": +0.5,
        "icon": "\U0001f4c8",
        "keywords": [
            # Tourism increase
            "tourism", "tourist", "tourists", "tourism boom", "tourism growth",
            "visitor", "visitors", "visitor numbers", "arrivals",
            "record arrivals", "record visitors", "record tourists",
            "travel demand", "travel surge", "travel boom",
            "overtourism", "tourism revenue",
            # Accommodation
            "hotel occupancy", "hotel bookings", "resort", "all-inclusive",
            "cruise", "cruise ship", "cruise tourism",
            # Special tourism
            "destination wedding", "digital nomad", "ecotourism",
            "cultural tourism", "heritage tourism", "medical tourism",
            "pilgrimage", "hajj", "umrah",
            # Seasonal
            "summer season", "winter season", "peak season", "holiday season",
            "spring break", "ski season",
            # General positive
            "travel guide", "things to do", "best time to visit", "hidden gem",
            "must visit", "top destination", "bucket list", "travel tips",
            "safe to travel", "worth visiting",
            "food", "street food", "restaurant", "cuisine", "nightlife",
            "beach", "island", "hiking", "adventure", "safari",
            "museum", "temple", "palace", "cathedral", "monument",
            "festival", "carnival", "celebration", "event",
            "homes in", "real estate", "property",
        ],
    },
    "political_instability": {
        "tr": "Political Instability",
        "impact": -0.6,
        "icon": "\u26a0\ufe0f",
        "keywords": [
            # Political crisis
            "coup", "coup attempt", "military coup", "overthrow",
            "political crisis", "political turmoil", "political unrest",
            "government collapse", "regime change", "power struggle",
            # Election / law
            "disputed election", "election violence", "martial law",
            "state of siege", "curfew", "internet shutdown", "media blackout",
            # International
            "sanctions", "embargo", "diplomatic crisis", "diplomatic row",
            "expelled diplomat", "recalled ambassador",
            "trade war", "economic sanctions", "asset freeze",
            # Migration / border
            "border closure", "border closed", "refugee crisis",
            "mass migration", "deportation",
        ],
    },
    "health_crisis": {
        "tr": "Health Crisis",
        "impact": -0.7,
        "icon": "\U0001f3e5",
        "keywords": [
            # Pandemic
            "pandemic", "epidemic", "outbreak", "new variant",
            "virus", "covid", "coronavirus", "influenza", "bird flu", "avian flu",
            "monkeypox", "mpox", "ebola", "cholera", "dengue", "malaria",
            # Measures
            "quarantine", "lockdown", "travel restriction", "travel advisory",
            "health emergency", "health alert", "WHO emergency",
            "mask mandate", "vaccination required", "vaccine passport",
            "testing requirement", "PCR test", "health screening",
            # Hospital
            "hospital overwhelmed", "healthcare crisis", "medical emergency",
            "disease", "contamination", "biohazard",
        ],
    },
    "positive_travel": {
        "tr": "Positive Travel News",
        "impact": +0.4,
        "icon": "\U0001f680",
        "keywords": [
            # New route / expansion
            "new route", "new routes", "new destination", "new service",
            "route launch", "inaugural flight", "first flight",
            "expansion", "fleet expansion", "new aircraft", "fleet order",
            "airline launch", "new airline",
            # Record / achievement
            "passenger record", "record passengers", "busiest day",
            "milestone", "on-time performance", "award", "best airline",
            "five star", "5-star", "skytrax",
            # Infrastructure
            "new terminal", "terminal opening", "runway opening",
            "airport expansion", "airport upgrade", "modernization",
            "direct flight", "nonstop flight", "increased frequency",
            # Price / accessibility
            "cheap flights", "low fare", "fare sale", "price drop",
            "open skies", "visa-free", "visa waiver", "e-visa",
            # General development
            "partnership", "codeshare", "interline", "alliance",
            "passenger growth", "revenue growth", "profit",
            "investment", "upgrade", "renovation", "smart airport",
            "nonstop", "non-stop", "first flight to", "new connection",
            "hotel", "resort opening", "new hotel",
            "AI", "robot", "innovation", "technology", "automated",
        ],
    },
    "general_news": {
        "tr": "General News",
        "impact": 0.05,
        "icon": "\U0001f4f0",
        "keywords": [],  # Default kategori, keyword eslesmezse buraya duser
    },
}

# Flat keyword list (fast lookup)
_EVENT_KEYWORDS = {key: meta["keywords"] for key, meta in EVENT_META.items() if meta["keywords"]}


# False positive prevention — these phrases are NOT real threats/events
_FALSE_POSITIVE_PHRASES = [
    "box office bomb", "bomb cyclone", "bombshell report", "bombshell",
    "bowling strike", "strike gold", "strike a deal", "strike a chord",
    "price war", "trade war", "star wars", "format war",
    "fire sale", "fire up", "fired up", "rapid fire", "crossfire",
    "crash course", "crash diet", "crash landing pad",
    "killer app", "killer deal", "killer feature",
    "travel advisory lifted", "ban lifted", "restriction lifted",
    "shooting star", "shooting up", "moon shot",
]

# Aviation context terms — at least one must be in title (for negative categories)
_AVIATION_CONTEXT = [
    "flight", "airport", "airline", "aviation", "passenger", "plane",
    "runway", "terminal", "airspace", "pilot", "cabin", "boarding",
    "travel", "tourism", "tourist", "hotel", "destination",
    "visa", "border", "embassy", "evacuat",
]


def classify_event(title):
    """
    Keyword-based event classification for a single headline.
    Includes false positive filtering + aviation context check.
    Returns: (event_key, confidence)
    """
    if not title or not title.strip():
        return "general_news", 0.0

    text = re.sub(r"http\S+", "", title).lower().strip()

    # False positive check
    for fp in _FALSE_POSITIVE_PHRASES:
        if fp in text:
            return "general_news", 0.3

    best_key = "general_news"
    best_score = 0.0

    for key, keywords in _EVENT_KEYWORDS.items():
        # Count keyword matches
        matches = 0
        matched_words = []
        for kw in keywords:
            if kw in text:
                matches += 1
                matched_words.append(kw)
                # Multi-word keywords are more valuable
                if " " in kw:
                    matches += 0.5

        if matches > 0:
            # Confidence: 1 match = 0.5, 2 = 0.7, 3+ = 0.85+
            confidence = min(0.4 + matches * 0.15, 0.95)
            if confidence > best_score:
                best_score = confidence
                best_key = key

    if best_score == 0.0:
        return "general_news", 0.3

    # Aviation context check for negative categories
    # "bomb" or "strike" present but no "flight/airport/travel" -> reduce confidence
    negative_cats = ("security_threat", "strike_protest", "weather_disaster",
                     "flight_disruption", "political_instability", "health_crisis")
    if best_key in negative_cats and best_score < 0.90:
        has_context = any(ctx in text for ctx in _AVIATION_CONTEXT)
        if not has_context:
            best_score *= 0.5  # no aviation context, halve confidence
            if best_score < 0.35:
                return "general_news", 0.3

    return best_key, round(best_score, 4)


def classify_batch(titles):
    """
    Batch classification — microsecond speed.
    Returns: list of (event_key, confidence)
    """
    if not titles:
        return []
    return [classify_event(t) for t in titles]


def ensure_loaded():
    """Compatibility — keyword classifier requires no loading."""
    pass
