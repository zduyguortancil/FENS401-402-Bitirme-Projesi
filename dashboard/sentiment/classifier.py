"""
Olay siniflandirma — Guclendirilmis keyword classifier.
ML model yok, sifir RAM, mikrosaniye hizinda.
~250 keyword, 9 kategori.
"""
import re

# ── Event tanimlari ───────────────────────────────────────────
EVENT_META = {
    "security_threat": {
        "tr": "Guvenlik Tehdidi",
        "impact": -0.8,
        "icon": "\U0001f6a8",
        "keywords": [
            # Silah / saldiri
            "missile", "bomb", "bombing", "explosion", "explosive", "airstrike", "air strike",
            "drone attack", "drone strike", "rocket", "shelling", "artillery", "gunfire",
            "shooting", "gunman", "sniper", "IED", "landmine",
            # Teror
            "terror", "terrorism", "terrorist", "extremist", "militant", "insurgent", "jihad",
            "hostage", "kidnap", "abduction", "assassination", "car bomb", "suicide bomb",
            # Savas / catisma
            "war", "warfare", "conflict", "combat", "invasion", "military operation",
            "armed forces", "troops deployed", "military", "warzone", "ceasefire violation",
            "escalation", "offensive", "siege",
            # Genel guvenlik
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
        "tr": "Grev / Protesto",
        "impact": -0.6,
        "icon": "\u270a",
        "keywords": [
            # Grev
            "strike", "walkout", "work stoppage", "industrial action", "labor dispute",
            "labour dispute", "union action", "picket", "go-slow",
            # Protesto
            "protest", "demonstration", "rally", "march", "riot", "civil unrest",
            "tear gas", "water cannon", "barricade", "road block", "blockade",
            "crackdown", "clashes", "detained", "opposition",
            # Isciler
            "worker", "workers", "labor union", "trade union", "collective bargaining",
            "airline staff", "pilot strike", "cabin crew strike", "ground staff",
            "air traffic control strike", "ATC strike", "baggage handler",
        ],
    },
    "weather_disaster": {
        "tr": "Hava / Dogal Afet",
        "impact": -0.7,
        "icon": "\U0001f329\ufe0f",
        "keywords": [
            # Firtina
            "storm", "thunderstorm", "severe weather", "extreme weather",
            "hurricane", "typhoon", "cyclone", "tropical storm",
            "tornado", "twister", "waterspout",
            # Kis
            "snow", "snowstorm", "blizzard", "ice storm", "freezing rain",
            "frost", "polar vortex", "cold snap", "avalanche",
            # Sicaklik
            "heatwave", "heat wave", "extreme heat", "record temperature",
            # Diger
            "fog", "dense fog", "visibility", "sandstorm", "dust storm",
            "flood", "flooding", "flash flood", "mudslide", "landslide",
            "earthquake", "tsunami", "volcanic", "volcano", "eruption",
            "wildfire", "bushfire", "forest fire", "drought",
        ],
    },
    "flight_disruption": {
        "tr": "Ucus Aksakligi",
        "impact": -0.5,
        "icon": "\u2708\ufe0f\u26a0\ufe0f",
        "keywords": [
            # Iptal / gecikme
            "cancel", "cancelled", "canceled", "cancellation",
            "delay", "delayed", "delays", "hours delay", "long delay",
            "disrupt", "disrupted", "disruption",
            # Operasyonel
            "ground", "grounded", "ground stop", "divert", "diverted", "diversion",
            "suspend", "suspended", "suspension", "halt", "halted",
            "reroute", "rerouted", "stranded", "stuck at airport",
            "missed connection", "overbooking", "overbooked",
            "closed due to", "shut down", "not operating", "out of service",
            # Teknik
            "technical fault", "mechanical issue", "engine failure",
            "bird strike", "emergency landing", "precautionary landing",
            "runway closure", "runway closed", "runway incursion",
            "airport closure", "airport closed", "terminal closed",
            "system outage", "IT failure", "booking system down",
            # Kapasite
            "overcrowding", "overcrowded", "long queues", "queue chaos",
            "baggage delay", "lost luggage", "luggage chaos",
        ],
    },
    "tourism_growth": {
        "tr": "Turizm Buyumesi",
        "impact": +0.5,
        "icon": "\U0001f4c8",
        "keywords": [
            # Turizm artisi
            "tourism", "tourist", "tourists", "tourism boom", "tourism growth",
            "visitor", "visitors", "visitor numbers", "arrivals",
            "record arrivals", "record visitors", "record tourists",
            "travel demand", "travel surge", "travel boom",
            "overtourism", "tourism revenue",
            # Konaklama
            "hotel occupancy", "hotel bookings", "resort", "all-inclusive",
            "cruise", "cruise ship", "cruise tourism",
            # Ozel turizm
            "destination wedding", "digital nomad", "ecotourism",
            "cultural tourism", "heritage tourism", "medical tourism",
            "pilgrimage", "hajj", "umrah",
            # Mevsimsel
            "summer season", "winter season", "peak season", "holiday season",
            "spring break", "ski season",
            # Genel olumlu
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
        "tr": "Siyasi Istikrarsizlik",
        "impact": -0.6,
        "icon": "\u26a0\ufe0f",
        "keywords": [
            # Siyasi kriz
            "coup", "coup attempt", "military coup", "overthrow",
            "political crisis", "political turmoil", "political unrest",
            "government collapse", "regime change", "power struggle",
            # Secim / hukuk
            "disputed election", "election violence", "martial law",
            "state of siege", "curfew", "internet shutdown", "media blackout",
            # Uluslararasi
            "sanctions", "embargo", "diplomatic crisis", "diplomatic row",
            "expelled diplomat", "recalled ambassador",
            "trade war", "economic sanctions", "asset freeze",
            # Goc / sinir
            "border closure", "border closed", "refugee crisis",
            "mass migration", "deportation",
        ],
    },
    "health_crisis": {
        "tr": "Saglik Krizi",
        "impact": -0.7,
        "icon": "\U0001f3e5",
        "keywords": [
            # Pandemi
            "pandemic", "epidemic", "outbreak", "new variant",
            "virus", "covid", "coronavirus", "influenza", "bird flu", "avian flu",
            "monkeypox", "mpox", "ebola", "cholera", "dengue", "malaria",
            # Onlemler
            "quarantine", "lockdown", "travel restriction", "travel advisory",
            "health emergency", "health alert", "WHO emergency",
            "mask mandate", "vaccination required", "vaccine passport",
            "testing requirement", "PCR test", "health screening",
            # Hastane
            "hospital overwhelmed", "healthcare crisis", "medical emergency",
            "disease", "contamination", "biohazard",
        ],
    },
    "positive_travel": {
        "tr": "Olumlu Seyahat Haberi",
        "impact": +0.4,
        "icon": "\U0001f680",
        "keywords": [
            # Yeni rota / genisleme
            "new route", "new routes", "new destination", "new service",
            "route launch", "inaugural flight", "first flight",
            "expansion", "fleet expansion", "new aircraft", "fleet order",
            "airline launch", "new airline",
            # Rekor / basari
            "passenger record", "record passengers", "busiest day",
            "milestone", "on-time performance", "award", "best airline",
            "five star", "5-star", "skytrax",
            # Altyapi
            "new terminal", "terminal opening", "runway opening",
            "airport expansion", "airport upgrade", "modernization",
            "direct flight", "nonstop flight", "increased frequency",
            # Fiyat / erisilebilirlik
            "cheap flights", "low fare", "fare sale", "price drop",
            "open skies", "visa-free", "visa waiver", "e-visa",
            # Genel gelisme
            "partnership", "codeshare", "interline", "alliance",
            "passenger growth", "revenue growth", "profit",
            "investment", "upgrade", "renovation", "smart airport",
            "nonstop", "non-stop", "first flight to", "new connection",
            "hotel", "resort opening", "new hotel",
            "AI", "robot", "innovation", "technology", "automated",
        ],
    },
    "general_news": {
        "tr": "Genel Haber",
        "impact": 0.05,
        "icon": "\U0001f4f0",
        "keywords": [],  # Default kategori, keyword eslesmezse buraya duser
    },
}

# Duz keyword listesi (hizli lookup icin)
_EVENT_KEYWORDS = {key: meta["keywords"] for key, meta in EVENT_META.items() if meta["keywords"]}


# Yanlis pozitif onleme — bu ifadeler gercek tehdit/olay DEGIL
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

# Havacilik context terimleri — en az biri baslikta olmali (negatif kategoriler icin)
_AVIATION_CONTEXT = [
    "flight", "airport", "airline", "aviation", "passenger", "plane",
    "runway", "terminal", "airspace", "pilot", "cabin", "boarding",
    "travel", "tourism", "tourist", "hotel", "destination",
    "visa", "border", "embassy", "evacuat",
]


def classify_event(title):
    """
    Tek bir baslik icin keyword-based olay siniflandirmasi.
    Yanlis pozitif onleme + havacilik context kontrolu dahil.
    Returns: (event_key, confidence)
    """
    if not title or not title.strip():
        return "general_news", 0.0

    text = re.sub(r"http\S+", "", title).lower().strip()

    # Yanlis pozitif kontrolu
    for fp in _FALSE_POSITIVE_PHRASES:
        if fp in text:
            return "general_news", 0.3

    best_key = "general_news"
    best_score = 0.0

    for key, keywords in _EVENT_KEYWORDS.items():
        # Her keyword icin eslesme say
        matches = 0
        matched_words = []
        for kw in keywords:
            if kw in text:
                matches += 1
                matched_words.append(kw)
                # Cok kelimeli keyword daha degerli
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

    # Negatif kategoriler icin havacilik context kontrolu
    # "bomb" veya "strike" var ama "flight/airport/travel" yoksa -> guvenilirlik dusur
    negative_cats = ("security_threat", "strike_protest", "weather_disaster",
                     "flight_disruption", "political_instability", "health_crisis")
    if best_key in negative_cats and best_score < 0.90:
        has_context = any(ctx in text for ctx in _AVIATION_CONTEXT)
        if not has_context:
            best_score *= 0.5  # havacilik baglami yok, guvenilirlik yariya dusur
            if best_score < 0.35:
                return "general_news", 0.3

    return best_key, round(best_score, 4)


def classify_batch(titles):
    """
    Toplu siniflandirma — mikrosaniye hizinda.
    Returns: list of (event_key, confidence)
    """
    if not titles:
        return []
    return [classify_event(t) for t in titles]


def ensure_loaded():
    """Uyumluluk icin — keyword classifier yukleme gerektirmez."""
    pass
