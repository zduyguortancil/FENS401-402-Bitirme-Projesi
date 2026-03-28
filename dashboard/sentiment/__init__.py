# sentiment package — v2 (Google News RSS + Keyword Classifier)

from .cities import CITIES, AIRPORT_TO_CITY
from .cache_db import init_db, load_cached_scores
from .scheduler import start_scheduler, stop_scheduler
from .classifier import EVENT_META
