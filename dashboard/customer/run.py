import os
import sys

# Add parent directory to path so we can import customer app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from customer.customer_app import app

if __name__ == "__main__":
    debug_mode = os.getenv("SEATWISE_CUSTOMER_DEBUG", "0") == "1"
    print("Starting FindTicket 2.0 Customer App on http://localhost:5006")
    app.run(
        host="0.0.0.0",
        port=5006,
        debug=debug_mode,
        use_reloader=False,
        threaded=True,
    )
