"""
predict_cancellation.py
Hotel Booking Cancellation Predictor - StayNest Hospitality Group

Loads the trained pipeline and predicts cancellation for a sample booking.

Usage:
    python predict_cancellation.py
"""

import os
import joblib
import pandas as pd

BASE       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "outputs", "hotel_cancellation_model.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model not found at " + MODEL_PATH + "\n"
        "Run  python train_model.py  first."
    )

pipeline = joblib.load(MODEL_PATH)
print("Model loaded from:", MODEL_PATH, "\n")

# ── Sample booking ────────────────────────────────────────────────────────────
# All 28 features used during training must be provided.
# Engineered features (total_nights, total_guests, room_match,
# has_special_request) must be computed manually here.

stays_in_weekend_nights     = 2
stays_in_week_nights        = 3
adults                      = 2
children                    = 0.0
babies                      = 0
reserved_room_type          = "A"
assigned_room_type          = "A"   # used only to compute room_match
total_of_special_requests   = 1

# Derived / engineered
total_nights        = stays_in_weekend_nights + stays_in_week_nights
total_guests        = adults + children + babies
room_match          = int(reserved_room_type == assigned_room_type)
has_special_request = int(total_of_special_requests > 0)

sample = pd.DataFrame([{
    # Categorical
    "hotel":                          "City Hotel",
    "arrival_date_month":             "August",
    "meal":                           "BB",
    "market_segment":                 "Online TA",
    "distribution_channel":           "TA/TO",
    "reserved_room_type":             reserved_room_type,
    "deposit_type":                   "No Deposit",
    "customer_type":                  "Transient",
    "country":                        "PRT",

    # Numerical
    "lead_time":                      120,
    "stays_in_weekend_nights":        stays_in_weekend_nights,
    "stays_in_week_nights":           stays_in_week_nights,
    "adults":                         adults,
    "children":                       children,
    "babies":                         babies,
    "is_repeated_guest":              0,
    "previous_cancellations":         0,
    "previous_bookings_not_canceled": 0,
    "booking_changes":                0,
    "days_in_waiting_list":           0,
    "adr":                            110.0,
    "required_car_parking_spaces":    0,
    "total_of_special_requests":      total_of_special_requests,
    "agent":                          240.0,

    # Engineered
    "total_nights":                   total_nights,
    "total_guests":                   total_guests,
    "room_match":                     room_match,
    "has_special_request":            has_special_request,
}])

# ── Predict ───────────────────────────────────────────────────────────────────
prediction  = pipeline.predict(sample)[0]
probability = pipeline.predict_proba(sample)[0]

cancel_prob = probability[1]
stay_prob   = probability[0]
label       = "CANCELLED" if prediction == 1 else "NOT CANCELLED"

print("=" * 50)
print("  BOOKING CANCELLATION PREDICTION")
print("=" * 50)
print("  Prediction         :", label)
print("  Cancellation Prob. :", str(round(cancel_prob * 100, 1)) + "%")
print("  Stay Prob.         :", str(round(stay_prob * 100, 1)) + "%")
print("=" * 50)

if cancel_prob >= 0.7:
    print("  [HIGH RISK]   Consider overbooking or deposit policy.")
elif cancel_prob >= 0.4:
    print("  [MEDIUM RISK] Monitor this booking.")
else:
    print("  [LOW RISK]    Booking likely to proceed.")
print()
