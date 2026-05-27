import json
from datetime import datetime, timedelta


def get_flight_info(loc_origin, loc_destination):
    """Get flight information between two locations."""
    flight_info = {
        "loc_origin": loc_origin,
        "loc_destination": loc_destination,
        "result_label": "from_file",
        "datetime": str(datetime.now() + timedelta(hours=2)),
        "airline": "KLM",
        "flight": "KL643",
    }

    return json.dumps(flight_info)


def book_flight(loc_origin, loc_destination, datetime, airline):
    """Book a flight based on flight information."""
    book_info = {
        "loc_origin": loc_origin,
        "loc_destination": loc_destination,
        "result_label": "from_file",
        "datetime": datetime,
        "airline": airline,
        "bookstatus": "Success",
    }

    return json.dumps(book_info)


def file_complaint(name, email, text):
    """File a complaint as a customer."""
    complaint_info = {
        "customername": name,
        "customeremail": email,
        "result_label": "from_file",
        "complaintcontent": text,
        "submitstatus": "Success",
    }

    return json.dumps(complaint_info)