import sqlite3
import datetime

def adapt_date_iso(val):
    """Adapt datetime.date to ISO 8601 date."""
    return val.isoformat()

def adapt_datetime_iso(val):
    """Adapt datetime.datetime to timezone-naive ISO 8601 date."""
    return val.isoformat()

def adapt_datetime_epoch(val):
    """Adapt datetime.datetime to Unix timestamp."""
    return int(val.timestamp())


def convert_date(val):
    """Convert ISO 8601 date to datetime.date object."""
    return datetime.date.fromisoformat(val.decode())

def convert_datetime(val):
    """Convert ISO 8601 datetime to datetime.datetime object."""
    return datetime.datetime.fromisoformat(val.decode())

def convert_timestamp(val):
    """Convert Unix epoch timestamp to datetime.datetime object."""
    return datetime.datetime.fromtimestamp(int(val))




# In the end, I did not have to extend the sqlite with custom types
# Keeping this for legacy reasons and I do no want to have to refactor this function everywhere again
def connect_to_sqlite(*args, **kwargs):
    """
    Passing all args to sqlite3.connect
    But also registeres all needed adapters and converters
    """

    sqlite3.register_adapter(datetime.date, adapt_date_iso)
    sqlite3.register_adapter(datetime.datetime, adapt_datetime_iso)
    # sqlite3.register_adapter(datetime.datetime, adapt_datetime_epoch)
    sqlite3.register_converter("date", convert_date)
    sqlite3.register_converter("datetime", convert_datetime)
    # sqlite3.register_converter("timestamp", convert_timestamp)

    return sqlite3.connect(*args, **kwargs, detect_types=sqlite3.PARSE_DECLTYPES)
