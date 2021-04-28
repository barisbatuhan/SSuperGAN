from datetime import datetime


def get_dt_string():
    now = datetime.now()
    # dd/mm/YY-H:M:S
    dt_string = now.strftime("%d/%m/%Y-%H:%M:%S")
    return dt_string
