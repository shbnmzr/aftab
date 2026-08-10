def flush(message: str, **kwargs):
    kwargs["flush"] = True
    print(message, **kwargs)
