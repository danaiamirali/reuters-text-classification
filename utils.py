def config(attr):
    """
    Retrieves the queried attribute value from the config file. Loads the
    config file on first call.
    """
    if not hasattr(config, "config"):
        with open("config.json") as f:
            config.config = eval(f.read())
    node = config.config
    for part in attr.split("."):
        node = node[part]
    return node