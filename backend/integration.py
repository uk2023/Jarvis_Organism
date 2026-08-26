# -*- coding: utf-8 -*-
"""
The three "glue" hooks the rest of the Jarvis Organism project calls into
this web server with. Kept as their own tiny module so routes_ws.py can
grab the current executor without importing all of server.py.
"""
import sys

jarvis = None
brain = None
query_executor_func = None


def set_shared_organism(shared_jarvis, shared_brain=None):
    global jarvis, brain
    jarvis = shared_jarvis
    brain = shared_brain or (
        shared_jarvis.get_organ("brain") if shared_jarvis else None
    )


def bind_query_executor(executor_func):
    global query_executor_func
    query_executor_func = executor_func


def get_query_executor():
    """Falls back to a top-level process_query() if nothing was explicitly bound."""
    return query_executor_func or getattr(
        sys.modules.get("__main__"), "process_query", None
    )
