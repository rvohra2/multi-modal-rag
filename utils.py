import uuid
import base64

def get_or_create_uuid(session_state):
    """ Retrieves an existing UUID from session state or generates a new one if missing. """
    if session_state.get("user_uuid") is None:
        session_state["user_uuid"] = str(uuid.uuid4())
    return session_state["user_uuid"]

