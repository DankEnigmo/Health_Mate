import pyttsx3

_engine = None

def _init_engine():
    global _engine
    if _engine is None:
        _engine = pyttsx3.init()
        _engine.setProperty('rate', 150)
        _engine.setProperty('volume', 1.0)

def speak(text: str):
    if not text:
        return
    _init_engine()
    _engine.say(text)
    _engine.runAndWait()
