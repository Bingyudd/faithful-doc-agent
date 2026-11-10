import re


def numbers(text):
    """Literal numeric/measurement tokens, normalized only for case and whitespace."""
    return {re.sub(r'\s+',' ',x).casefold() for x in re.findall(
        r'(?<![\w.])\d+(?:\.\d+)?(?:\s*(?:mm|cm|MHz|Hz|°C|%))?(?!\w|\.\d)',text)}
