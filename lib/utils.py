import json 

"""
Sort a dictionary by its values.
"""
def sort_dict(d: dict) -> dict:
    return dict(sorted(d.items(), key=lambda item: item[1], reverse=True))

"""
Pretty print a dictionary.
"""
def print_dict(d: dict, sorted=False) -> None:
    if sorted:
        d = sort_dict(d)
    print(json.dumps(d, indent=4))


def lyrics_to_verses(lyrics: str) -> list:
    return lyrics.split("\n")
    
def verses_to_lyrics(verses: list) -> str:
    return "\n".join(verses)
