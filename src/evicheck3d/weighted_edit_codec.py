import json
from edit_codec import build_edits,serialize_edits

def serialize_with_replacement_ranges(source,target):
    edits=build_edits(source,target);text='[';ranges=[]
    for i,edit in enumerate(edits):
        if i:text+=','
        text+='{"old":'+json.dumps(edit['old'],ensure_ascii=False)+',"new":'
        start=len(text);text+=json.dumps(edit['new'],ensure_ascii=False)
        ranges.append((start,len(text)));text+='}'
    text+=']'
    assert text==serialize_edits(source,target)
    return text,ranges
