"""Exact, original-input-anchored patches, supporting insertions and deletions."""
import difflib,json,re

def unique_anchor(source,old):
    first=source.find(old)
    return bool(old) and first>=0 and source.find(old,first+1)<0

def apply_edits(source, edits):
    if not isinstance(edits,list):raise ValueError('Expected a list of edits')
    ranges=[]
    for edit in edits:
        if not isinstance(edit,dict) or set(edit)!= {'old','new'}:raise ValueError('Invalid edit fields')
        old,new=edit['old'],edit['new']
        if not isinstance(old,str) or not isinstance(new,str) or not old:raise ValueError('Invalid edit strings')
        if not unique_anchor(source,old):raise ValueError('Anchor must appear exactly once in the input')
        start=source.index(old);ranges.append((start,start+len(old),new))
    ranges.sort()
    if any(a[1]>b[0] for a,b in zip(ranges,ranges[1:])):raise ValueError('Overlapping edits')
    result=source
    for a,b,new in reversed(ranges):result=result[:a]+new+result[b:]
    return result

def build_edits(source,target):
    if source==target:return []
    if not source:raise ValueError('An empty input cannot anchor a repair')
    a=re.findall(r'\s+|\S+',source);b=re.findall(r'\s+|\S+',target)
    opcodes=difflib.SequenceMatcher(None,a,b,autojunk=False).get_opcodes()
    intervals=[]
    for i,(tag,x,y,u,v) in enumerate(opcodes):
        if tag=='equal':continue
        if i and opcodes[i-1][0]=='equal':
            k=min(4,opcodes[i-1][2]-opcodes[i-1][1]);x-=k;u-=k
        if i+1<len(opcodes) and opcodes[i+1][0]=='equal':
            k=min(4,opcodes[i+1][2]-opcodes[i+1][1]);y+=k;v+=k
        if intervals and x<=intervals[-1][1]:
            prev=intervals[-1];intervals[-1]=(prev[0],max(y,prev[1]),prev[2],max(v,prev[3]))
        else:intervals.append((x,y,u,v))
    edits=[dict(old=''.join(a[x:y]),new=''.join(b[u:v])) for x,y,u,v in intervals]
    # Ambiguous repeated anchors are never guessed. Full-input replacement is
    # an explicit, exact fallback for training targets only.
    if any(not unique_anchor(source,e['old']) for e in edits):
        edits=[dict(old=source,new=target)]
    assert apply_edits(source,edits)==target
    return edits

def serialize_edits(source,target):
    return json.dumps(build_edits(source,target),ensure_ascii=False,separators=(',',':'))
