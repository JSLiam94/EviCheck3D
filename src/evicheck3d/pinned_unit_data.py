"""TRAIN-only fixed64 sparse projection and exact unit-pair edit supervision.
No model imports, private-data reads or materialization occur on import.
"""
import collections
import csv
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from edit_codec import apply_edits, unique_anchor
from frozen_unit_adapter import normalize_train_manifest, SOURCES

FIXED64='7935261e0ee110e3ef325bd9fe1ab6fbe182ccdc098f2c3c3cf45c41b28415c1'
# Historical materialization inputs must be supplied explicitly on the owning host.
SOURCE_PATHS = {
    'selection': os.environ.get('EVICHECK3D_LEGACY_SELECTION', ''),
    'TRAIN_current_private': os.environ.get('EVICHECK3D_LEGACY_TRAIN_CURRENT_PRIVATE', ''),
    'MIXED_manifest_bytes_only': os.environ.get('EVICHECK3D_LEGACY_MIXED_MANIFEST_BYTES_ONLY', ''),
    'TRAIN_allowed_split_metadata': os.environ.get('EVICHECK3D_LEGACY_TRAIN_ALLOWED_SPLIT_METADATA', ''),
    'TRAIN_exclusion_policy': os.environ.get('EVICHECK3D_LEGACY_TRAIN_EXCLUSION_POLICY', ''),
}
EXCLUSION_SHA='eb383a25a166a4a255dcf1964359445d6a5fe459f9b5ca8642e925ecb5c9351e'
PROGRESS_PROTOCOL='unit-program-repair-v3-materialize-progress-v1'
PROGRESS_AUDIT_INTERVAL_SECONDS=0.5
PROGRESS_WRITE_INTERVAL_SECONDS=2.0
PROGRESS_STAGES=('SOURCE_BINDING','INITIAL_SOURCE_HASH','SELECTION_PARSE','EXCLUSION_POLICY_PARSE',
    'TRAIN_JSON_SCAN','ALLOWED_CSV_SCAN','MIXED_CSV_SCAN','FIXED64_TARGET_BUILD',
    'FINAL_SOURCE_HASH','PROJECTION_ENCODE','PROJECTION_WRITE')
PROGRESS_COUNTERS=('train_rows','csv_records','selected_rows','source_bytes_expected','source_bytes_read',
    'projection_bytes_written','hash_blocks','realpath_checks','progress_checks','expensive_audit_checks')
REFUSALS=('MISSING_TARGET','MISSING_UNIT','AMBIGUOUS_OLD','OVERLAP_OR_INVALID',
          'RECONSTRUCTION_MISMATCH','NONLOCAL_UNIT','IDENTITY_ONLY_ERROR','NORMAL_MISMATCH')
def need(ok,code):
    if not ok:raise ValueError(code)
def encoded(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(value):return hashlib.sha256(encoded(value)).hexdigest()
def strict_json(raw):
    def pairs(items):
        out={}
        for key,value in items:
            need(key not in out,'DUPLICATE_JSON_KEY');out[key]=value
        return out
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _:need(False,'NONFINITE_JSON'))
class MaterializeProgress:
    """Throttle expensive audits to 0.5 s while checking deadline on each call."""
    def __init__(self,path,*,deadline,audit,clock=time.monotonic):
        self.path=Path(path);self.deadline=deadline;self.audit=audit;self.clock=clock
        now=clock();self.started=now;self.status='ACTIVE';self.current=None;self.stages=[]
        self.projection_sha256=None;self.sequence=0;self.last_audit=float('-inf');self.last_write=float('-inf')
        self.totals={key:0 for key in PROGRESS_COUNTERS}
        need(self.path.is_absolute() and self.path.parent.resolve(strict=True)==self.path.parent,'PROGRESS_PATH_REFUSED')
        need(not self.path.exists() and not self.path.is_symlink(),'PROGRESS_FILE_EXISTS')
        self._write(now)

    def _duration(self,start,now):
        value=max(0.0,now-start);need(math.isfinite(value),'PROGRESS_DURATION_INVALID');return round(value,6)

    def _snapshot(self,now):
        result=dict(protocol=PROGRESS_PROTOCOL,status_enum=self.status,
            current_stage_enum=self.current['stage_enum'] if self.current else 'NONE',
            elapsed_seconds=self._duration(self.started,now),**self.totals,
            stages=[dict(stage_enum=row['stage_enum'],status_enum=row['status_enum'],
                elapsed_seconds=row['elapsed_seconds'],**row['counts']) for row in self.stages])
        if self.projection_sha256 is not None:result['projection_sha256']=self.projection_sha256
        return result

    def _write(self,now=None):
        now=self.clock() if now is None else now
        raw=encoded(self._snapshot(now));self.sequence+=1
        pending=self.path.with_name(self.path.name+'.pending.'+str(self.sequence).zfill(8))
        fd=os.open(pending,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600)
        try:
            view=memoryview(raw)
            while view:
                wrote=os.write(fd,view);need(wrote>0,'PROGRESS_WRITE_REFUSED');view=view[wrote:]
            os.fsync(fd)
        finally:os.close(fd)
        os.replace(pending,self.path)
        dfd=os.open(self.path.parent,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
        try:os.fsync(dfd)
        finally:os.close(dfd)
        self.last_write=self.clock()

    def count(self,**values):
        if set(values)-set(PROGRESS_COUNTERS):need(False,'PROGRESS_COUNTER_REFUSED')
        for key,value in values.items():
            need(type(value) is int and value>=0,'PROGRESS_COUNTER_REFUSED')
            self.totals[key]+=value
            if self.current is not None:self.current['counts'][key]+=value

    def __call__(self,*,force=False):
        now=self.clock();self.totals['progress_checks']+=1
        if self.current is not None:self.current['counts']['progress_checks']+=1
        if now>=self.deadline:
            if self.current is None:self.fail()
            need(False,'MATERIALIZE_PHASE_CAP')
        if force or now-self.last_audit>=PROGRESS_AUDIT_INTERVAL_SECONDS:
            self.audit();self.last_audit=self.clock()
            self.totals['expensive_audit_checks']+=1
            if self.current is not None:self.current['counts']['expensive_audit_checks']+=1
        now=self.clock()
        if self.current is not None and now-self.last_write>=PROGRESS_WRITE_INTERVAL_SECONDS:self._write(now)

    @contextmanager
    def stage(self,stage_enum):
        need(stage_enum in PROGRESS_STAGES and self.current is None,'PROGRESS_STAGE_REFUSED')
        try:self(force=True)
        except BaseException:
            self.fail();raise
        now=self.clock();record=dict(stage_enum=stage_enum,status_enum='ACTIVE',started=now,
            elapsed_seconds=0.0,counts={key:0 for key in PROGRESS_COUNTERS})
        self.stages.append(record);self.current=record;self._write(now)
        try:
            yield
            self(force=True)
        except BaseException:
            now=self.clock();record['status_enum']='FAILED';record['elapsed_seconds']=self._duration(record['started'],now)
            record.pop('started',None);self.current=None;self.status='FAILED';self._write(now);raise
        else:
            now=self.clock();record['status_enum']='COMPLETE';record['elapsed_seconds']=self._duration(record['started'],now)
            record.pop('started',None);self.current=None;self._write(now)

    def fail(self):
        now=self.clock()
        if self.current is not None:
            self.current['status_enum']='FAILED';self.current['elapsed_seconds']=self._duration(self.current['started'],now)
            self.current.pop('started',None);self.current=None
        self.status='FAILED';self._write(now)

    def complete(self,projection_sha256):
        need(type(projection_sha256) is str and len(projection_sha256)==64 and
            all(ch in '0123456789abcdef' for ch in projection_sha256),'PROJECTION_SHA_REQUIRED')
        need(self.current is None and self.status=='ACTIVE','PROGRESS_COMPLETE_REFUSED')
        self.projection_sha256=projection_sha256;self.status='COMPLETE';self._write()

    def receipt_sha256(self):
        with self.path.open('rb') as stream:return hashlib.sha256(stream.read()).hexdigest()


def file_sha(path,tick=lambda:None,*,on_block=None,on_boundary=None):
    path=Path(path)
    if on_boundary:on_boundary()
    need(path.is_absolute() and path.resolve(strict=True)==path and path.is_file() and not path.is_symlink(),'SOURCE_FILE_REFUSED')
    before=path.stat();h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            tick()
            if on_block:on_block(len(block))
            h.update(block)
    after=path.stat();need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'SOURCE_CHANGED')
    if on_boundary:on_boundary()
    return h.hexdigest()
def bounded_json(path,limit=32*2**20):
    need(Path(path).stat().st_size<=limit,'BOUNDED_JSON_CAP')
    return strict_json(Path(path).read_bytes())

def stream_array(path,tick=lambda:None,on_block=None):
    """One JSON object at a time; never constructs the 85k-row container."""
    def pairs(items):
        value={}
        for key,item in items:need(key not in value,'DUPLICATE_JSON_KEY');value[key]=item
        return value
    decoder=json.JSONDecoder(object_pairs_hook=pairs,parse_constant=lambda _:need(False,'NONFINITE_JSON'))
    with Path(path).open(encoding='utf-8') as stream:
        buf='';eof=False
        def fill():
            nonlocal buf,eof
            block=stream.read(65536);tick()
            if block and on_block:on_block(len(block))
            eof=not block;buf+=block
            need(len(buf)<=8*2**20,'SOURCE_ROW_CAP')
        def space():
            nonlocal buf
            while True:
                buf=buf.lstrip()
                if buf or eof:return
                fill()
        space();need(buf.startswith('['),'TRAIN_ARRAY_REQUIRED');buf=buf[1:];first=True
        while True:
            space()
            if buf.startswith(']'):buf=buf[1:];break
            if not first:
                need(buf.startswith(','),'TRAIN_ARRAY_INVALID');buf=buf[1:];space()
            need(buf.startswith('{'),'TRAIN_ROW_OBJECT_REQUIRED')
            while True:
                try:value,end=decoder.raw_decode(buf);break
                except json.JSONDecodeError:
                    need(not eof,'TRAIN_ARRAY_INVALID');fill()
            yield value;buf=buf[end:];first=False
        while not eof:fill()
        need(not buf.strip(),'TRAIN_TRAILING_DATA')

def fixed_selection(selection):
    selected=selection['selected'];need(len(selected)==640 and len({x['case_id'] for x in selected})==640,'SELECTION_DENOMINATOR')
    fit={kind:[] for kind in range(8)};dev=[]
    for row in selected:
        need(row['primary_type'] in range(8),'TYPE_REFUSED')
        if row['split']=='train_fit':fit[row['primary_type']].append(row)
        elif row['split']=='train_internal_dev':dev.append(row)
        else:raise ValueError('NON_TRAIN_SELECTION')
    need(all(len(fit[k])==64 for k in fit) and len(dev)==128,'SELECTION_DENOMINATOR')
    rng=random.Random(20261024)
    for kind in range(8):rng.shuffle(fit[kind])
    chosen=[dict(fit[k][i],internal_split='fit') for i in range(4) for k in range(8)]
    seen=collections.Counter()
    for row in dev:
        kind=row['primary_type']
        if seen[kind]<4:chosen.append(dict(row,internal_split='dev'))
        seen[kind]+=1
    need(all(seen[k]==16 for k in range(8)),'DEV_TYPE_DENOMINATOR')
    signature=hashlib.sha256(json.dumps([(r['internal_split'],r['case_id'],r['primary_type']) for r in chosen],separators=(',',':')).encode()).hexdigest()
    need(signature==FIXED64,'FIXED64_SELECTION_CHANGED')
    groups={s:{r['ct_key'] for r in chosen if r['internal_split']==s} for s in ('fit','dev')}
    need(all(len(x)==32 for x in groups.values()) and not groups['fit']&groups['dev'],'CT_SPLIT_OVERLAP')
    return chosen

def local_pair(report,old,new):
    need(isinstance(old,str) and isinstance(new,str) and old and new,'MISSING_UNIT')
    if old==new:return None
    need(unique_anchor(report,old),'AMBIGUOUS_OLD')
    if old==report:
        # Only this annotated pair is trimmed; no whole-report diff/fallback.
        a=0
        while a<min(len(old),len(new)) and old[a]==new[a]:a+=1
        tail=0
        while tail<min(len(old)-a,len(new)-a) and old[-1-tail]==new[-1-tail]:tail+=1
        b=len(old)-tail;d=len(new)-tail
        left,right=a,b
        while not unique_anchor(report,old[left:right]):
            if left>0:left-=1
            elif right<len(old):right+=1
            else:raise ValueError('NONLOCAL_UNIT')
        old,new=old[left:right],new[left:d+(right-b)]
        need(old!=report and old!=new,'NONLOCAL_UNIT')
    return dict(old=old,new=new)

def serialize_program(edits):
    text='[';ranges=[]
    for index,edit in enumerate(edits):
        if index:text+=','
        text+='{"old":'+json.dumps(edit['old'],ensure_ascii=False)+',"new":'
        start=len(text);text+=json.dumps(edit['new'],ensure_ascii=False);ranges.append([start,len(text)]);text+='}'
    return text+']',ranges

def decode_program(report,raw):
    edits=strict_json(raw)
    if not isinstance(edits,list):raise ValueError('PROGRAM_NOT_ARRAY')
    for edit in edits:
        if not isinstance(edit,dict) or set(edit)!={'old','new'}:raise ValueError('PROGRAM_SCHEMA')
        if edit['old']==report:raise ValueError('WHOLE_REPORT_EDIT_REFUSED')
        if edit['old']==edit['new']:raise ValueError('IDENTITY_EDIT_REFUSED')
    result=apply_edits(report,edits)
    if not result.strip():raise ValueError('EMPTY_REPORT_REFUSED')
    return result,edits

def compile_target(report,reference,pairs,primary_type):
    assessed=[p for p in pairs if isinstance(p,dict) and isinstance(p.get('error_unit'),str) and isinstance(p.get('source_unit'),str)]
    identity=sum(p['error_unit']==p['source_unit'] for p in assessed)
    counts=dict(annotation_pairs=len(pairs),identity_pairs=identity,nonidentity_pairs=len(assessed)-identity,unassessed_pairs=len(pairs)-len(assessed),
        ambiguous_old_pairs=sum(bool(p['error_unit']) and report.find(p['error_unit'],report.find(p['error_unit'])+1)>=0 for p in assessed),
        missing_old_pairs=sum(not p['error_unit'] or p['error_unit'] not in report for p in assessed),
        missing_reference_target_pairs=sum(not p['source_unit'] or not isinstance(reference,str) or p['source_unit'] not in reference for p in assessed))
    try:
        need(isinstance(reference,str) and reference,'MISSING_TARGET')
        edits=[]
        for pair in pairs:
            old,new=pair['error_unit'],pair['source_unit']
            need(isinstance(new,str) and new and new in reference,'MISSING_TARGET')
            if old==new:continue
            edits.append(local_pair(report,old,new))
        if primary_type==0:need(report==reference and not edits,'NORMAL_MISMATCH')
        else:need(bool(edits),'IDENTITY_ONLY_ERROR')
        try:rebuilt=apply_edits(report,edits)
        except (ValueError,TypeError):raise ValueError('OVERLAP_OR_INVALID')
        need(rebuilt==reference,'RECONSTRUCTION_MISMATCH')
        target,ranges=serialize_program(edits)
        return dict(target_status='READY',target=target,new_value_ranges=ranges,edits=edits,**counts)
    except (ValueError,KeyError,TypeError) as exc:
        reason=str(exc) if str(exc) in REFUSALS else 'MISSING_UNIT'
        return dict(target_status='REFUSED',refusal_enum=reason,target=None,new_value_ranges=[],edits=None,**counts)

def materialize(bindings,destination,tick):
    """Owner-only fixed64 preparation with throttled progress and preserved checks."""
    sources=bindings['sources']
    with tick.stage('SOURCE_BINDING'):
        need(set(sources)==set(SOURCE_PATHS),'SOURCE_ROLE_SET')
        need(all(os.path.isabs(SOURCE_PATHS[role]) and item['path']==SOURCE_PATHS[role] for role,item in sources.items()),'TRAIN_SOURCE_PATH_REFUSED')
        hashes={k:sources[k]['sha256'] for k in ('TRAIN_current_private','MIXED_manifest_bytes_only','TRAIN_allowed_split_metadata')}
        need(all(hashes[k]==SOURCES[k] for k in hashes) and sources['TRAIN_exclusion_policy']['sha256']==EXCLUSION_SHA,'ACTUAL_UNIT_CONTRACT_WHOLE_SOURCE_REFUSED')

    def source_hash(item):
        return file_sha(item['path'],tick,
            on_block=lambda count:tick.count(source_bytes_read=count,hash_blocks=1),
            on_boundary=lambda:tick(force=True))

    with tick.stage('INITIAL_SOURCE_HASH'):
        for role,item in sources.items():
            tick.count(source_bytes_expected=Path(item['path']).stat().st_size)
            need(source_hash(item)==item['sha256'],'SOURCE_SHA_REFUSED')

    with tick.stage('SELECTION_PARSE'):
        selection_path=Path(sources['selection']['path'])
        tick.count(source_bytes_expected=selection_path.stat().st_size)
        chosen=fixed_selection(bounded_json(selection_path,4*2**20));keys={r['case_id'] for r in chosen}
        tick.count(source_bytes_read=selection_path.stat().st_size,selected_rows=len(chosen))

    with tick.stage('EXCLUSION_POLICY_PARSE'):
        policy_path=Path(sources['TRAIN_exclusion_policy']['path'])
        tick.count(source_bytes_expected=policy_path.stat().st_size)
        policy=bounded_json(policy_path,1024*1024)
        tick.count(source_bytes_read=policy_path.stat().st_size)
        need(policy['reason']=='confirmed_truncated_CT_payload' and policy['user_authorized_exclusion'] is True and policy['source_files_changed'] is False and len(policy['images'])==3,'EXCLUSION_POLICY_REFUSED')
        excluded=set()
        for item in policy['images']:
            excluded.add(os.path.realpath(item['path']));tick.count(realpath_checks=1)
        need(len(excluded)==3,'EXCLUSION_POLICY_REFUSED')
        need(all(Path(x['path']).stat().st_size==x['compressed_bytes'] for x in policy['images']),'EXCLUSION_METADATA_CHANGED')

    reports={};original=effective=0
    with tick.stage('TRAIN_JSON_SCAN'):
        tick.count(source_bytes_expected=Path(sources['TRAIN_current_private']['path']).stat().st_size)
        for row in stream_array(sources['TRAIN_current_private']['path'],tick,
                on_block=lambda count:tick.count(source_bytes_read=count)):
            tick()
            original+=1;valid=os.path.realpath(row['image_path']) not in excluded;effective+=valid
            tick.count(train_rows=1,realpath_checks=1)
            if row['case_id'] in keys:
                need(valid and row['case_id'] not in reports,'SELECTED_EXCLUDED_OR_DUPLICATE');reports[row['case_id']]=row
        need(original==85625 and effective==85615,'EFFECTIVE_TRAIN_DENOMINATOR_CHANGED')

    selected_sources={}
    for role,key,stage_enum in (
        ('TRAIN_allowed_split_metadata','case_id','ALLOWED_CSV_SCAN'),
        ('MIXED_manifest_bytes_only','BDMAP ID','MIXED_CSV_SCAN'),
    ):
        with tick.stage(stage_enum):
            found={};csv.field_size_limit(8*2**20)
            csv_source_path=Path(sources[role]['path'])
            tick.count(source_bytes_expected=csv_source_path.stat().st_size)
            with csv_source_path.open(newline='',encoding='utf-8-sig') as stream:
                reader=csv.DictReader(stream);need(reader.fieldnames and len(reader.fieldnames)==len(set(reader.fieldnames)),'CSV_HEADER_REFUSED')
                for row in reader:
                    tick();tick.count(csv_records=1)
                    # Filter mixed roles BEFORE decoding any annotation/report field.
                    if role=='MIXED_manifest_bytes_only' and row.get('ecfst_split')!='train':continue
                    if row.get(key) in keys:
                        need(row[key] not in found,'DUPLICATE_SELECTED_ROW');found[row[key]]=row
            need(set(found)==keys,'SELECTED_SOURCE_ROW_MISSING')
            tick.count(source_bytes_read=Path(sources[role]['path']).stat().st_size)
            selected_sources[role]=found

    need(set(reports)==keys,'SELECTED_SOURCE_ROW_MISSING')
    rows=[]
    with tick.stage('FIXED64_TARGET_BUILD'):
        for meta in chosen:
            tick()
            row=reports[meta['case_id']];manifest=selected_sources['MIXED_manifest_bytes_only'][meta['case_id']]
            allowed=selected_sources['TRAIN_allowed_split_metadata'][meta['case_id']]
            need(row['case_id']==manifest['BDMAP ID']==allowed['case_id']==meta['case_id'],'TRAIN_CROSS_SOURCE_KEY_MISMATCH')
            need(row['source_dataset']+'\0'+row['source_case_id']==meta['ct_key'],'CT_KEY_CHANGED')
            image=Path(row['image_path']);st=image.stat()
            stamp=hashlib.sha256(f'{image.resolve()}\0{st.st_size}\0{st.st_mtime_ns}'.encode()).hexdigest()
            tick.count(realpath_checks=1)
            need(stamp==meta['ct_file_stat_sha256'],'CT_IDENTITY_CHANGED')
            provenance=dict(whole_source_hashes=hashes,exclusion_policy_sha256=EXCLUSION_SHA,
                current_private_row=row,manifest_row=manifest,allowed_split_row=allowed,selection_metadata=meta)
            provenance['record_binding_sha256']=digest(provenance)
            pairs,target=target_from_provenance(provenance)
            rows.append(dict(case_id=meta['case_id'],internal_split=meta['internal_split'],primary_type=meta['primary_type'],
                ct_key=meta['ct_key'],ct_stat_sha256=stamp,image_memory=row['image_path'],report=row['report'],reference=row['reference'],
                source_provenance=provenance,supervision_pairs=pairs,**target))
        physical={'fit':set(),'dev':set()}
        for row in rows:
            tick();physical[row['internal_split']].add(os.path.realpath(row['image_memory']))
            tick.count(realpath_checks=1)
        need(all(len(v)==32 for v in physical.values()) and not physical['fit']&physical['dev'],'CT_FILE_SPLIT_OVERLAP')

    with tick.stage('FINAL_SOURCE_HASH'):
        for item in sources.values():
            tick.count(source_bytes_expected=Path(item['path']).stat().st_size)
            need(source_hash(item)==item['sha256'],'SOURCE_CHANGED')

    with tick.stage('PROJECTION_ENCODE'):
        body=dict(protocol='unit-program-fixed64-projection-v3',fixed64_selection_sha256=FIXED64,sources=sources,rows=rows)
        body['targets_sha256']=digest([(r['case_id'],r['target_status'],r['target'],r.get('refusal_enum')) for r in rows])
        raw=encoded(body);need(len(raw)<=32*2**20,'PROJECTION_CAP')

    with tick.stage('PROJECTION_WRITE'):
        tick(force=True)
        fd=os.open(destination,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        try:need(os.write(fd,raw)==len(raw),'PROJECTION_WRITE');os.fsync(fd)
        finally:os.close(fd)
        tick.count(projection_bytes_written=len(raw))
    return hashlib.sha256(raw).hexdigest()

def target_from_provenance(provenance):
    body={k:v for k,v in provenance.items() if k!='record_binding_sha256'}
    need(digest(body)==provenance['record_binding_sha256'],'TRAIN_ROW_BINDING_CHANGED')
    row=provenance['current_private_row'];manifest=provenance['manifest_row'];meta=provenance['selection_metadata']
    need(row['case_id']==manifest['BDMAP ID']==provenance['allowed_split_row']['case_id']==meta['case_id'],'TRAIN_CROSS_SOURCE_KEY_MISMATCH')
    need(provenance['exclusion_policy_sha256']==EXCLUSION_SHA,'EXCLUSION_POLICY_REFUSED')
    n=normalize_train_manifest(manifest,row,source_hashes=provenance['whole_source_hashes'],
        record_binding_sha256=provenance['record_binding_sha256'],effective_train_membership_verified=True)
    need(meta['primary_type'] in n['annotation_labels'],'PRIMARY_TYPE_NOT_IN_LITERAL_LABELS')
    if n['flat_status']!='PAIRED_POSITIONAL_LISTS' or n['typed_status']!='EXACT_TYPED_CONCATENATION':
        return [],dict(target_status='REFUSED',refusal_enum='UNIT_CONTRACT_PENDING',target=None,new_value_ranges=[],edits=None,
            annotation_pairs=n['annotation_pair_count'],identity_pairs=0,nonidentity_pairs=0,unassessed_pairs=n['annotation_pair_count'])
    pairs=[dict(type_id=p['annotation_type_id'],error_unit=p['error_text'],source_unit=p['target_text']) for p in n['typed_pairs']]
    return pairs,compile_target(row['report'],row['reference'],pairs,meta['primary_type'])

def load_projection(path,pin):
    need(file_sha(path)==pin,'PROJECTION_SHA_REFUSED');data=bounded_json(path)
    need(data['protocol']=='unit-program-fixed64-projection-v3','PROJECTION_PROTOCOL_REFUSED')
    rows=data['rows'];need(data['fixed64_selection_sha256']==FIXED64 and len(rows)==64,'FIXED64_DENOMINATOR')
    signature=hashlib.sha256(json.dumps([(r['internal_split'],r['case_id'],r['primary_type']) for r in rows],separators=(',',':')).encode()).hexdigest()
    need(signature==FIXED64,'FIXED64_ORDER_CHANGED')
    need(len({r['case_id'] for r in rows})==64,'DUPLICATE_SELECTED_ROW')
    counts={s:collections.Counter(r['primary_type'] for r in rows if r['internal_split']==s) for s in ('fit','dev')}
    need(all(counts[s]==collections.Counter({k:4 for k in range(8)}) for s in counts),'FIXED64_TYPE_DENOMINATOR')
    clusters={s:{r['ct_key'] for r in rows if r['internal_split']==s} for s in counts}
    need(all(len(v)==32 for v in clusters.values()) and not clusters['fit']&clusters['dev'],'CT_SPLIT_OVERLAP')
    for row in rows:
        pairs,expected=target_from_provenance(row['source_provenance'])
        origin=row['source_provenance']['current_private_row'];meta=row['source_provenance']['selection_metadata']
        need(row['supervision_pairs']==pairs and row['report']==origin['report'] and row['reference']==origin['reference'] and row['case_id']==origin['case_id'] and row['image_memory']==origin['image_path'] and row['primary_type']==meta['primary_type'] and row['internal_split']==meta['internal_split'],'PROJECTION_ROW_BINDING_CHANGED')
        need(all(row.get(k)==v for k,v in expected.items()),'TARGET_RECONSTRUCTION_CHANGED')
    physical={s:{os.path.realpath(r['image_memory']) for r in rows if r['internal_split']==s} for s in ('fit','dev')}
    need(all(len(v)==32 for v in physical.values()) and not physical['fit']&physical['dev'],'CT_FILE_SPLIT_OVERLAP')
    need(data['targets_sha256']==digest([(r['case_id'],r['target_status'],r['target'],r.get('refusal_enum')) for r in rows]),'TARGET_SHA_REFUSED')
    summary=dict(planned_reports=64,fit_reports=32,dev_reports=32,atomic_fact_denominator=None,
        by_split={s:dict(annotation_pairs_known=sum(r['annotation_pairs'] or 0 for r in rows if r['internal_split']==s),annotation_pair_count_NA_reports=sum(r['annotation_pairs'] is None for r in rows if r['internal_split']==s),
            reports=32,primary_type_reports={str(k):counts[s][k] for k in range(8)},ready_reports=sum(r['target_status']=='READY' for r in rows if r['internal_split']==s),
            identity_pairs_known=sum(r['identity_pairs'] or 0 for r in rows if r['internal_split']==s),
            unassessed_pairs_known=sum(r['unassessed_pairs'] or 0 for r in rows if r['internal_split']==s),
            nonidentity_annotation_pairs_known=sum(r['nonidentity_pairs'] or 0 for r in rows if r['internal_split']==s),
            ambiguous_old_pairs_known=sum(r.get('ambiguous_old_pairs') or 0 for r in rows if r['internal_split']==s),
            missing_old_pairs_known=sum(r.get('missing_old_pairs') or 0 for r in rows if r['internal_split']==s),
            missing_reference_target_pairs_known=sum(r.get('missing_reference_target_pairs') or 0 for r in rows if r['internal_split']==s),
            refusals=dict(collections.Counter(r.get('refusal_enum') for r in rows if r['internal_split']==s and r['target_status']!='READY'))) for s in counts})
    need(file_sha(path)==pin,'PROJECTION_CHANGED')
    return data,summary
