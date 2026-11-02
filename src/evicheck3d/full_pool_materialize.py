"""One admitted streaming full-TRAIN index build; no tokenizer, model or CT read."""
import argparse,csv,hashlib,io,json,os,resource,signal,sqlite3,stat,sys,time
from pathlib import Path

GIB=1<<30;MIB=1<<20

def need(ok,code):
    if not ok:raise RuntimeError(code)

def canonical(v):return json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()+b'\n'
def sha(raw):return hashlib.sha256(raw).hexdigest()

def read_small(path,pin,limit=16*MIB):
    p=Path(path);need(p.resolve(strict=True)==p,'METADATA_CANONICAL');fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb') as f:
        a=os.fstat(f.fileno());need(stat.S_ISREG(a.st_mode) and a.st_size<=limit,'METADATA_CAP');raw=f.read(limit+1);b=os.fstat(f.fileno())
    z=p.lstat();fields=('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')
    need(len(raw)==a.st_size and all(getattr(a,k)==getattr(b,k)==getattr(z,k) for k in fields) and sha(raw)==pin,'METADATA_CHANGED_OR_PIN')
    return json.loads(raw)


def stream_array(path,pin,guard):
    """Single content pass: hash exactly the bytes decoded, retain only one row."""
    p=Path(path);target=p.resolve(strict=True);fd=os.open(target,os.O_RDONLY|os.O_NOFOLLOW)
    rawfile=os.fdopen(fd,'rb',buffering=0);before=os.fstat(fd)
    need(stat.S_ISREG(before.st_mode) and 0<before.st_size<=2*GIB,'FULL_TRAIN_SOURCE_BOUND')
    h=hashlib.sha256()
    class HashReader(io.RawIOBase):
        def readable(self):return True
        def readinto(self,b):
            data=rawfile.read(len(b));h.update(data);b[:len(data)]=data;return len(data)
    reader=io.TextIOWrapper(io.BufferedReader(HashReader()),encoding='utf-8')
    decoder=json.JSONDecoder();buf='';eof=False
    def fill():
        nonlocal buf,eof
        guard();part=reader.read(65536);eof=not part;buf+=part;need(len(buf)<=4*MIB,'SINGLE_TRAIN_ROW_CAP')
    def trim():
        nonlocal buf
        buf=buf.lstrip()
        while not buf and not eof:fill();buf=buf.lstrip()
    try:
        trim();need(buf.startswith('['),'FULL_TRAIN_ARRAY');buf=buf[1:];first=True
        while True:
            trim()
            if buf.startswith(']'):buf=buf[1:];break
            if not first:
                need(buf.startswith(','),'FULL_TRAIN_ARRAY_DELIMITER');buf=buf[1:];trim()
            while True:
                try:item,end=decoder.raw_decode(buf);break
                except json.JSONDecodeError:
                    need(not eof,'FULL_TRAIN_JSON_TRUNCATED');fill()
            need(type(item) is dict,'FULL_TRAIN_ROW_OBJECT');buf=buf[end:];first=False;yield item
        trim();need(not buf and eof,'FULL_TRAIN_TRAILING_DATA')
        after=os.fstat(fd);named=target.lstat();fields=('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')
        need(all(getattr(before,k)==getattr(after,k)==getattr(named,k) for k in fields) and h.hexdigest()==pin,'FULL_TRAIN_SOURCE_CHANGED_OR_PIN')
    finally:reader.close();rawfile.close()


def write_new(path,value):
    raw=canonical(value);need(len(raw)<=16*MIB,'OUTPUT_METADATA_CAP')
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return sha(raw)


def materialize(c,guard):
    from full_pool_adapter import adapt_existing_training_row
    scope=Path(c['scope'])
    need(bool(os.environ.get('EVICHECK3D_LEGACY_CONTROL')),'LEGACY_CONTROL_NOT_CONFIGURED')
    control=Path(os.environ['EVICHECK3D_LEGACY_CONTROL'])
    need(control.is_absolute(),'LEGACY_CONTROL_NOT_ABSOLUTE')
    need(set(c['sources'])=={'public','center'} and
      Path(c['sources']['public']['path'])==control/'train_public_private.json' and
      Path(c['sources']['center']['path'])==control/'train_center_private.json','EXACT_FULL_TRAIN_SOURCES')
    need(Path(c['exclusion_policy']['path'])==control/'training_exclusions_private.json' and
      c['exclusion_policy']['sha256']=='eb383a25a166a4a255dcf1964359445d6a5fe459f9b5ca8642e925ecb5c9351e','EXACT_FROZEN_DAMAGE_POLICY')
    members=read_small(c['train_members']['path'],c['train_members']['sha256'])
    allowed=set(members['member_ids']);need(len(allowed)==len(members['member_ids'])>0,'TRAIN_MEMBER_SET')
    from full_pool_protected import collect
    protected=collect(read_small);protected_pin=write_new(scope/'protected_metadata_private.json',protected)
    protected_ids=set(protected['case_ids']);protected_paths=set(protected['image_paths'])
    protected_source_keys={tuple(x) for x in protected['source_ct_keys']};protected_opaque=set(protected['opaque_ct_keys'])
    protected_ct=set();matched_protected_ids=set()
    policy=read_small(c['exclusion_policy']['path'],c['exclusion_policy']['sha256'],1*MIB)
    need(policy['reason']=='confirmed_truncated_CT_payload' and policy['user_authorized_exclusion'] is True,'FROZEN_TECHNICAL_EXCLUSION')
    damaged={os.path.realpath(x['path']) for x in policy['images']}
    pending=scope/'full_train_index.sqlite.pending';final=scope/'full_train_index.sqlite'
    fd=os.open(pending,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600);os.close(fd)
    db=sqlite3.connect(pending);db.execute('PRAGMA journal_mode=DELETE');db.execute('PRAGMA synchronous=FULL');db.execute('PRAGMA temp_store=MEMORY')
    db.execute('CREATE TABLE members(case_id TEXT PRIMARY KEY,is_error INTEGER NOT NULL,ct_key TEXT NOT NULL,order_key TEXT NOT NULL,row_json TEXT NOT NULL,ordinal INTEGER)')
    counts=dict(public_source_rows=0,center_source_rows=0,protected_rows_excluded=0,damaged_rows_excluded=0,training_reports=0,error_reports=0,normal_reports=0)
    seen_public=set();ct_keys=set();next_progress=time.monotonic()
    try:
        for name in ('public','center'):
            spec=c['sources'][name]
            for original in stream_array(spec['path'],spec['sha256'],guard):
                counts[name+'_source_rows']+=1;cid=original['case_id']
                if name=='public':
                    need(cid in allowed and cid not in seen_public,'PUBLIC_EXACT_TRAIN_SPLIT_MEMBERSHIP');seen_public.add(cid)
                else:need(original['source_dataset']=='center' and c['center_training_source_admitted'] is True,'CENTER_TRAIN_SOURCE')
                ct=json.dumps([original['source_dataset'],str(original['source_case_id'])],separators=(',',':'))
                direct_protected=(cid in protected_ids or os.path.normpath(original['image_path']) in protected_paths or
                  (original['source_dataset'],str(original['source_case_id'])) in protected_source_keys or
                  any(type(original.get(k)) is str and original[k] in protected_opaque for k in ('ct_key','canonical_ct_id','ct_group_key')))
                if direct_protected:
                    if cid in protected_ids:matched_protected_ids.add(cid)
                    protected_ct.add(ct);counts['protected_rows_excluded']+=1;continue
                if os.path.realpath(original['image_path']) in damaged:counts['damaged_rows_excluded']+=1;continue
                row=adapt_existing_training_row(original);err=int(bool(row['primary_type']))
                # One fixed whole-pool order, shared by all seeds/arms. No old exposure audit.
                order=sha(canonical([c['order_salt'],row['source_dataset'],row['source_case_id'],cid]))
                db.execute('INSERT INTO members VALUES(?,?,?,?,?,NULL)',(cid,err,ct,order,canonical(row).decode()))
                counts['training_reports']+=1;counts['error_reports' if err else 'normal_reports']+=1;ct_keys.add(ct)
                if counts['training_reports']%1000==0:db.commit()
                if time.monotonic()>=next_progress:
                    db.commit();guard();next_progress=time.monotonic()+2
                    progress=scope/'materialize_progress_safe.json';tmp=scope/'materialize_progress_safe.pending'
                    write_new(tmp,dict(status='MATERIALIZING_FULL_TRAIN_INDEX',**counts));os.replace(tmp,progress)
        # Remove earlier variants of a protected source CT as well. This one
        # indexed deletion needs no second report-content pass or alias audit.
        for ct in protected_ct:
            by_class=dict(db.execute('SELECT is_error,count(*) FROM members WHERE ct_key=? GROUP BY is_error',(ct,)))
            deleted=sum(by_class.values());counts['protected_rows_excluded']+=deleted;counts['training_reports']-=deleted
            counts['error_reports']-=by_class.get(1,0);counts['normal_reports']-=by_class.get(0,0)
            db.execute('DELETE FROM members WHERE ct_key=?',(ct,))
        ct_keys={x[0] for x in db.execute('SELECT DISTINCT ct_key FROM members')}
        need(seen_public==allowed,'ALL_PUBLIC_TRAIN_MEMBERS_ACCOUNTED')
        need(min(counts['error_reports'],counts['normal_reports'])>0,'NONEMPTY_FULL_TRAIN_STRATA')
        for err in (0,1):
            # Only IDs/order metadata materialized here, not the whole report container.
            ids=[x[0] for x in db.execute('SELECT case_id FROM members WHERE is_error=? ORDER BY order_key,case_id',(err,))]
            db.executemany('UPDATE members SET ordinal=? WHERE case_id=?',((i,cid) for i,cid in enumerate(ids)))
        db.execute('CREATE UNIQUE INDEX ordered_members ON members(is_error,ordinal)');db.commit()
    finally:db.close()
    guard();os.rename(pending,final)
    directory=os.open(scope,os.O_RDONLY|os.O_DIRECTORY);os.fsync(directory);os.close(directory)
    h=hashlib.sha256()
    with final.open('rb') as f:
        while part:=f.read(MIB):guard();h.update(part)
    st=final.stat();stamp={k:getattr(st,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')}
    manifest=dict(protocol='evicheck3d-full-train-index-v1',complete=True,database_path=str(final),database_sha256=h.hexdigest(),database_stat=stamp,
      **counts,training_unique_CT=len(ct_keys),source_commitment=c['sources'],train_members_whole_sha256=c['train_members']['sha256'],
      protected_metadata_whole_sha256=protected_pin,protected_case_ids_matched_in_train=len(matched_protected_ids),protected_case_ids_not_matched_in_train=len(protected_ids-matched_protected_ids),exclusion_policy_whole_sha256=c['exclusion_policy']['sha256'],
      order_salt=c['order_salt'],paired_cycle_length=max(counts['error_reports'],counts['normal_reports']),
      traversal='independent deterministic full-stratum cycles E/N plus error clean twin; persist cursor across chunks',
      all_training_members_consumed_by_this_task=False,tokenizer_calls=0,CT_reads=0,model_calls=0)
    manifest_pin=write_new(scope/'full_train_manifest_private.json',manifest)
    return dict(status='COMPLETE_FULL_TRAIN_INDEX_NOT_MODEL_TRAINING',**counts,training_unique_CT=len(ct_keys),paired_cycle_length=manifest['paired_cycle_length'],
      manifest_whole_sha256=manifest_pin,database_whole_sha256=h.hexdigest(),protected_metadata_whole_sha256=protected_pin,protected_CT_groups_excluded=len(protected_ct),protected_case_ids_matched_in_train=len(matched_protected_ids),protected_case_ids_not_matched_in_train=len(protected_ids-matched_protected_ids),all_training_members_consumed_by_this_task=False,
      report_rows_read_for_actual_materialization=True,CT_reads=0,model_calls=0,automatic_retry=False)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--sha256',required=True);a=ap.parse_args()
    c=read_small(a.config,a.sha256,1*MIB);scope=Path(c['scope'])
    need(bool(os.environ.get('EVICHECK3D_LEGACY_POLICY_ROOT')),'LEGACY_POLICY_ROOT_NOT_CONFIGURED')
    policy_root=Path(os.environ['EVICHECK3D_LEGACY_POLICY_ROOT'])
    need(policy_root.is_absolute(),'LEGACY_POLICY_ROOT_NOT_ABSOLUTE')
    need(c['execution_allowed'] is True and c['decision']=='ROOT_EXECUTION_APPROVED_FULL_TRAIN_MATERIALIZATION' and scope.parent==policy_root and scope.name==c['scope_id'] and scope.name.startswith('fulltrain-index-') and scope.resolve(strict=True)==scope,'NEW_CONCRETE_FULL_TRAIN_SCOPE')
    need(c['cpu_set']==[40,41] and c['growth_bytes']==2*GIB and c['rss_bytes']==2*GIB and c['wall_seconds']==1800,'MATERIALIZATION_FINITE_BUDGET')
    manifest=read_small(Path(__file__).parent/'SOURCE_SHA256.json',c['source_manifest_sha256'],1*MIB)
    for name,pin in manifest['files'].items():
        need(Path(name).name==name and sha(Path(__file__).with_name(name).read_bytes())==pin,'FULL_TRAIN_SOURCE_PIN')
    os.umask(0o077);os.sched_setaffinity(0,{40,41});os.nice(max(0,15-os.getpriority(os.PRIO_PROCESS,0)))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_AS,(2*GIB,2*GIB));resource.setrlimit(resource.RLIMIT_FSIZE,(2*GIB,2*GIB))
    deadline=min(time.monotonic()+1800,time.monotonic()+c['deadline_unix']-time.time());last_check=0
    def stop(*_):raise RuntimeError('FULL_TRAIN_MATERIALIZATION_DEADLINE')
    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.monotonic()))
    def guard():
        nonlocal last_check
        need(time.monotonic()<deadline and not os.path.lexists(scope/'STOP_REQUEST'),'MATERIALIZATION_STOP')
        if time.monotonic()-last_check<.5:return
        last_check=time.monotonic();need(os.sched_getaffinity(0)=={40,41} and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=2*GIB,'MATERIALIZATION_CPU_RSS')
        actual=sum(p.stat().st_size for p in scope.rglob('*') if p.is_file());need(actual<=2*GIB,'INDEX_SCOPE_GROWTH_CAP')
        ledgerraw=(policy_root/'global_growth_ledger_private.json').read_bytes();need(len(ledgerraw)<=4*MIB,'LEDGER_BOUND');ledger=json.loads(ledgerraw)
        row=ledger['scopes'][c['scope_id']];need(row['state']=='ACTIVE_REGISTERED' and row['reserved_bytes']==2*GIB and row['owner_config_sha256']==a.sha256 and row['owner_source_manifest_sha256']==c['source_manifest_sha256'],'NEW_INDEX_SHARED_REGISTRATION')
        other=sum(r['reserved_bytes']-r['consumed_bytes'] for k,r in ledger['scopes'].items() if k!=c['scope_id'] and r['state']=='ACTIVE_REGISTERED')
        d=os.statvfs('/data');r=os.statvfs('/');need(d.f_bavail*d.f_frsize>=max(300*GIB,300*GIB+(2*GIB-actual)+other+4*GIB) and r.f_bavail*r.f_frsize>=64*GIB,'INDEX_SHARED_DISK_FLOORS')
    guard();write_new(scope/'RUN_ONCE.json',dict(config_sha256=a.sha256,automatic_retry=False))
    if 'legacy_trainer_root' in c:
        legacy=Path(c['legacy_trainer_root']);legacy_pin=c['legacy_trainer_manifest_sha256']
    else:
        ref=c['legacy_config_ref'];prior=read_small(ref['path'],ref['sha256'],4*MIB)
        legacy=Path(prior['legacy_trainer_root']);legacy_pin=prior['legacy_trainer_manifest_sha256']
    legacy_manifest=read_small(legacy/'SOURCE_SHA256.json',legacy_pin,1*MIB)
    for name,pin in legacy_manifest['files'].items():need(Path(name).name==name and sha((legacy/name).read_bytes())==pin,'EXACT_LEGACY_CODEC_SOURCE')
    sys.path.insert(0,str(Path(__file__).resolve().parent));sys.path.append(str(legacy))
    try:result=materialize(c,guard)
    except BaseException as exc:
        reason=str(exc);result=dict(status='FAILED_FULL_TRAIN_MATERIALIZATION_NA',failure_enum=reason if reason.isupper() and len(reason)<96 else 'MATERIALIZATION_FAILURE',automatic_retry=False,model_calls=0,CT_reads=0)
    finally:signal.setitimer(signal.ITIMER_REAL,0)
    pin=write_new(scope/'terminal_safe.json',result);print(json.dumps(dict(result,terminal_whole_sha256=pin,shared_release_requires_root_reap=True)))
    return 0 if result['status'].startswith('COMPLETE_') else 2


if __name__=='__main__':raise SystemExit(main())
