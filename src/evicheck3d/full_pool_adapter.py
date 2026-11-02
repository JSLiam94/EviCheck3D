"""Dynamic full-training row adapter; no import-time I/O or neural calls."""
import copy,hashlib,json,os,sqlite3,stat
from pathlib import Path
from pref_inputs import need,serialize,apply,semantic_pure_deletion,private_bytes,strict_json
from main96_dev_adapter import preference_bindings


def adapt_existing_training_row(row):
    """Uses the owning full-train report/reference/target schema, not MAIN96 quotas."""
    need(type(row) is dict and all(k in row for k in ('case_id','source_dataset','source_case_id','report','reference','image_path','error','labels')),'FULL_TRAIN_ROW_SCHEMA')
    need(type(row['error']) is bool and type(row['report']) is str and row['report'] and type(row['reference']) is str and row['reference'],'FULL_TRAIN_TEXT')
    if 'target' in row:
        edits=json.loads(row['target']);need(type(edits) is list,'EXISTING_EDIT_TARGET_SCHEMA')
    else:
        # Deterministic exact owning codec, never sample/filter missing targets.
        from edit_codec import build_edits
        edits=build_edits(row['report'],row['reference'])
    target,ranges,_=serialize(edits)
    need(apply(row['report'],edits)==row['reference'],'FULL_TRAIN_EXACT_RECONSTRUCTION')
    need(row['error']==bool(edits)==(row['report']!=row['reference']),'FULL_TRAIN_ERROR_IDENTITY')
    positive=[int(x) for x in row['labels'] if int(x)>0]
    need(bool(positive)==row['error'] and all(1<=x<=7 for x in positive),'FULL_TRAIN_LABELS')
    result=dict(case_id=row['case_id'],source_dataset=row['source_dataset'],source_case_id=row['source_case_id'],
      report=row['report'],reference=row['reference'],image_memory=row['image_path'],
      canonical_ct_id=os.path.realpath(row['image_path']),internal_split='fit',primary_type=positive[0] if positive else 0,
      target=target,new_value_ranges=ranges,edits=edits,pref_edits=preference_bindings(edits),
      pure_deletion_common_boundary=semantic_pure_deletion(edits),target_origin='EXISTING_FULL_TRAIN_EXACT_TARGET_OR_OWNING_CODEC',
      token_eligibility='PENDING_RUNTIME_NOT_TOKENIZED')
    return result


class FullPoolIndex:
    """Read-only immutable SQLite index; fetch only the current paired reports."""
    def __init__(self,spec,guard):
        self.guard=guard;self.manifest=strict_json(private_bytes(spec['manifest_path'],spec['manifest_sha256'],4<<20))
        m=self.manifest;need(m['protocol']=='evicheck3d-full-train-index-v1' and m['complete'] is True,'FULL_TRAIN_MANIFEST')
        self.n_error=m['error_reports'];self.n_normal=m['normal_reports'];self.n_total=m['training_reports']
        need(type(self.n_error) is int and type(self.n_normal) is int and min(self.n_error,self.n_normal)>0 and self.n_error+self.n_normal==self.n_total,'FULL_TRAIN_STRATA')
        self.cycle_length=max(self.n_error,self.n_normal)
        self.path=Path(m['database_path']);need(self.path.resolve(strict=True)==self.path,'FULL_TRAIN_DATABASE_CANONICAL')
        self.stamp=m['database_stat'];self.check_stable()
        self.db=sqlite3.connect(self.path.as_uri()+'?mode=ro&immutable=1',uri=True)
        self.db.execute('PRAGMA query_only=ON')
        count=self.db.execute('SELECT count(*) FROM members').fetchone()[0];need(count==self.n_total,'FULL_TRAIN_DATABASE_COUNT')

    def check_stable(self):
        s=self.path.stat();need(stat.S_ISREG(s.st_mode) and s.st_uid==os.getuid() and s.st_nlink==1 and stat.S_IMODE(s.st_mode)==0o600,'PRIVATE_TRAIN_DATABASE')
        need(all(getattr(s,k)==v for k,v in self.stamp.items()),'MATERIALIZED_DATABASE_CHANGED')

    def get(self,is_error,ordinal):
        self.guard();self.check_stable()
        rec=self.db.execute('SELECT row_json FROM members WHERE is_error=? AND ordinal=?',(int(is_error),ordinal)).fetchone()
        need(rec is not None,'FULL_TRAIN_ORDER_MEMBER_MISSING');r=json.loads(rec[0])
        need(r['internal_split']=='fit' and bool(r['primary_type'])==is_error,'FULL_TRAIN_FIT_ONLY')
        return r

    def pair(self,persistent_cursor):
        need(type(persistent_cursor) is int and persistent_cursor>=0,'PERSISTENT_POOL_CURSOR')
        return self.get(True,persistent_cursor%self.n_error),self.get(False,persistent_cursor%self.n_normal)

    def by_id(self,case_id):
        self.guard();self.check_stable();rec=self.db.execute('SELECT row_json FROM members WHERE case_id=?',(case_id,)).fetchone()
        need(rec is not None,'FEATURE_TRAIN_MEMBER_MISSING');return json.loads(rec[0])

    def coverage(self,start_cursor,end_cursor):
        need(0<=start_cursor<=end_cursor,'CURSOR_SPAN')
        # Counts describe this prefix from the FIRST full-pool stage; do not
        # infer any historical exposure before this pool was introduced.
        return dict(full_pool_reports=self.n_total,full_pool_error_reports=self.n_error,full_pool_normal_reports=self.n_normal,
          pool_cursor_start=start_cursor,pool_cursor_end=end_cursor,
          error_unique_in_committed_updates_since_pool_start=min(end_cursor,self.n_error),normal_unique_in_committed_updates_since_pool_start=min(end_cursor,self.n_normal),
          full_pool_first_committed_traversal_complete=end_cursor>=self.cycle_length,
          paired_cycle_length=self.cycle_length,normal_repeats_possible=self.n_normal<self.n_error,
          error_repeats_possible=self.n_error<self.n_normal)
