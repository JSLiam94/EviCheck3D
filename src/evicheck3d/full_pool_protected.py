"""Targeted frozen evaluation metadata only; never inspects text semantics."""
import os
from pref_inputs import need

CUSTODIAN=os.environ.get('EVICHECK3D_PROTECTED_CUSTODIAN', '')
CUSTODIAN_SHA='16e252d174e9382e9698db383020a3394e82137c0789868d30adb402ff617e1d'
MAIN96=os.environ.get('EVICHECK3D_PROTECTED_PROJECTION', '')
MAIN96_SHA='536c6af8af4aa20ee6e3466de655a5bcb9c22fca5663015542527a4335e8eb86'


def collect(read_small):
    need(os.path.isabs(CUSTODIAN) and os.path.isabs(MAIN96),'PROTECTED_METADATA_PATHS_NOT_CONFIGURED')
    custody=read_small(CUSTODIAN,CUSTODIAN_SHA,4<<20)
    need(custody['protocol']=='r2-group-overlap-private-review-packet-v1' and type(custody['records']) is list and len(custody['records'])==192,'FROZEN_CUSTODY_METADATA')
    main=read_small(MAIN96,MAIN96_SHA,4<<20)
    need(main['protocol']=='unit-program-main96-projection-v1' and type(main['rows']) is list and len(main['rows'])==96,'FROZEN_MAIN96_METADATA')
    ids=set();paths=set();source_keys=set();opaque=set()
    def add(row):
        cid=row['case_id'];need(type(cid) is str and 0<len(cid)<=256,'PROTECTED_CASE_ID');ids.add(cid)
        for k in ('image_memory','image_path','canonical_image_path'):
            p=row.get(k)
            if type(p) is str and os.path.isabs(p):paths.add(os.path.normpath(p))
        dataset=row.get('source_dataset',row.get('dataset'));source_id=row.get('source_case_id')
        if type(dataset) is str and type(source_id) is str:source_keys.add((dataset,source_id))
        for k in ('ct_key','canonical_ct_id','ct_group_key'):
            v=row.get(k)
            if type(v) is str and v:opaque.add(v)
    for row in custody['records']:add(row)
    dev=[r for r in main['rows'] if r['internal_split']=='dev'];need(len(dev)==32,'FROZEN_DEV32_MEMBERS')
    for row in dev:
        add(row)
        origin=row.get('source_provenance',{}).get('current_private_row',{})
        if type(origin) is dict:
            dataset=origin.get('source_dataset');source_id=origin.get('source_case_id')
            if type(dataset) is str and type(source_id) is str:source_keys.add((dataset,source_id))
    return dict(protocol='evicheck3d-targeted-protected-training-cut-v1',case_ids=sorted(ids),image_paths=sorted(paths),source_ct_keys=[list(x) for x in sorted(source_keys)],opaque_ct_keys=sorted(opaque),custodian_sha256=CUSTODIAN_SHA,MAIN96_projection_sha256=MAIN96_SHA,review_packet_records=192,MAIN96_DEV_records=32,clinical_text_interpreted=False,history_exposure_audit=False)
