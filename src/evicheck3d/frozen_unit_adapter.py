"""Actual TRAIN/dev field adapter draft. Private text objects stay in memory.
No files, model, tokenizer, remote access, semantic atomization or scoring.
"""
from copy import deepcopy
import hashlib
import json
VERSION='unit-pair-grounded-v0.5-actual-adapter-v1'
DELIVERY_SHA='7a9aec3c04021083ac8e2a81720a28e87f59af1eb6b364cf116805f740a7c2f5'
SOURCES={'TRAIN_current_private':'9f667deb96d07780fc91e5f72cd8fb7a924c611ea009a985748db846869511c6','TRAIN_allowed_split_metadata':'a8b897fa1cbc02c41a7c3ec94fd65570c1822b10cd2f18ecb52a30bd250eeec3','MIXED_manifest_bytes_only':'a9638f01c3537f47392255266fd4bb464bb53f675799e195d608c9e47362ba25','DEV125_reports':'323cd8c82bce8a411ceff3b578f78d744579b7c233a379d9fdacf2c97f6283ee','DEV125_units':'edd84b0474e4c43d59215c20f85ce803140f234b47f9899106f9a193bd5ee1e3'}
class Refusal(ValueError):pass
def need(ok,code):
    if not ok:raise Refusal(code)
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def text_sha(v):return hashlib.sha256(v.encode()).hexdigest()
def hash_ok(v):return type(v) is str and len(v)==64 and all(x in '0123456789abcdef' for x in v)
def proof(scope,source_hashes,record_binding_sha256):
    required=set(SOURCES)-{'DEV125_reports','DEV125_units'} if scope=='TRAIN' else {'DEV125_reports','DEV125_units'}
    need(type(source_hashes) is dict and set(source_hashes)==required and all(source_hashes[k]==SOURCES[k] for k in required),'WHOLE_SOURCE_ROLE_HASH_MISMATCH')
    need(hash_ok(record_binding_sha256),'RECORD_BINDING_HASH_REQUIRED')
    return dict(contract_delivery_sha256=DELIVERY_SHA,source_role_hashes=dict(source_hashes),record_binding_sha256=record_binding_sha256,scope=scope)
def labels(value,json_string=False):
    if json_string:
        need(type(value) is str,'LABEL_STORAGE_TYPE_MISMATCH')
        try:value=json.loads(value)
        except (ValueError,TypeError):raise Refusal('LABEL_JSON_INVALID') from None
    need(type(value) is list and value and all(type(x) is int and 0<=x<=7 for x in value) and len(value)==len(set(value)) and not (0 in value and len(value)>1),'LABEL_LIST_INVALID')
    return list(value)
def decode(value,storage):
    """No strip/coercion/literal_eval; missing/empty raw and explicit [] differ."""
    if value is None:return None,'MISSING'
    if storage=='json_list':
        if type(value) is not str:return None,'WRONG_STORAGE_TYPE'
        if value=='':return None,'EMPTY_RAW_FIELD'
        try:items=json.loads(value)
        except (ValueError,TypeError):return None,'JSON_INVALID'
    else:
        if type(value) is not list:return None,'WRONG_STORAGE_TYPE'
        items=value
    if type(items) is not list or any(type(x) is not str for x in items):return None,'NON_STRING_LIST'
    return list(items),'EXPLICIT_EMPTY_LIST' if not items else 'NONEMPTY_LIST'
def occurrences(report,unit):
    if not unit:return []
    spans=[];start=0
    while True:
        pos=report.find(unit,start)
        if pos<0:return spans
        spans.append([pos,pos+len(unit)]);start=pos+1

def assemble(inp,ref,ls,raw,source,error,decode_states,typed,typed_status,provenance):
    need(type(inp) is str and type(ref) is str,'REPORT_FIELDS_NOT_TEXT')
    graph=[];flat_status='PAIRED_POSITIONAL_LISTS'
    if source is None or error is None:flat_status='PENDING_UNIT_STORAGE'
    elif len(source)!=len(error):flat_status='PENDING_CARDINALITY_MISMATCH'
    elif any(ls) and not source:flat_status='PENDING_ERROR_LABEL_WITHOUT_PAIRS'
    elif ls==[0] and source:flat_status='PENDING_NORMAL_LABEL_WITH_PAIRS'
    type_ids=[t for t,_,_ in typed] if typed_status=='EXACT_TYPED_CONCATENATION' else None
    # Never truncate unequal arrays: no guessed zip is supplied as an asserted graph.
    if source is not None and error is not None and len(source)==len(error):
        repeated={}
        for i,(target,corrupted) in enumerate(zip(source,error)):
            key=digest([target,corrupted]);repeated.setdefault(key,[]).append(i)
        for i,(target,corrupted) in enumerate(zip(source,error)):
            ins=occurrences(inp,corrupted);refs=occurrences(ref,target);flags=[]
            if not target.strip() or not corrupted.strip():flags.append('EMPTY_UNIT_REQUIRES_DISPOSITION')
            if target==corrupted:flags.append('IDENTITY_PAIR_NO_ASSUMED_ERROR')
            if target and not refs:flags.append('TARGET_NOT_VERBATIM_REFERENCE')
            if corrupted and not ins:flags.append('ERROR_NOT_VERBATIM_INPUT')
            if len(ins)>1:flags.append('REPEATED_INPUT_OCCURRENCE')
            if len(refs)>1:flags.append('REPEATED_TARGET_OCCURRENCE')
            if len(repeated[digest([target,corrupted])])>1:flags.append('DUPLICATE_ANNOTATION_PAIR')
            graph.append(dict(group_ref='U'+str(i+1),flat_index=i,annotation_type_id=type_ids[i] if type_ids is not None else None,target_unit=dict(role='constructed_correct_text_target',text=target,sha256=text_sha(target),verbatim_occurrences=refs),error_unit=dict(role='corrupted_input_fragment',text=corrupted,sha256=text_sha(corrupted),verbatim_occurrences=ins),flags=flags,selected_input_occurrence=None,selected_target_occurrence=None,semantic_disposition='PENDING_JUDGE_ADJUDICATION'))
    value=dict(adapter_version=VERSION,provenance=provenance,input_report=inp,reference_report=ref,input_report_sha256=text_sha(inp),reference_report_sha256=text_sha(ref),annotation_labels=ls,original_unit_fields=deepcopy(raw),decoded_source_units=source,decoded_error_units=error,decode_states=decode_states,flat_status=flat_status,typed_status=typed_status,typed_pairs=[dict(annotation_type_id=t,target_text=s,error_text=e) for t,s,e in typed],unit_graph=graph,annotation_pair_count=len(graph) if flat_status=='PAIRED_POSITIONAL_LISTS' else None,E_A=None,CT_evidence='NA')
    value['candidate_blind_source_binding_sha256']=digest(value)
    return value

def normalize_train_manifest(manifest_row,current_private_row,*,source_hashes,record_binding_sha256,effective_train_membership_verified):
    """Call only after authorized TRAIN+effective exclusion membership check on host.
    Actual CSV fields: provided_* flat, numeric1..7 typed (already remapped).
    Ordinal fields are preserved, NEVER interpreted as type IDs.
    """
    need(effective_train_membership_verified is True,'EFFECTIVE_TRAIN_MEMBERSHIP_UNVERIFIED')
    row=manifest_row;own=current_private_row
    need(row.get('ecfst_split')=='train','TRAIN_SCOPE_REFUSED')
    ls=labels(row.get('provided_labels'),True)
    need(own.get('report')==row.get('provided_corrupted_report') and own.get('reference')==row.get('fusion narrative report') and sorted(labels(own.get('labels')))==sorted(ls),'CURRENT_REPORT_REFERENCE_LABEL_BINDING_MISMATCH')
    prov=proof('TRAIN',source_hashes,record_binding_sha256)
    keys=['provided_source_unit','provided_error_unit','unit_mapping_version']+[str(t)+'_'+side+'_unit' for t in range(1,8) for side in ('source','error')]+['ordinal_'+str(t)+'_'+side+'_unit' for t in range(1,8) for side in ('source','error')]
    raw={k:deepcopy(row[k]) for k in keys if k in row}
    source,ss=decode(row.get('provided_source_unit'),'json_list');error,es=decode(row.get('provided_error_unit'),'json_list')
    typed=[];typed_ok=row.get('unit_mapping_version')=='ordinal_slots_to_error_types_v1';states={'provided_source_unit':ss,'provided_error_unit':es}
    for t in range(1,8):
        s,sstate=decode(row.get(str(t)+'_source_unit'),'json_list');e,estate=decode(row.get(str(t)+'_error_unit'),'json_list');states[str(t)+'_source_unit']=sstate;states[str(t)+'_error_unit']=estate
        if t not in ls and sstate==estate=='EMPTY_RAW_FIELD':s=e=[]
        if s is None or e is None or len(s)!=len(e) or (t not in ls and (s or e)) or (t in ls and (not s or not e)):typed_ok=False;continue
        typed.extend((t,a,b) for a,b in zip(s,e))
    typed_ok=typed_ok and [s for _,s,_ in typed]==source and [e for _,_,e in typed]==error
    return assemble(own['report'],own['reference'],ls,raw,source,error,states,typed,'EXACT_TYPED_CONCATENATION' if typed_ok else 'PENDING_TYPED_MAPPING',prov)

def normalize_dev(report_row,annotation_row,*,source_hashes,record_binding_sha256):
    """DEV125 actual separate annotation keys are source_units/error_units native lists.
    Type-specific DEV pairing is NOT established by the accepted flat audit.
    """
    ls=labels(report_row.get('labels'));prov=proof('DEV125',source_hashes,record_binding_sha256)
    raw={k:deepcopy(annotation_row[k]) for k in ('source_units','error_units') if k in annotation_row}
    source,ss=decode(annotation_row.get('source_units'),'native_list');error,es=decode(annotation_row.get('error_units'),'native_list')
    return assemble(report_row.get('report'),report_row.get('reference'),ls,raw,source,error,{'source_units':ss,'error_units':es},[],'NOT_SUPPLIED_DEV_FLAT_ONLY',prov)

def build_annotated_baseline_payload(normalized,baseline_payload_builder):
    """Private judge input; returned object contains exact unit text. Never safe-export.
    Invoke the source/reference-only legacy builder on these exact report bytes.
    A later v0.5 judge uses the new unit contract, not unchanged v0.4 baseline task.
    """
    n=deepcopy(normalized);binding=n.pop('candidate_blind_source_binding_sha256',None)
    need(binding==digest(n),'NORMALIZED_ANNOTATION_MUTATED')
    meta=dict(annotation_known_error=any(n['annotation_labels']),annotation_labels=n['annotation_labels'],annotation_types=sorted(x for x in n['annotation_labels'] if x),annotation_audit_level='report_label_only')
    full_report_baseline_payload=baseline_payload_builder(n['input_report'],n['reference_report'],meta)
    need(type(full_report_baseline_payload) is dict and set(full_report_baseline_payload)=={'input_segments','reference_segments','input_risk_tokens','reference_risk_tokens','annotation_metadata','taxonomy','taxonomy_hash','heading_allowlist'},'BASELINE_PAYLOAD_FIELDS_REFUSED')
    if n['flat_status']!='PAIRED_POSITIONAL_LISTS':return dict(status='PENDING_ANNOTATED_STORAGE_OR_PAIRING',reason_code=n['flat_status'],private_annotation_evidence=deepcopy(normalized),E_A=None)
    payload=dict(version=VERSION,source_binding_sha256=binding,full_report_baseline=deepcopy(full_report_baseline_payload),exact_input_report=n['input_report'],exact_reference_report=n['reference_report'],annotated_units=deepcopy(n),unit_review_contract=dict(required_group_refs=[g['group_ref'] for g in n['unit_graph']],require_all_group_dispositions=True,pair_is_not_atom=True,primary='R_A/E_A',full_report_H_N_independent=True,CT_evidence='NA'))
    payload['private_baseline_payload_sha256']=digest(payload)
    return payload
