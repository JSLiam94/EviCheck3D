"""Bounded lazy full-pool compiler, no 32-template reset or all-pool token RAM."""
import copy
from collections import OrderedDict
from pref_inputs import need,validate_pref_binding
from pref_tokens import compile_row,encode,content_indices,INAPPLICABLE
from center_decision_objective_v2 import derive_shared_action_index


class FullPoolPlan:
    def __init__(self,model,pool,start_cursor,guard,max_templates=8,restored_action_spec=None):
        need(type(start_cursor) is int and start_cursor>=0 and max_templates==8,'FULL_POOL_COMPILER_LIMITS')
        self.model=model;self.pool=pool;self.guard=guard;self.cursor=start_cursor;self.cache=OrderedDict();self.max_templates=max_templates
        self.keep=model.tokenizer('[]',add_special_tokens=False)['input_ids']+[model.tokenizer.eos_token_id]
        self.action_spec=restored_action_spec
        self.compiled_templates=0;self.inapplicable={x:0 for x in INAPPLICABLE}

    def __getitem__(self,key):
        if key=='action_spec':need(self.action_spec is not None,'ACTION_SPEC_NOT_COMPILED');return self.action_spec
        if key=='template_at':return self.template_at
        if key=='next_pool_cursor':return self.cursor
        raise KeyError(key)

    def template_at(self,cursor):
        self.guard();need(cursor==self.cursor,'EXACT_PERSISTENT_CURSOR')
        if cursor in self.cache:self.cache.move_to_end(cursor);return self.cache[cursor]
        error,normal=self.pool.pair(cursor);tok=self.model.tokenizer
        error_ids=tok(error['target'],add_special_tokens=False)['input_ids']+[tok.eos_token_id]
        action=derive_shared_action_index([error_ids],self.keep)
        if self.action_spec is None:self.action_spec=action
        else:need(action==self.action_spec,'FULL_POOL_SHARED_ACTION_INDEX')
        need(tok(normal['target'],add_special_tokens=False)['input_ids']+[tok.eos_token_id]==self.keep,'FULL_POOL_NORMAL_KEEP')
        clean=copy.deepcopy(error);clean.update(report=error['reference'],target='[]',new_value_ranges=[],primary_type=0,edits=[],pref_edits=[],pure_deletion_common_boundary=False)
        items=[]
        for role,row in zip(('E','N','CLEAN'),(error,normal,clean)):
            self.guard();layout=compile_row(row,tok,self.model.prompt_token_count(row['report']));items.append(dict(role=role,row=row,layout=layout))
        pairs=[];statuses=[];positive_layout=items[0]['layout']
        for j,edit in enumerate(error['edits']):
            self.guard();binding=error['pref_edits'][j];validate_pref_binding(error['edits'],j,binding)
            reason=None if binding['disposition']=='POTENTIALLY_ELIGIBLE' else binding['disposition']
            positive=binding['positive_target'];negative=binding['negative_target'];need(positive==error['target'],'FULL_POOL_POSITIVE_TARGET')
            negative_layout=encode(tok,negative);common_length=positive_layout['prompt_length']+max(len(positive_layout['ids']),len(negative_layout['ids']))+1
            need(common_length<=32768,'PREF_CONTEXT_OVERFLOW')
            plus=content_indices(tok,positive,positive_layout,binding['positive_new_content_range']) if reason is None else None
            minus=content_indices(tok,negative,negative_layout,binding['negative_new_content_range']) if reason is None else None
            if reason is None and (plus is None or minus is None or [positive_layout['ids'][i] for i in plus]==[negative_layout['ids'][i] for i in minus]):reason='TOKEN_BOUNDARY_UNSCORABLE'
            if reason:
                self.inapplicable[reason]+=1;statuses.append(dict(edit_index=j,status=reason));continue
            pairs.append(dict(edit_index=j,positive_target=positive,negative_target=negative,positive_ids=positive_layout['ids'],negative_ids=negative_layout['ids'],positive_indices=plus,negative_indices=minus,common_length=common_length,prompt_length=positive_layout['prompt_length']))
            statuses.append(dict(edit_index=j,status='ELIGIBLE'))
        template=dict(items=items,pairs=pairs,edit_statuses=statuses,pool_cursor=cursor)
        self.cache[cursor]=template;self.compiled_templates+=1
        while len(self.cache)>self.max_templates:self.cache.popitem(last=False)
        return template

    def committed(self):
        self.cursor+=1

    def safe(self):
        return dict(compiled_templates=self.compiled_templates,persistent_pool_cursor=self.cursor,template_cache_capacity=self.max_templates,
          full_pool_reports=self.pool.n_total,paired_cycle_length=self.pool.cycle_length,all_training_members_in_committed_updates=self.cursor>=self.pool.cycle_length,
          preference_inapplicable_counts=self.inapplicable,context=32768,targets_truncated=False,records_filtered_for_token_length=False)
