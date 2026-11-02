"""NEW tuple update integration, called only by separately admitted owner.
Uses fresh-public main96_model.Main96UnitProgramRepair('bound_ct') for both
arms; feature donor policy differs solely in the compiled private plan.
"""
from __future__ import annotations
import math
from typing import Any
from main96_backward import tuple_backward
from main96_training_plan import update_items


def need(value,code):
    if not value:raise RuntimeError(code)


def _diagnose_gradients(model, torch_module) -> tuple[dict[str, Any], list[Any]]:
    groups = {
        "LoRA": {"trainable_tensors": 0, "grad_none_tensors": 0, "grad_zero_tensors": 0,
                 "grad_nonzero_tensors": 0, "grad_norm_squared": 0.0},
        "DenseBridge": {"trainable_tensors": 0, "grad_none_tensors": 0, "grad_zero_tensors": 0,
                        "grad_nonzero_tensors": 0, "grad_norm_squared": 0.0},
    }
    params = []
    for name, parameter in model.trainable().items():
        group = "LoRA" if name.startswith("llm.") else "DenseBridge" if name.startswith("bridge.") else None
        need(group is not None, "UNEXPECTED_TRAINABLE_PARAMETER")
        record = groups[group]
        record["trainable_tensors"] += 1
        params.append(parameter)
        grad = parameter.grad
        if grad is None:
            record["grad_none_tensors"] += 1
            continue
        need(bool(torch_module.isfinite(grad).all()), "NONFINITE_GROUP_GRADIENT")
        if bool(torch_module.count_nonzero(grad)):
            record["grad_nonzero_tensors"] += 1
        else:
            record["grad_zero_tensors"] += 1
        squared = float(grad.detach().float().square().sum())
        need(math.isfinite(squared), "NONFINITE_GROUP_GRADIENT_NORM")
        record["grad_norm_squared"] += squared
    for record in groups.values():
        record["grad_norm_L2"] = math.sqrt(record.pop("grad_norm_squared"))
        need(math.isfinite(record["grad_norm_L2"]), "NONFINITE_GROUP_GRADIENT_NORM")
    return groups, params



def train_step(torch_module,model,optimizer,private_plan,update,features_by_case_id,decision_spec,guard):
    guard();torch_module.cuda.set_device(model.device);model.train()
    optimizer.zero_grad(set_to_none=True);before=model.export()
    items=update_items(private_plan,update)
    def lookup(case_id):
        need(case_id in features_by_case_id,'PLANNED_FEATURE_MISSING')
        return features_by_case_id[case_id]
    try:
        result=tuple_backward(model,items,action_index=decision_spec['action_index'],
            feature_lookup=lookup,autocast=lambda:torch_module.autocast('cuda',dtype=torch_module.bfloat16),guard=guard)
        stats,params=_diagnose_gradients(model,torch_module)
        for group in ('LoRA','DenseBridge'):
            need(stats[group]['trainable_tensors']>0 and stats[group]['grad_none_tensors']==0 and stats[group]['grad_nonzero_tensors']>0,'TRAINABLE_GROUP_GRADIENT_PARTICIPATION')
        norm=torch_module.nn.utils.clip_grad_norm_(params,1.0)
        need(bool(torch_module.isfinite(norm)),'NONFINITE_GLOBAL_GRADIENT')
        guard();optimizer.step();model.assert_frozen();torch_module.cuda.synchronize(model.device)
        after=model.export();need(set(before)==set(after),'TRAINABLE_KEYS_CHANGED')
        changed={'LoRA':0,'DenseBridge':0};delta={'LoRA':0.0,'DenseBridge':0.0}
        for name,current in after.items():
            group='LoRA' if name.startswith('llm.') else 'DenseBridge' if name.startswith('bridge.') else None
            need(group is not None and bool(torch_module.isfinite(current).all()),'PARAMETER_CONTRACT')
            changed[group]+=int(not torch_module.equal(before[name],current))
            value=float((current.float()-before[name].float()).square().sum())
            need(math.isfinite(value),'NONFINITE_PARAMETER_DELTA');delta[group]+=value
        need(all(changed[group]>0 for group in ('LoRA','DenseBridge')),'SOME_CHANGE_EACH_TRAINABLE_GROUP')
        guard()
        result.update(update=update,cursor=update,tuple_index=(update-1)%32,
                      optimizer_updates_committed=1,preclip_gradient_norm=float(norm),
                      gradient_diagnostics=stats,changed_tensors=changed,
                      parameter_delta_l2={k:math.sqrt(v) for k,v in delta.items()})
        return result
    except BaseException:
        optimizer.zero_grad(set_to_none=True)
        # Owner aborts scope; no hidden retry or data replacement. A failure
        # after step must resume only from a complete bound checkpoint.
        raise


def derive_action_spec(fit,tokenizer):
    from center_decision_objective_v2 import derive_shared_action_index
    need(len(fit)==64 and all(r['internal_split']=='fit' for r in fit),'TRAIN64_ACTION_SCOPE')
    errors=[r for r in fit if r['primary_type']!=0]
    normals=[r for r in fit if r['primary_type']==0]
    need(len(errors)==len(normals)==32,'TRAIN64_ACTION_STRATA')
    eos=tokenizer.eos_token_id;need(type(eos) is int and eos>=0,'EOS_REQUIRED')
    keep=tokenizer('[]',add_special_tokens=False)['input_ids']+[eos]
    need(all(tokenizer(r['target'],add_special_tokens=False)['input_ids']+[eos]==keep for r in normals),'NORMAL_KEEP_ACTION')
    return derive_shared_action_index([tokenizer(r['target'],add_special_tokens=False)['input_ids']+[eos] for r in errors],keep)
