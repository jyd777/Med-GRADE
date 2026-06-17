import argparse
import json
import copy
import concurrent.futures
import re
from pathlib import Path
from typing import Dict, List, Optional
from openai import OpenAI

from model_list import model_list
from prompt import DEFAULT_INSTRUCTION, INSTRUCTION_API_SUFFIX, _SUFFIX_MARKER
from tqdm.auto import tqdm
import threading

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_FILE = 'qa.jsonl'
OUTPUT_DIR = 'output'


def _resolve_path(path: str) -> Path:
    """relative to root directory"""
    p = Path(path)
    return p if p.is_absolute() else ROOT_DIR / p

DEFAULT_MAX_WORKERS = 16  # default number of concurrent requests per model
DEFAULT_TEMPERATURE = 0.7

DIMENSION_KEYS = [
    "1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7", "1.8",
    "2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "2.7", "2.8",
    "3.1", "3.2", "3.3", "3.4", "3.5", "3.6", "3.7",
]


def build_system_instruction(
    instruction: str,
    append_suffix: bool = True,
    suffix_extra: Optional[str] = None,
) -> str:
    text = instruction
    if append_suffix and _SUFFIX_MARKER not in instruction:
        text += INSTRUCTION_API_SUFFIX
    if suffix_extra:
        text += suffix_extra
    return text


def _scores_from_dimension_dict(obj: dict) -> Optional[List[int]]:
    """convert {"1.1": 1, "1.2": 0, ...} to 23-dimensional list in dimension order"""
    if not isinstance(obj, dict):
        return None
    scores = []
    for key in DIMENSION_KEYS:
        if key not in obj:
            return None
        try:
            val = int(obj[key])
        except (TypeError, ValueError):
            return None
        if val not in (0, 1):
            return None
        scores.append(val)
    return scores


def _strip_markdown_code_fence(text: str) -> str:
    cleaned = text.strip()
    m = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return cleaned


def _strip_thinking_from_response(text: str) -> str:
    """remove thinking blocks that may still be returned by the model, for JSON parsing"""
    if not text:
        return text
    cleaned = _strip_markdown_code_fence(text)
    if re.search(r"##\s*Final\s+Answer\b", cleaned, re.IGNORECASE):
        cleaned = re.split(
            r"##\s*Final\s+Answer\b",
            cleaned,
            maxsplit=0,
            flags=re.IGNORECASE,
        )[-1].strip()
    think_patterns = [
        r"##\s*Thinking\b.*", 
        r"<think[^>]*>.*?</think[^>]*>",
        r"<think>.*?</think>",
        r"<think>.*",
        r"【思考】.*?【/思考】",
    ]
    for pattern in think_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def _parse_model_output(response: str):
    """parse model output to 23-dimensional score list; return (None, raw_text) on failure"""
    cleaned = _strip_thinking_from_response(response)
    if not cleaned:
        return None, response
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list) and len(parsed) == 23:
            return [int(x) for x in parsed], cleaned
        if isinstance(parsed, dict):
            scores = _scores_from_dimension_dict(parsed)
            if scores is not None:
                return scores, cleaned
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(cleaned[start : end + 1])
            if isinstance(parsed, dict):
                scores = _scores_from_dimension_dict(parsed)
                if scores is not None:
                    return scores, cleaned
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start != -1 and end > start:
        try:
            parsed = json.loads(cleaned[start : end + 1])
            if isinstance(parsed, list) and len(parsed) == 23:
                return [int(x) for x in parsed], cleaned
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    score_tags = re.findall(r"\*\*Score:\s*([01])\*\*", cleaned, flags=re.IGNORECASE)
    if len(score_tags) == 23:
        return [int(s) for s in score_tags], cleaned
    return None, cleaned


def _build_extra_body(model_info: dict) -> dict:
    """merge model-specific extra_body; can disable thinking mode for some models"""
    extra_body = dict(model_info.get("extra_body") or {})
    if model_info.get("disable_thinking"):
        extra_body.setdefault("enable_thinking", False)
        ctk = dict(extra_body.get("chat_template_kwargs") or {})
        ctk.setdefault("enable_thinking", False)
        extra_body["chat_template_kwargs"] = ctk
        if "thinking" not in extra_body:
            extra_body["thinking"] = {"type": "disabled"}
    return extra_body


def _build_user_content(user_input: str, disable_thinking: bool = False) -> str:
    if disable_thinking and "/no_think" not in user_input:
        return user_input + "\n/no_think"
    return user_input


def _build_messages(
    system_content: str, user_content: str, merge_system_to_user: bool = False
) -> List[dict]:
    """some vLLM models only support alternating user/assistant in chat template, not separate system"""
    if merge_system_to_user:
        return [
            {
                "role": "user",
                "content": f"{system_content}\n\n{user_content}",
            }
        ]
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def _build_user_input(record: dict) -> str:
    """convert qa.jsonl record to user content for model input"""
    meta_lines = []
    if record.get('department_level1'):
        meta_lines.append(f"department: {record['department_level1']}")
    if record.get('department_level2'):
        meta_lines.append(f"secondary department: {record['department_level2']}")
    if record.get('clinical_domain'):
        meta_lines.append(f"clinical domain: {record['clinical_domain']}")
    if record.get('consultation_intent'):
        meta_lines.append(f"consultation intent: {record['consultation_intent']}")
    parts = []
    if meta_lines:
        parts.append('\n'.join(meta_lines))
    parts.append(record['input'])
    return '\n\n'.join(parts)


# load qa.jsonl; use default instruction if no instruction field
def load_examples(filepath: Path):
    examples = []
    with filepath.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if 'instruction' not in data:
                data['instruction'] = DEFAULT_INSTRUCTION
            examples.append(data)
    return examples

def _is_success(record: dict) -> bool:
    """API success and valid output (no error / parse_error) is required to skip"""
    return (
        record is not None
        and 'error' not in record
        and 'parse_error' not in record
        and isinstance(record.get('output'), list)
    )


def _record_id(record: dict):
    return record.get('id')


def load_checkpoint(output_file: Path) -> dict:
    """load existing output; prioritize successful records for same id, otherwise last one"""
    by_id: dict = {}
    if not output_file.exists():
        return by_id
    with output_file.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = _record_id(rec)
            if rid is None:
                continue
            prev = by_id.get(rid)
            if prev is None:
                by_id[rid] = rec
            elif _is_success(rec):
                by_id[rid] = rec
            elif not _is_success(prev):
                by_id[rid] = rec
    return by_id


def save_checkpoint(output_file: Path, completed: dict) -> None:
    """sort by id and write back, ensure consistent checkpoint file and no duplicate lines"""
    tmp = output_file.with_suffix('.jsonl.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        for rid in sorted(completed.keys(), key=lambda x: (isinstance(x, str), x)):
            f.write(json.dumps(completed[rid], ensure_ascii=False) + '\n')
    tmp.replace(output_file)


def _compute_token_stats(
    completed: dict,
    model: str,
    save_name: str,
    examples: list,
) -> dict:
    ok = [
        completed[ex['id']]
        for ex in examples
        if _is_success(completed.get(ex.get('id')))
    ]
    n_ok = len(ok)
    input_tokens = sum(r.get('input_tokens', 0) for r in ok)
    output_tokens = sum(r.get('output_tokens', 0) for r in ok)
    total_tokens = sum(r.get('total_tokens', 0) for r in ok)
    return {
        'model': model,
        'save_name': save_name,
        'total_examples': len(examples),
        'successful_requests': n_ok,
        'failed_requests': len(examples) - n_ok,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'total_tokens': total_tokens,
        'avg_input_tokens': round(input_tokens / n_ok, 2) if n_ok else 0,
        'avg_output_tokens': round(output_tokens / n_ok, 2) if n_ok else 0,
    }


def _extract_token_usage(completion) -> dict:
    """extract token usage from API response; return 0 if no usage"""
    usage = getattr(completion, "usage", None)
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    prompt = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    total = getattr(usage, "total_tokens", None)
    if isinstance(usage, dict):
        prompt = usage.get("prompt_tokens", prompt)
        completion_tokens = usage.get("completion_tokens", completion_tokens)
        total = usage.get("total_tokens", total)
    input_tokens = int(prompt or 0)
    output_tokens = int(completion_tokens or 0)
    total_tokens = int(total or (input_tokens + output_tokens))
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


# concurrent API calls
def _resolve_top_p(model_info: dict):
    """default 0.95 if not configured; set to None to omit top_p (e.g. Bedrock Claude)"""
    if 'top_p' in model_info:
        return model_info['top_p']
    return 0.95


def call_api(
    example,
    base_url,
    model,
    api_key,
    stream=False,
    temperature=DEFAULT_TEMPERATURE,
    top_p=0.95,
    append_instruction_suffix=True,
    extra_body=None,
    disable_thinking=False,
    merge_system_to_user=False,
    instruction_suffix_extra=None,
    max_tokens=1024,
):
    # deep copy to avoid shared modifications in multi-threading
    ex = copy.deepcopy(example)
    try:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        system_content = build_system_instruction(
            ex['instruction'],
            append_suffix=append_instruction_suffix,
            suffix_extra=instruction_suffix_extra,
        )
        user_content = _build_user_content(
            _build_user_input(ex), disable_thinking=disable_thinking
        )
        request_kwargs = {
            "messages": _build_messages(
                system_content, user_content, merge_system_to_user=merge_system_to_user
            ),
            "model": model,
            "stream": stream,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": 42,
        }
        if top_p is not None:
            request_kwargs["top_p"] = top_p
        if extra_body:
            request_kwargs["extra_body"] = extra_body
        chat_completion = client.chat.completions.create(**request_kwargs)
        raw_response = chat_completion.choices[0].message.content or ""
        ex.update(_extract_token_usage(chat_completion))
        parsed, cleaned = _parse_model_output(raw_response)
        if parsed is not None:
            ex['output'] = parsed
        else:
            ex['output'] = cleaned or raw_response
            ex['parse_error'] = 'JSON parsing failed, original text retained'
        return ex
    except Exception as e:
        return {
            'id': ex.get('id', '?'),
            'error': str(e),
            'instruction': ex.get('instruction', ''),
            'input': ex.get('input', ''),
            'input_tokens': 0,
            'output_tokens': 0,
            'total_tokens': 0,
        }

def run_model(model_info, examples, max_workers=DEFAULT_MAX_WORKERS):
    """run high-concurrency evaluation for a single model (supports checkpointing)"""
    base_url = model_info['base_url']
    api_key = model_info['api_key']
    model = model_info['model']
    save_name = model_info['save_name']
    temperature = model_info.get('temperature', DEFAULT_TEMPERATURE)
    top_p = _resolve_top_p(model_info)
    append_instruction_suffix = model_info.get('append_instruction_suffix', True)
    extra_body = _build_extra_body(model_info)
    disable_thinking = model_info.get('disable_thinking', False)
    merge_system_to_user = model_info.get('merge_system_to_user', False)
    instruction_suffix_extra = model_info.get('instruction_suffix_extra')
    max_tokens = model_info.get('max_tokens', 1024)

    output_dir = _resolve_path(OUTPUT_DIR)
    output_file = output_dir / f'{save_name}.jsonl'
    token_stats_file = output_dir / f'{save_name}_token_stats.json'
    output_dir.mkdir(parents=True, exist_ok=True)

    completed = load_checkpoint(output_file)
    n_skip = 0
    n_retry = 0
    pending = []
    for ex in examples:
        rid = ex.get('id')
        prev = completed.get(rid)
        if _is_success(prev):
            n_skip += 1
            continue
        if prev is not None and not _is_success(prev):
            n_retry += 1
        pending.append(ex)

    if completed:
        save_checkpoint(output_file, completed)

    print(
        f'[{save_name}] checkpointing: {n_skip} completed, '
        f'{len(pending)} pending'
        + (f' (with {n_retry} failed retries)' if n_retry else '')
        + f', temperature={temperature}'
        + (f', top_p={top_p}' if top_p is not None else ', top_p=(omit)')
        + (', disable_thinking=True' if disable_thinking else '')
        + (', merge_system_to_user=True' if merge_system_to_user else '')
        + f', max_tokens={max_tokens}'
    )

    lock = threading.Lock()
    pbar = tqdm(total=len(examples), initial=n_skip, desc=f'Evaluating {save_name}')

    def task(ex):
        result = call_api(
            ex,
            base_url,
            model,
            api_key,
            temperature=temperature,
            top_p=top_p,
            append_instruction_suffix=append_instruction_suffix,
            extra_body=extra_body,
            disable_thinking=disable_thinking,
            merge_system_to_user=merge_system_to_user,
            instruction_suffix_extra=instruction_suffix_extra,
            max_tokens=max_tokens,
        )
        rid = result.get('id', ex.get('id'))
        with lock:
            completed[rid] = result
            save_checkpoint(output_file, completed)
            ex_id = result.get('id', '?')
            if 'error' in result:
                print(f'[{save_name}] id={ex_id} ✗ error: {result["error"]}')
            elif 'parse_error' in result:
                print(
                    f'[{save_name}] id={ex_id} ✗ parse error '
                    f'in={result.get("input_tokens", 0)} out={result.get("output_tokens", 0)} '
                    f'{str(result.get("output", ""))[:80]}'
                )
            else:
                print(
                    f'[{save_name}] id={ex_id} ✓ '
                    f'in={result.get("input_tokens", 0)} out={result.get("output_tokens", 0)} '
                    f'output: {json.dumps(result.get("output", ""), ensure_ascii=False)[:80]}'
                )
            pbar.update(1)
        return result

    if pending:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(task, ex) for ex in pending]
            concurrent.futures.wait(futures)

    pbar.close()

    stats = _compute_token_stats(completed, model, save_name, examples)
    n_ok = stats['successful_requests']
    with open(token_stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f'[{save_name}] all done, {len(completed)} records, results saved to {output_file}')
    print(
        f'[{save_name}] token stats: '
        f'input={stats["input_tokens"]}, output={stats["output_tokens"]}, '
        f'total={stats["total_tokens"]} '
        f'({n_ok} successful, avg in={stats["avg_input_tokens"]}, out={stats["avg_output_tokens"]})'
    )
    print(f'[{save_name}] token stats saved to {token_stats_file}')
    results = [completed[rid] for rid in sorted(completed.keys(), key=lambda x: (isinstance(x, str), x))]
    return save_name, results


def resolve_models(selected_names: Optional[List[str]]) -> List[Dict]:
    """return all models in model_list if no models specified; otherwise filter by save_name"""
    if not selected_names:
        return list(model_list)
    by_save_name = {m['save_name']: m for m in model_list}
    resolved = []
    unknown = []
    for name in selected_names:
        if name in by_save_name:
            resolved.append(by_save_name[name])
        else:
            unknown.append(name)
    if unknown:
        available = ', '.join(by_save_name.keys())
        raise SystemExit(
            f'unknown model: {", ".join(unknown)}\n'
            f'available save_name: {available}'
        )
    return resolved


def parse_args():
    parser = argparse.ArgumentParser(
        description='concurrent evaluation of qa.jsonl dataset; if no models specified, run all models in model_list'
    )
    parser.add_argument(
        '--input',
        default=DEFAULT_INPUT_FILE,
        help=f'input jsonl file path (default: {DEFAULT_INPUT_FILE}, relative to project directory)',
    )
    parser.add_argument(
        '-m', '--models',
        nargs='+',
        metavar='SAVE_NAME',
        help='run only specified models (save_name), can write multiple, e.g. -m medreason kimi-k2.6',
    )
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='list all save_name in model_list and exit',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.list_models:
        for m in model_list:
            print(m['save_name'])
        raise SystemExit(0)

    models_to_run = resolve_models(args.models)
    input_file = _resolve_path(args.input)
    if not input_file.exists():
        raise SystemExit(f'input file does not exist: {input_file}')
    examples = load_examples(input_file)
    print(f'loaded {len(examples)} examples: {input_file}')

    if args.models:
        print(f'selected {len(models_to_run)} models: {", ".join(m["save_name"] for m in models_to_run)}')
    else:
        print(f'no models specified, will run all {len(models_to_run)} models')

    # concurrent between models, also high-concurrency within each model
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(models_to_run)) as model_executor:
        model_futures = {
            model_executor.submit(run_model, model_info, examples): model_info['save_name']
            for model_info in models_to_run
        }
        for future in concurrent.futures.as_completed(model_futures):
            name = model_futures[future]
            try:
                future.result()
            except Exception as e:
                print(f'[{name}] error: {e}')

    print('\nall models evaluation completed!')
