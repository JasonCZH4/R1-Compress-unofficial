import json, os
from openai import OpenAI
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

client = OpenAI(
    api_key='000',
    base_url="http://0.0.0.0/v1",
)

def get_oai_completion(prompt, temperature):
    response = client.chat.completions.create(
        model='Qwen2.5-72B',
        messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            
            ],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        temperature=temperature,
        max_tokens=1024,
    )
    res = response.choices[0].message.content
    return res


def query_llm(ins, temperature=0.75):
    try:
        ans = get_oai_completion(ins, temperature)
        return ans
    except Exception as e:
        print(e)


def get_chunks_candidates(chunk, temperature=0.75, candidate_nums=8):
    chunk_candidates = []
    prompt = """
    Here is an reasoning piece excerpt from some math problem solving process (it is incomplete, but this doesn’t matter.): {step} 
    Instructions:  You need to simplify the wording of given reasoning piece to get a concise reasoning piece. 
    Notice:  1. Avoid omitting any reasoning steps. You should keep all the reflection, analysing, checking steps and even steps making mistakes. (Especially steps contains word “wait”, “hmm”) 
    2. Directly give me the simplified content without any additional words. 
    3. Do not add additional steps or continue the reasoning process. 
    4. Follow the format of given reasoning piece. 
    Output format: <start> (simplified content) <end>
    """
    for m in candidate_nums:
        chunk_candidates.append(query_llm(prompt.format(step=chunk), temperature))
    return chunk_candidates

        
def get_perplexity(tokenizer, model, question, text, candidate, device, max_length=20480):
    try:
        text = question + text + candidate
        text_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        
        start_index = text.rfind(candidate)
        start_token = len(tokenizer.encode(text[:start_index]))

        labels = text_ids.clone()
        labels[0, :start_token] = -100
    
        with torch.no_grad():
            outputs = model(text_ids, labels=labels)

        loss = outputs.loss
        perplexity = torch.exp(loss)

        return perplexity.to('cpu').item()
    
    except:
        return 0


def split_chunk(text, tokenizer, chunk_size=50):
    chunks = text.split('\n\n')
    processed_chunks = []
    buffer = ''
    # 遍历chunks，保证chunks里面的chunk长度都超过chunk_size
    for chunk in chunks:
        if len(tokenizer(chunk)) > chunk_size:
            processed_chunks.append(buffer + chunk)
            buffer = ''
        else:
            buffer += chunk
            if len(tokenizer(buffer)) > chunk_size:
                processed_chunks.append(buffer)
                buffer = ''
    return processed_chunks

def compress_chunk(question, response, model, tokenizer, save_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    processed_chunks = split_chunk(response, tokenizer)
    
    chunks_candidates = []
    for chunk in processed_chunks:
        chunks_candidates.append(get_chunks_candidates(chunk))
    
    pre_candidates = ''
    for chunk_idx, chunk in enumerate(processed_chunks):
        best_condition_probability = 0
        for candidate_idx, candidate in enumerate(chunks_candidates[chunk_idx]):
            condition_probability = get_perplexity(tokenizer, model, question, pre_candidates, candidate, device)
            if condition_probability > best_condition_probability:
                best_condition_probability = condition_probability
                best_candidate = candidate
        pre_candidates += best_candidate
    
    result = {'instruction': question, 'input': '', 'output': ''.join(pre_candidates)}
    with open(save_path, 'a') as f:
        f.write(json.dumps(result) + '\n')
        f.flush()
    
    return result


def main():
    with open('test.json') as f:
        ipts = json.load(f)
    
    save_path = 'test_result.jsonl'
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            opts = [x for x in f]
    else:
        opts = []
    s = set(x['instruction'] for x in opts)
    # 只处理不在输出文件中的数据
    need_list = [(q['instruction'], q['output'], save_path) for q in ipts if q['instruction'] not in s]
    
    with ThreadPoolExecutor() as executor:
        future_to_question = {executor.submit(compress_chunk, q): q for q in need_list}
        for future in tqdm(as_completed(future_to_question)):
            _ = future_to_question[future]
            rst = future.result()
            opts.append(rst)
    
    with open(f'{save_path}.json', 'w', encoding='utf-8') as f:
        json.dump(opts, f, ensure_ascii=False)