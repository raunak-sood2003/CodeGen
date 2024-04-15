from human_eval.data import write_jsonl, read_problems
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from codeT import CodeT
import argparse
from tqdm import tqdm

checkpoint = 'facebook/incoder-6B'
model = AutoModelForCausalLM.from_pretrained(checkpoint, revision="float16", 
                                             torch_dtype=torch.float16, 
                                             low_cpu_mem_usage=True, device_map = "auto")
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
CUDA = True

def extract_function(string):
    defs = [m.start() for m in re.finditer('def ', string)]
    if len(defs) > 1:
        return string[:defs[1]].strip()
    return string.strip()

def extract_testcase(text):
    # Regular expression pattern to match 'assert x == y' or 'assert(x == y)'
    
    pattern1 = r'assert\s+\(?[^\n]*\s*==\s*[^\n]*\)?'
    pattern2 = r'assert\s*\(.*?=\s*.*?\)'
    
    # Find all matches in the text using the pattern
    matches1 = re.findall(pattern1, text)
    matches2 = re.findall(pattern2, text)

    if len(matches1) > 0:
        return matches1[0]
    elif len(matches2) > 0:
        return matches2[0]
    else:
        return ""

def generate_programs(input, max_to_generate=128, temperature=0.2, num_sequences=1):
    BOS = "<|endoftext|>"
    EOM = "<|endofmask|>"
    
    input_ids = tokenizer(input, return_tensors="pt").input_ids
    if CUDA:
        input_ids = input_ids.to("cuda:0")
    max_length = max_to_generate + input_ids.flatten().size(0)
    if max_length > 2048:
        print("warning: max_length {} is greater than the context window {}".format(max_length, 2048))
    with torch.no_grad():
        output = model.generate(input_ids=input_ids, do_sample=True, top_p=0.95, 
                                temperature=temperature, max_length=max_length, 
                                num_return_sequences=num_sequences)
    
    generated_programs = []
    for i in range(num_sequences):
        output_seq = output[i]
        detok_hypo_str = tokenizer.decode(output_seq.flatten(), clean_up_tokenization_spaces=False)
        if detok_hypo_str.startswith(BOS):
            detok_hypo_str = detok_hypo_str[len(BOS):]
        
        generated_programs.append(extract_function(detok_hypo_str))
    
    return generated_programs

def generate_testcases(input, max_to_generate=128, temperature=0.2, num_sequences=1):
    BOS = "<|endoftext|>"
    EOM = "<|endofmask|>"
    
    test_input = input + "\n    " + "pass\n\n" + "assert("
    input_ids = tokenizer(test_input, return_tensors="pt").input_ids
    
    if CUDA:
        input_ids = input_ids.to("cuda:1")
    max_length = max_to_generate + input_ids.flatten().size(0)
    if max_length > 2048:
        print("warning: max_length {} is greater than the context window {}".format(max_length, 2048))
    with torch.no_grad():
        output = model.generate(input_ids=input_ids, do_sample=True, top_p=0.95, 
                                temperature=temperature, max_length=max_length, 
                                num_return_sequences=num_sequences)
    
    generated_tests = []
    for i in range(num_sequences):
        output_seq = output[i]
        detok_hypo_str = tokenizer.decode(output_seq.flatten(), clean_up_tokenization_spaces=False)
        if detok_hypo_str.startswith(BOS):
            detok_hypo_str = detok_hypo_str[len(BOS):]
        
        generated_tests.append(extract_testcase(detok_hypo_str))
    
    return generated_tests

def incoder_generate(input, k=1):
    programs = generate_programs(input, max_to_generate=128, temperature=0.2, num_sequences=k)
    return programs

def incoder_generate_codet(input, k=1):
    n = 2 * k
    programs = generate_programs(input, max_to_generate=128, temperature=0.2, num_sequences=n)
    tests = generate_testcases(input, max_to_generate=128, temperature=0.2, num_sequences=n)

    try:
        codet = CodeT(programs, tests, n, k)
        return codet.programs
    except:
        return programs


if __name__ == '__main__':
    problems = read_problems()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--codet', type=int)
    args = parser.parse_args()

    k=10
    num_problems = 164
    ctr = 0

    if int(args.codet) == 1:
        samples = []

        for task_id in problems:
            if ctr == num_problems:
                break
            
            programs = incoder_generate_codet(problems[task_id]["prompt"], k)
            for program in programs:
                gen = dict(task_id=task_id, completion=program)
                samples.append(gen)
            
            print(ctr)
            ctr += 1
        
        write_jsonl("incoder_codet_humaneval.jsonl", samples)
    
    else:
        samples = []

        for task_id in tqdm(problems):
            if ctr == num_problems:
                break
            
            programs = incoder_generate(problems[task_id]["prompt"], k)
            for program in programs:
                gen = dict(task_id=task_id, completion=program)
                samples.append(gen)
            
            ctr += 1
        
        write_jsonl("incoder_humaneval.jsonl", samples)