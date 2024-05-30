from vLLM.vllm_prompting import prompt_vllm
from human_eval.data import write_jsonl, read_problems
from pragmatic_codegen.utils import extract_function, extract_testcase
import json
 
if __name__ == '__main__':
    model_name = "codellama/CodeLlama-7b-hf"
    output_dir = "."
    api_base = "http://localhost:8080/v1" 
    n = 500
    
    params = {
        "temperature" : 0.8,
        "top_p" : 0.95,
        "max_tokens" : 128,
        "n" : 200
    }

    problems = read_problems()
    
    to_gen_tests = False

    if to_gen_tests:
        prompts = []
        for i, task_id in enumerate(problems):                
            prefix = "# Write test cases for the following function.\n"
            suffix = "    pass\n\nassert"
            prompt = prefix + problems[task_id]["prompt"] + suffix
            prompts.append(prompt)
        
        responses = prompt_vllm(model_name, prompts, output_dir, api_base, params)

        res = []
        for i, task_id in enumerate(problems):    
            for j in range(len(responses[i])):
                prompt = problems[task_id]["prompt"]
                gen_code = prompt + responses[i][j]
                gen_test = extract_testcase(gen_code)
                res.append({"task_id" : task_id, "completion" : gen_test})

        write_jsonl("codellama_humaneval_tests_k%d.jsonl" % n, res)
    
    else:
        prompts = []
        for i, task_id in enumerate(problems):          
            prefix = "# Complete the following function.\n"
            prompt = prefix + problems[task_id]["prompt"]
            prompts.append(prompt)
        
        responses = prompt_vllm(model_name, prompts, output_dir, api_base, params)

        res = []
        for i, task_id in enumerate(problems):    
            for j in range(len(responses[i])):
                prompt = problems[task_id]["prompt"]
                gen_code = prompt + responses[i][j]
                gen_function = extract_function(gen_code)
                res.append({"task_id" : task_id, "completion" : gen_function})

        write_jsonl("codellama_humaneval_programs_k%d.jsonl" % n, res)

