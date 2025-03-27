import os
import asyncio
import pandas as pd
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as atqdm
from dotenv import load_dotenv

load_dotenv()

# Create async client
client_nebius_async = AsyncOpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.getenv('NEBIUS_API_KEY')
)

def get_prompt_messages(problem):
    system_prompt = r"""
    You are Deepseek, an expert mathematical reasoning Large Language Model. 
    
    Your task is to analyze a mathematical problem and determine whether the problem statement states it has an exact integer.
    """
    
    user_prompt = f"""
    Problem:
    
    {problem}
    
    Briefly analyze the problem statement and determine whether it states the problem does not require an exact integer solution. That is, look for indicators like "return you answer in terms of a X" or "give your best estimate".
    After your analysis, provide your answer as either \\boxed{{yes}} or \\boxed{{no}}.
    
    - Return \\boxed{{no}} if the problem indicates it does not require an exact integer solution
    - Return \\boxed{{yes}} otherwise.

    
    Note that solutions that involve rounding to an integer are acceptable and should be marked as \\boxed{{yes}}.
    Think step-by-step and conclude with either \\boxed{{yes}} or \\boxed{{no}}.
    """
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

async def generate_response_nebius_async(client, messages, verbose=False):
    try:
        response = await client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1",
            max_tokens=1024,
            temperature=0.6,
            top_p=0.95,
            messages=messages,
            stream=True
        )
        
        completion = ""
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                completion += chunk.choices[0].delta.content
                if verbose:
                    print(chunk.choices[0].delta.content, end='', flush=True)
        
        return completion
    except Exception as e:
        print(f"Error in generate_response_nebius_async: {e}")
        raise e

async def process_problem(p_id, problem, semaphore):
    async with semaphore:
        while True:
            try:
                print(f'Processing problem {p_id}')
                messages = get_prompt_messages(problem)
                response = await generate_response_nebius_async(client_nebius_async, messages)
                print(response)
                
                # Extract the boxed answer
                has_integer_solution = None
                if r'\boxed{yes}' in response:
                    has_integer_solution = True
                elif r'\boxed{no}' in response:
                    has_integer_solution = False
                else:
                    has_integer_solution = True
                
                print(f'Problem {p_id} processed: {has_integer_solution}')
                return p_id, response, has_integer_solution
            except Exception as e:
                print(f'Error processing problem {p_id}: {e}, retrying...')
                await asyncio.sleep(1)

async def main():
    # Read the DataFrame
    df = pd.read_csv('df_recombined.csv')
    
    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(40)
    
    # Create tasks for each problem
    tasks = []
    for _, row in df.iterrows():
        p_id = row['p_id']
        problem = row['problem']
        task = asyncio.create_task(
            process_problem(p_id, problem, semaphore)
        )
        tasks.append(task)
    
    # Gather all results
    results = await atqdm.gather(*tasks)
    
    # Process results
    p_ids = []
    responses = []
    has_integer_solutions = []
    
    for p_id, response, has_integer_solution in results:
        p_ids.append(p_id)
        responses.append(response)
        has_integer_solutions.append(has_integer_solution)
    
    # Create a new DataFrame with the results
    results_df = pd.DataFrame({
        'p_id': p_ids,
        'response': responses,
        'has_integer_solution': has_integer_solutions
    })

    results_df.to_csv('problem_filter_results.csv', index=False)
    
    # Merge with the original DataFrame
    df_filtered = df.merge(results_df, on='p_id', how='left')
    
    # Save the results
    df_filtered.to_csv('df_filtered_integer_solutions.csv', index=False)
    
    # Also save a filtered version with only the problems that have integer solutions
    if not all(pd.isna(df_filtered['has_integer_solution'])):
        df_integer_only = df_filtered[df_filtered['has_integer_solution'] == True]
        df_integer_only.to_csv('df_integer_solutions_only.csv', index=False)
    
    print(f"Total problems: {len(df)}")
    print(f"Problems processed: {len(results_df)}")
    print(f"Problems with integer solutions: {sum(has_integer_solutions)}")

if __name__ == "__main__":
    asyncio.run(main()) 