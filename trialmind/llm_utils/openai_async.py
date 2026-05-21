import pdb
import asyncio
from typing import List, Union
import httpx
from openai import AsyncOpenAI
from openai import AsyncAzureOpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv, find_dotenv
import os
import logging
import tenacity
import json
import httpx

OPENAI_MODEL_NAME_GPT4 = "gpt-4-turbo"  # new gpt-4-turbo
OPENAI_MODEL_NAME_GPT35 = "gpt-3.5-turbo"
OPENAI_MODEL_NAME_GPT4o = "gpt-4o"
OPENAI_MODEL_NAME_MAP = {
    "openai-gpt-4": OPENAI_MODEL_NAME_GPT4,
    "openai-gpt-35": OPENAI_MODEL_NAME_GPT35,
    "openai-gpt-4o": OPENAI_MODEL_NAME_GPT4o,
}

async_openai_client = AsyncOpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    http_client=httpx.AsyncClient(verify=False)
)

logger = logging.getLogger('my_module_name').setLevel(logging.INFO)
# Silence other loggers
for log_name, log_obj in logging.Logger.manager.loggerDict.items():
    if log_name != 'my_module_name':
        log_obj.disabled = True
load_dotenv(find_dotenv(usecwd=True))

@tenacity.retry(wait=tenacity.wait_random_exponential(min=5, max=20),
                stop=tenacity.stop_after_attempt(3), 
                #reraise=True
               )
async def api_call_single(client: AsyncOpenAI, model: str, messages: list[dict], temperature: float = 0.0, thinking: bool = False, **kwargs):
    # Call the API
    #print('\napi_call_single', client.base_url)
    if not thinking:
        messages[0]['content'] = '/set nothink '+messages[0]['content']
    
    response = await client.chat.completions.create(
        model=model,
        messages=messages,  # Ensure messages is a list
        temperature=temperature,
        extra_body={"reasoning_effort": "none"},
        **kwargs
    )

    return response

@tenacity.retry(wait=tenacity.wait_random_exponential(min=5, max=20),
                stop=tenacity.stop_after_attempt(3), 
                #reraise=True
               )
async def api_function_call_single(client: AsyncOpenAI, model: str, messages: list[dict], tools: list[dict], temperature: float = 0.0, thinking: bool = False, **kwargs):
    # Call the API
    if not thinking:
        messages[0]['content'] = '/set nothink '+messages[0]['content']
    if ('gpt' in model) or ('next' in model):
        reason_e = "low"
    else:
        reason_e = "none"
    response = await client.chat.completions.parse(
        model=model,
        messages=messages,
        #tools=tools,
        temperature=temperature,
        #extra_body={"structured_outputs": {"json": tools[0]}},
        response_format=tools[0],
        extra_body={"reasoning_effort": reason_e}
        #**kwargs
    )
    print(response.choices[0].message.parsed)
    return response

async def apply_async(client: AsyncOpenAI, model: str, messages_list: list[list[dict]], **kwargs):
    """
    Apply the OpenAI API asynchronously to a list of messages using high-level asyncio APIs.
    """
    tasks = [api_call_single(client, model, messages, **kwargs) for messages in messages_list]
    results = await asyncio.gather(*tasks)
    return results

async def apply_function_call_async(client: AsyncOpenAI, model: str, messages_list: list[list[dict]], tools: list[dict], **kwargs):
    """
    Apply the OpenAI API asynchronously to a list of messages using high-level asyncio APIs.
    """
    tasks = [api_function_call_single(client, model, messages, tools, **kwargs) for messages in messages_list]
    results = await asyncio.gather(*tasks)
    return results

def batch_call_openai(batch_messages, llm, temperature, thinking=False):
    async_openai_client = AsyncOpenAI(
        base_url=os.getenv("BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        http_client=httpx.AsyncClient(verify=False)
    )

    model = llm#OPENAI_MODEL_NAME_MAP.get(llm)
    if model is not None:
        results = _async_execute(
            async_function = apply_async, 
            client = async_openai_client, 
            model=model, 
            messages_list=batch_messages, 
            temperature=temperature, 
            seed=0,
            thinking=thinking
            )
    else:
        raise ValueError(f"Unknown llm: {llm}")

    parsed_results = []
    for result in results:
        try:
            content = result.choices[0].message.content
            parsed_results.append(content)
        except:
            parsed_results.append("")
    return parsed_results

def batch_function_call_openai(batch_messages, llm, tools, temperature,thinking=False):
    async_openai_client = AsyncOpenAI(
        base_url=os.getenv("BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        http_client=httpx.AsyncClient(verify=False)
    )
 
    model = llm#OPENAI_MODEL_NAME_MAP.get(llm)
    if model is not None:
        results = _async_execute(
            async_function = apply_function_call_async, 
            client = async_openai_client, 
            model=model, 
            messages_list=batch_messages, 
            tools=tools, 
            temperature=temperature, 
            seed=0,
            thinking=thinking
            )
        #print('batch_function_call_openai in asynch',results)
    else:
        raise ValueError(f"Unknown llm: {llm}")
    #print('\nTOOLS: ',tools)
    
    parsed_results = []
    for result in results:
        try:
            #content = result.choices[0].message.content
            content = result.choices[0].message.parsed
            parsed_results.append(content)
        except:
            parsed_results.append("")
        logging.info(f'parsed_results in asynch: {content}')
    return parsed_results


def _async_execute(async_function, **kwargs):
    from concurrent.futures import ThreadPoolExecutor
    try:
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(1) as executor:
            results = executor.submit(lambda: asyncio.run(async_function(**kwargs)))
            results = results.result()
    except RuntimeError:
        results = async_function(**kwargs)
        results = asyncio.run(results)
    return results


def prompts_as_chatcompletions_messages(prompts: List[str]):
    """
    chat messages for the OpenAI GPT4 chat completions API
    """
    conversations = []
    for prompt in prompts:
        messages = [{
            "role": "user",
            "content": prompt
        }]
        conversations.append(messages)

    return conversations