import requests
import logging
import pandas as pd
import numpy as np
from langchain_community.document_loaders import PyMuPDFLoader
#from langchain_pymupdf4llm import PyMuPDF4LLMLoader, PyMuPDF4LLMParser
#import pymupdf4llm
import re
import os
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
import httpx
from statistics import mode
from trialmind.llm_utils.openai_async import batch_call_openai,batch_function_call_openai
from pydantic import BaseModel, validator, Field,\
field_validator, conlist 
from typing_extensions import Literal
from typing import Dict,TypeVar
from deepmerge import always_merger
import ast
from aux_prompts import TRANSLATE_PROMPT,\
ENHANCE_RESULTS_PROMPT2,RESULTS_TOP_PROMPT,PROMPT_RES_EXTRACTION
from queries import criteria_text3# as criteria_text
from trialmind.pydantic_classes import ClinicalTrialOutcomes,GroupOutcomes

logger = logging.getLogger('my_module_name').setLevel(logging.INFO)
# Silence other loggers
for log_name, log_obj in logging.Logger.manager.loggerDict.items():
    if log_name != 'my_module_name':
        log_obj.disabled = True
        
T = TypeVar("T", bound=BaseModel)

# to override existing env.variables     
load_dotenv(find_dotenv(usecwd=True), override=True)
openai_client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    http_client=httpx.Client(verify=False)
)


def info_from_doc(file_path, with_ru=False):

    loader = PyMuPDFLoader(file_path, mode='single')
    docs = loader.load()
    
    text = re.search('База клинических испытаний:(.*)', docs[0].page_content
         ).groups()[0].strip()
    
    # ищем след.раздел от нашей таблицы
    table_oc = re.search('(?:Содержание)[^\\n]*(\\n[\D]*)', 
          docs[0].page_content.replace('\x0c',''), 
          flags=re.DOTALL).groups()[0].split('\n')
    table_treats_id = table_oc.index('Полный список препаратов')
    next_title = table_oc[table_treats_id+1]
    # первая часть, тк при переносе строк символы,...
    xx = next_title[:25].replace(' ','[ \\n]')
    
    table_part = re.search('(Ранг.{,10}Препарат.{,10}Активированные.{,10}мишени'+\
                          '.{,10}Подавленные.{,10}мишени.{,10}Drug.{,10}score)'+\
                          f'(.*)(?:{xx})',
                          #'(.*)(Полный)',
                          docs[0].page_content.replace('\x0c',''), 
                          flags=re.DOTALL).group()
    
    treats=re.findall("(\s?\\n\d+([ а-яА-Я]*))"+\
              "((\\n[ a-zA-Zа\d,\n]*((?!\\n\d).)*)| )"+\
              "(\\n(- |(-?\d+\.\d+)))",
             table_part,
             flags=re.DOTALL)
    treats_df = pd.DataFrame(treats, 
                             columns=['name_','treat',
                                      'm_','m2_','_',
                                      'drugscore_','score','d_']
                            )
    df = treats_df[['treat','score']]
    # просто не указано число; меняем
    df.loc[df.score.isin(['- ','-',' -']),'score'] = '10.00'
    
    di = pd.read_csv('docs/ru_en_drugs.csv', usecols=[0,1])
    df.score = df.score.astype(float)
    # removing whitespace at the start and end of the string
    df.treat = df.treat.apply(lambda x: x.strip())
    df_en = df[(df.score>=0.01)].merge(di, how='left', left_on='treat', 
                           right_on='name').drop(columns=['name'])
    treat001 = ', '.join(df_en[df_en.drug.isna()].treat.values)
    
    messages = [{'role':'system', 'content':'You are a helpful assistant /no_think'},
                {'role':'user', 'content':f"Translate into English: {text}"}]
    fin_condition = use_llm(os.getenv("MODEL_NAME"), messages)
    
    messages = [{'role':'system', 'content':'You are a helpful assistant /no_think'},
                {'role':'user', 'content':f"Translate into English: {treat001}. Separate each term with a comma. Answer only with translations, do not include any clarifications"}]
    treatments_eng = use_llm(os.getenv("MODEL_NAME"), messages).split(',')
    treatments_eng = [i.strip().strip('.') for i in treatments_eng]
    df_en.loc[df_en.drug.isna(), 'drug'] = treatments_eng
    
    if with_ru:
        return fin_condition,df_en.drug.values, df_en, text, df_en.treat.values
    else:
        return fin_condition,df_en.drug.values, df_en


def use_llm(model='Qwen/Qwen3-32B', messages=[],openai_client=openai_client,
           tool=[]):
    logging.info(os.getenv("BASE_URL"))#, os.getenv("OPENAI_API_KEY"))
    # when assigned outside, can sometimes give errors
    openai_client = OpenAI(
        base_url=os.getenv("BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        http_client=httpx.Client(verify=False)
    )
    
    if tool:
        response = openai_client.chat.completions.parse(
            model=model,
            messages=messages,
            temperature=0,
            extra_body={"reasoning_effort": "none"},
            response_format = tool[0],
        )
        fin = response.choices[0].message.parsed
        
    else:
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            extra_body={"reasoning_effort": "none"}
        )
        #print(response)
        fin = response.choices[0].message.content
        fin = fin.strip('<think>\n\n</think>\n\n')
    return fin




def results_ct(all_studies):
    chosen = all_studies[all_studies.results==True]
    outcome_participants = [obj for item_main in chosen.outcomes.values for obj in item_main 
                        if(obj.get('unitOfMeasure','') == 'Participants')] 
    messages = []
    for i in outcome_participants:
        messages.append([{'role':'system', 'content':'You are a helpful medical assistant. Extract information about the efficiency statistics from each outcome measure. Give: 1) a concise description of results with percentages and 2) a short description of study objectives /no_think'},
                {'role':'user', 'content':f"{i}"}])
    
    results = batch_call_openai(messages, os.getenv('MODEL_NAME'), int(os.getenv('TEMPERATURE', 0)), thinking=False)
    return results
    


def get_clinicaltrials(query_term, query_intr, max_studies=20):
    base = 'https://clinicaltrials.gov/api/v2/studies'  # new API

    # fields for CSV and JSON: https://clinicaltrials.gov/data-api/about-api/csv-download

    # new API fields - for JSON
    extract_fields = [
        'NCTId',
        'Condition',
        'BriefTitle',
        'HasResults',
        "Intervention",
        #'PrimaryOutcome',
        'OutcomeMeasuresModule',
        #'NumPhases',
        'Phase',
        'OverallStatus',
        'StudyType',
        'BriefSummary',
        #'LastKnownStatus',

    ]
    #OutcomeMeasuresModule 
    # new API fields
    params = {
        'fields': ",".join(extract_fields), 
        'query.term': query_term,
        'query.intr': query_intr,
        #'filter.overallStatus':'',
        'pageSize': max_studies, 
        'aggFilters':'results:with',
        'format': 'json', 
        #'pageToken': None  # first page doesn't need it
    }    
    all_studies = []
    next_page_token='a'
    page_temp = 0
    while next_page_token and page_temp<max_studies:

        #print(f'--- page: {page} ---')

        response = requests.get(base, params=params)

        if not response.ok:
            logging.info(f'response.text: {response.text}')
            break

        data = response.json()
        #print(data.keys())
        data_responses = data['studies']
        #print(data_responses)

        all_studies = all_studies+data_responses

        try:
        # next page
        #next_page_token = response.headers.get('x-next-page')  # (probably) for `CSV`
            next_page_token = data['nextPageToken']                 # for `JSON`
            #print(data['nextPageToken'])
        #print('x-next-page', next_page_token)    
            params['pageToken'] = next_page_token
        except KeyError:
            next_page_token=''
            pass
        page_temp+=1
        
    all_studies = pd.json_normalize(all_studies)
    all_studies.columns = [i.split('.')[-1] for i in \
                           all_studies.columns]

    if 'outcomeMeasures' not in all_studies.columns:
        all_studies['outcomeMeasures']=None 
    # раскрываем препараты из описания клин.испытаний

    # https://clinicaltrials.gov/study/NCT03792568 
    # here intervention == Other: ALK Inhibitor
    # [{'type': 'BIOLOGICAL', 'name': 'cetuximab'}, ?
    # мб убрать terminated suspended withdrawn unknown ...
    # в https://clinicaltrials.gov/study/NCT02510001?cond=Colorectal%20Cancer&intr=Crizotinib&rank=1 in interventions crizotinib is not mentioned
    if 'interventions' in all_studies.columns:
        all_studies['interventions'] = all_studies['interventions'].apply(lambda d: d if isinstance(d, list) else [{}]
                                      ).apply(lambda x: \
                                              [obj['name'].lower() for obj in x \
                                                   if obj.get('type','') in [#'BIOLOGICAL',
            'DRUG'] ])



    return all_studies


def treat_studies(treatment, all_studies):
    idx = [treatment in ' '.join(d) for d in all_studies.interventions]
    return all_studies[idx]


def the_func(x):
    if x:
        return Outcomes.model_validate_json(x)
    else:
        return ''


def merge_pydantic_models(base: T, nxt: T) -> T:
    """Merge two Pydantic model instances.

    The attributes of 'base' and 'nxt' that weren't explicitly set are dumped into dicts
    using '.model_dump(exclude_unset=True)', which are then merged using 'deepmerge',
    and the merged result is turned into a model instance using '.model_validate'.

    For attributes set on both 'base' and 'nxt', the value from 'nxt' will be used in
    the output result.
    """
    base_dict = base.model_dump(exclude_unset=True)
    nxt_dict = nxt.model_dump(exclude_unset=True)
    merged_dict = always_merger.merge(base_dict, nxt_dict)
    return base.model_validate(merged_dict)
    

def merge_pydantic_fs2(base: T, nxt: T) -> T:
    base_dict = base.model_dump(exclude_unset=True)
    nxt_dict = nxt.model_dump(exclude_unset=True)
    
    merged = base_dict.copy()
    for key, value in nxt_dict.items():
        if key in merged:
            # Check type to decide how to merge
            if isinstance(value, float):
                merged[key] = max(merged[key],value)#merged[key] or value 
            elif isinstance(value, str):
                merged[key] += value
                
    return base.model_validate(merged)        
 
def merge_pydantic_fs(base, nxt):
    d1 = base.model_dump(exclude_unset=True)
    d2 = nxt.model_dump(exclude_unset=True)

    result = {
        'main_reasoning': d1.get('main_reasoning', '') + ' | ' + d2.get('main_reasoning', '')
    }
    
    # 2. Создаем карту групп из первого словаря
    # group_name -> словарь данных
    groups_map = {g['group_name']: g.copy() for g in d1.get('groups_outcomes', [])}
    
    # 3. Сливаем данные из второго словаря
    for g2 in d2.get('groups_outcomes', []):
        name = g2['group_name']
        if name in groups_map:
            # Если группа совпала, мержим поля
            target = groups_map[name]
            for key, val in g2.items():
                if isinstance(val, str) and key != 'group_name':
                    # Склеиваем строки
                    target[key] = target.get(key, '') + ' | ' + val
                elif isinstance(val, (int, float)):
                    # Обновляем числовые значения 
                    target[key] = max(val, target[key])
        else:
            # Если группы нет, просто добавляем её целиком
            groups_map[name] = g2

    result['groups_outcomes'] = list(groups_map.values())
    return base.model_validate(result)    
    
def fix_outcome(outcome_list):
    search_r = "\\b(cr|complete\Wresponse|pr|partial\Wresponse|sd|stable\disease|ort|objective\Wresponse|dcr|disease\Wcontrol|pfs|progression\Wfree\Wsurvival|os|overall\Wsurvival)\\b"
    fin_l = []
    # filter by unit of measure
    fin_l = [obj for obj in outcome_list if(
                ('participant' in obj.get('unitOfMeasure','').lower()) or \
                ('month' in obj.get('unitOfMeasure','').lower())
    )
            ]
    # filter by needed measurements 
    #print([i.get('title','').lower()+i.get('description','').lower() for i in fin_l])
    #print('\n\n')
    has_measures = [re.search(search_r, 
                              i.get('title','').lower()+'. '+i.get('description','').lower()
                             ) for i in fin_l]
    return [fin_l[i] for i in range(len(has_measures)) if has_measures[i]!=None]


def ctrials_res(treatment,ec_pred, all_studies, ct_screen_thresh):
    # filtering by evaluations
    evals = ec_pred#[i.evaluations for i in ec_pred]
    word2int = {"YES": 1, "UNCERTAIN": 0,"NO": -1}
    new_evals = []
    for one_e in evals:
        new_evals.append([word2int.get(item, 0) for item in one_e ])
    new_evals = np.array(new_evals)    
    all_studies[['s1','s2']] = new_evals

    at_once = True
    
    chosen = all_studies[(all_studies['s1']>=ct_screen_thresh[0]
                      )&(all_studies['s2']>=ct_screen_thresh[1]
                        )&(all_studies.hasResults==True)
                    ]
    chosen['res_with_part'] = chosen['outcomeMeasures'].apply(lambda x: fix_outcome(x))
    to_work = chosen[chosen['res_with_part'].str.len()>0]
    #to_work = to_work.iloc[:2]
    prompt = PROMPT_RES_EXTRACTION.format(treatment=treatment,
            list_info=['Group name; description'
                        'Population size; number of participants',
                       'Complete Response (CR); percentage of participants',
                       'Partial Response (PR); percentage of participants',
                       'Objective Response Rate (ORR); percentage of participants',
                       'Stable Disease (SD); percentage of participants',
                      'Disease Control Rate (DCR); percentage of participants',
                       'Progression Free Survival (PFS); percentage of participants',
                      'Overall Survival (OS); percentage of participants',
                       'Progression Free Survival (PFS); months',
                      'Overall Survival (OS); months'
                      ]
                      )
    user_prompt=\
'''
# User input

input_measures={inp_measures}

# Your response

'''
    
    m=10
    char_t=3
    resultsct = {}
    if to_work.shape[0]:
        openai_client = OpenAI(
            base_url=os.getenv("BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
            http_client=httpx.Client(verify=False)
        )
        messages = []
        ctx_size=int(os.getenv("CONTEXT_SIZE", 8192))
        for ct_results, ct_id in zip(to_work.res_with_part.values, 
                          to_work.nctId.values):
            print(ct_id)
            print(len(ct_results))
            # if results for one ctrial are greater than context_size
            print( (len(user_prompt.format(inp_measures=ct_results))+len(prompt)+m)/char_t)
            if (len(user_prompt.format(inp_measures=ct_results))+len(prompt)+m)/char_t >= ctx_size:
            #if len(str(ct_results)) >= (ctx_size - len(prompt)-len(user_prompt.format(inp_measures=ct_results))-m):
                
                idx_res = 0
    
                for one_ct_res in ct_results:
                    # check if each result is greater
                    # just skip; todo: cut?
                    print( (len(user_prompt.format(inp_measures=one_ct_res))+len(prompt)+m)/char_t)
                    if (len(user_prompt.format(inp_measures=one_ct_res))+len(prompt)+m)/char_t >= ctx_size:
                    #if len(str(one_ct_res)) >= (ctx_size - len(prompt)-len(user_prompt.format(inp_measures=one_ct_res))-m):
                        print('huge_skip')
                    else:
                        messages.append([{'ct_id':ct_id},
                                         {'role':'system', 
                                          'content':prompt+' \no_think'},
                                         {'role':'user', 
                                          'content':user_prompt.format(inp_measures=one_ct_res)
                                         }])
                        idx_res+=1
                print(ct_id,' with several:',idx_res+1)
            else:    
                messages.append([{'ct_id':ct_id},
                                 {'role':'system', 
                                  'content':prompt+' \no_think'},
                                 {'role':'user', 
                                  'content':user_prompt.format(inp_measures=ct_results)
                                 }])
        
        if at_once:
            
            outputs = batch_function_call_openai([i[1:] for i in messages], 
                                      os.getenv('MODEL_NAME'), 
                                      [ClinicalTrialOutcomes],
                                      int(os.getenv('TEMPERATURE', 0)), 
                                      thinking=False)
            '''
            output1 = ClinicalTrialOutcomes(main_reasoning="Grou", groups_outcomes=[GroupOutcomes(reasoning="The va", group_name='Anti-Tumor ', population_size=6, complete_response=0.0, partial_response=16.66, objective_response_rate=16.66, stable_disease=66.66, disease_control_rate=83.32, progression_free_survival=0.0, overall_survival=0.0)])
            output2 = ClinicalTrialOutcomes(main_reasoning="AAAAAA", groups_outcomes=[GroupOutcomes(reasoning="WWWWWW", group_name='XXXX', population_size=6, complete_response=0.0, partial_response=16.66, objective_response_rate=16.66, stable_disease=66.66, disease_control_rate=83.32, progression_free_survival=6.99, overall_survival=99.99)])
            outputs=[output1,output2,output1,output2,output1]
            '''
            for output, id in zip(outputs, [i[0]['ct_id'] for i in messages]):
                #print(output, id )
                if id in resultsct.keys():
                    resultsct[id].append(output)
                else:
                    resultsct[id] = [output]

            logging.info(resultsct)
        else:
            #print(messages)
            #last_id = ''
            for message in messages:
                response = openai_client.chat.completions.parse(
                    model=os.getenv('MODEL_NAME'),
                    messages=message[1:], # skipping 'ct_id' info
                    temperature=0,
                    response_format=ClinicalTrialOutcomes,
                    extra_body={"reasoning_effort": "none"}
                )
                fin = response.choices[0].message.parsed
                # appending to existing ct_id, if it exists
                if message[0]['ct_id'] in resultsct.keys():
                    resultsct[message[0]['ct_id']].append(fin)
                else:
                    resultsct[message[0]['ct_id']] = [fin]
                #last_id = message[0]['ct_id']
                #answer = fin.strip('<think>\n\n</think>\n\n')
                #resultsct.append(fin)

        # filling df with extracted results
        to_work['res']=''
        for ct_id in to_work.nctId.values:
            ct_results = resultsct.get(ct_id,'')
            logging.info(ct_id)
            logging.info(len(ct_results))
            logging.info(resultsct.get(ct_id,''))
            if ct_results:
                if len(ct_results)>1:
                    zero_part = ct_results[0]
                    for one_part in ct_results[1:]:
                        zero_part = merge_pydantic_fs(zero_part, one_part)
                        
                    to_work.loc[to_work.nctId==ct_id, 'res'] = zero_part.model_dump_json()
                else:
                    to_work.loc[to_work.nctId==ct_id,'res'] = ct_results[0].model_dump_json()

    return resultsct, to_work


def eval_results(res_extracted, papers_res, fin_condition, 
                 treatment, fields):
    criteria_text = [i.format(fin_condition=fin_condition,
                              treatment=treatment) \
                     for i in criteria_text3]
    
    n_criteria = len(criteria_text)
    logging.info(criteria_text)
    #print(ENHANCE_RESULTS_PROMPT2[:3000])
    class SentenceEval(BaseModel):
        enhanced_ver: str = Field(description="A rationale for each criteria evaluation",
                                      max_length=300)
        evaluations: conlist(Literal['YES', 'NO', 'UNCERTAIN', '1','2','3','4','5'], 
                             min_length=n_criteria, max_length=n_criteria
                            ) = Field(description="Evaluations for criteria")
        rationale: conlist(str,min_length=n_criteria, 
                           max_length=n_criteria
                          ) = Field(description="A rationale for each criteria evaluation")

    all_resp = []
    cit_in=3
    papers_res_fin = papers_res.copy()
    papers_res_fin['enhanced'] = ''
    for paper_id,paper_ress in enumerate(res_extracted):
        logging.info(paper_id)
        paper_list = f"<source id=\"{paper_id}\">"
    
        paper_ress_df = papers_res.iloc[paper_id]
        for res_id,one_res in enumerate(paper_ress):
            if one_res.fieldresult[0].value!="NP":
                try:
                    paper_list+=f'''<result id=\"{res_id}\">{one_res.fieldresult[0].value}</result>'''+\
                        f'''<context id=\"{res_id}\">'''+\
                        f'''{'. /n/n '.join(one_res.fieldresult[0]._cited_blocks[:cit_in])}</context>'''
                except AttributeError:
                    cited_blocks = ast.literal_eval(paper_ress_df[f'citations_{res_id}'])[:cit_in]
                    
                    paper_list+=f'''<result id=\"{res_id}\">{paper_ress_df[f'result_{res_id}']}</result>'''+\
                        f'''<context id=\"{res_id}\">'''+\
                        f'''{'. /n/n '.join(cited_blocks)}</context>'''
        paper_list+="</source>"
        #print(paper_list)
        
        fin_prompt = ENHANCE_RESULTS_PROMPT2.format(n_fields=len(fields),
                                                      fin_condition=fin_condition,
                                                      treatment=treatment
                                                   )
        user_part = '''
        - Provided results: {text_chunks}
        - Fields: {fields}
        - Criteria for inclusion: {criteria_text}
        '''
        user_part = user_part.format(text_chunks=paper_list,
                                      fields=[i.format(fin_condition=fin_condition,
                                                           treatment=treatment)\
                                              for i in  fields],
                                      criteria_text=[i.format(fin_condition=fin_condition,
                                                           treatment=treatment)\
                                              for i in  criteria_text]
                                    )
        #print(paper_ress)
        messages=[{'role':'system',
                   'content':fin_prompt},
                 {'role':'user',
                   'content':user_part}]
        logging.info(f'len_eval_p {len(fin_prompt)}')
        response = use_llm(os.getenv("MODEL_NAME"), messages, tool=[SentenceEval])
        
        papers_res_fin.iloc[paper_id,
                            papers_res_fin.columns.get_loc('enhanced')] = response.model_dump_json()
        logging.info(response)
        logging.info('')
        all_resp.append(response)
        
    return all_resp, papers_res_fin 
    

def choose_top_n(fin_condition, treatment, all_resp, paper_ids, top_n):
    # _______ choosing top 10
    full_pr=[]
    full_pr_text = []
    p_id=0
    chosen_p = []


    for paper_id,res in enumerate(all_resp):
        #print(paper_id)
        if ('YES' not in res.evaluations[:1]) and ('NO' not in res.evaluations[1:2]):
            paper_list = f"<source id=\"{p_id}\">"
            paper_list+=f"{res.enhanced_ver}"
            
            paper_list+="</source>"
            #print(paper_list)
            full_pr.append(paper_list)
            full_pr_text.append(res.enhanced_ver)
            chosen_p.append(paper_ids[paper_id])
            p_id+=1
        else:
            #paper_ids.pop(paper_id)
            #paper_ids = np.delete(paper_ids, paper_id)
            pass#print(res.rationale[:2])
    text_chunks = ''.join(full_pr)
    paper_ids = chosen_p
    logging.info(paper_ids)
    
    class ResultEval(BaseModel):
        id: int = Field(description='Index of the result')
        decision: Literal['YES', 'NO'] = Field(description='Decision to include: YES (to include) or NO (to exclude)')
        reasoning: str = Field(description='Reasoning for including or excluding the result')
    class ResultsEval(BaseModel):
        chosen_ids: list[int] = Field(description='Indexes of top-10 results according to the criteria',
                                     max_length=top_n)
        result_eval: list[ResultEval] = Field(description='Detailed reasoning for each result starting from 0',
                                             min_length=len(full_pr), max_length=len(full_pr))
    
    fin_prompt2 = RESULTS_TOP_PROMPT.format(
        top_n=top_n,
        text_chunks=text_chunks,
                              fin_condition=fin_condition,
                              treatment=treatment)
    
    user_part = '''
    text_chunks = \"\"\"{text_chunks}\"\"\"
    '''
    user_part = user_part.format(text_chunks=text_chunks)
    
    messages=[{'role':'system',
               'content':fin_prompt2},
               {'role':'user',
               'content':user_part}]

    logging.info(f'len_top10_p {len(fin_prompt2)}, {len(text_chunks)}, {len(paper_ids)}')
    response2 = use_llm(os.getenv("MODEL_NAME"), messages, tool=[ResultsEval])
    
    top_chosen_idx = response2.chosen_ids

    chosen_text_fin=[full_pr_text[i] for i in top_chosen_idx if i<len(full_pr_text)]
    chosen_id_fin=[paper_ids[i] for i in top_chosen_idx if i<len(full_pr_text)]
    return response2, chosen_text_fin, chosen_id_fin


def combine_res(fin_condition, treatment, 
                fin_res, pmid_list):
    def replacement_match(match):
        #print(match.groups()[0])
        return f'[[{pmid_list[int(match.groups()[0])]}]]'

    # _______ combining in one paragraph
    full_pr=[]
    for paper_id,res in enumerate(fin_res):
        paper_list = f"<source id=\"{paper_id}\">"
        paper_list+=f"{res}"
        paper_list+="</source>"
        #print(paper_list)
        full_pr.append(paper_list)
    text_chunks = ''.join(full_pr)
    
    
    
    prompt = RESULTS_SENTENCE_PROMPT.format(text_chunks=text_chunks,
                                       fin_condition=fin_condition)
    logging.info(f'len_combine_p {len(prompt)}, {len(text_chunks)}')
    messages=[{'role':'user',
               'content':prompt
              }]
    response = use_llm(os.getenv("MODEL_NAME"), messages)
    return re.sub(r'\[\[(\d+)\]\]', replacement_match, response)