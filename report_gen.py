# -*- coding: utf-8 -*-

import logging
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning, message=".*escape sequence.*")

import pandas as pd
import numpy as np
import re
import time
import os
import httpx
from dotenv import load_dotenv, find_dotenv
from markdown_pdf import MarkdownPdf, Section
import ast

from pydantic import BaseModel, validator, Field,field_validator, conlist 
from typing_extensions import Literal
from typing import Dict,TypeVar

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
# for importing trialmind, api_key should be set beforehand
from trialmind.pubmed import pmid2papers, PubmedAPIWrapper, parse_bioc_xml, pmid2fulltext
from trialmind.api import StudyCharacteristicsExtraction, LiteratureScreening, CTScreening
import extract
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams
from langchain_openai import OpenAIEmbeddings
import random
from aux_prompts import TRANSLATE_PROMPT,ENHANCE_RESULTS_PROMPT2,RESULTS_TOP_PROMPT

from queries import criteria_text3 as criteria_text
from trialmind.pydantic_classes import ClinicalTrialOutcomes,GroupOutcomes

logger = logging.getLogger('my_module_name').setLevel(logging.INFO)
# Silence other loggers
for log_name, log_obj in logging.Logger.manager.loggerDict.items():
    if log_name != 'my_module_name':
        log_obj.disabled = True
# to override existing env.variables        
load_dotenv(find_dotenv(usecwd=True), override=True)


#__________________
text_splitter=RecursiveCharacterTextSplitter(chunk_size=256, 
                                             chunk_overlap=100)


# Create a Qdrant client for local storage


def create_vector_store(cname='hybrid_fin0', col_name=''):
    #cname='hybrid'
    
    #cname = f"{cname}_{random.random():.3f}"
    if col_name=='':
        col_name = cname
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

    emb = OpenAIEmbeddings(model='Qwen/Qwen3-Embedding-4B-GGUF',
                              base_url='http://localhost:8080/v1/',
                            api_key=os.getenv("OPENAI_API_KEY"),
                           check_embedding_ctx_length =False,
                           #encoding_format ='float',
                            http_client=httpx.Client(verify=False)
                          )
    if not os.path.exists(f"tmp/{cname}"):
        # client creation is in both clauses, because it creates the directory,
        # the existance of which we check first
        logging.info(f"creating new db")
        clientqh = QdrantClient(path=f"tmp/{cname}")
        # Create a collection with both dense and sparse vectors
        clientqh.recreate_collection(
            collection_name=col_name,
            vectors_config={"dense": \
                            VectorParams(size=len(emb.embed_query('ggg')), 
                                         distance=Distance.COSINE)},
            
            sparse_vectors_config={
                "sparse": \
                SparseVectorParams(index=\
                                   models.SparseIndexParams(on_disk=False))
            },
        )
    else:
        logging.info(f"using existing db")
        clientqh = QdrantClient(path=f"tmp/{cname}")  
        
    vectore_store_qh = QdrantVectorStore(
            client=clientqh,
            collection_name=col_name,
            embedding=emb,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
            #force_recreate=True  # This forces deletion of old and creation of new
        )
        
    clientqh.collection_name=col_name
    logging.info(f"vector store NAME: {cname}")
    #clientqh.close()
    return clientqh,vectore_store_qh,cname,sparse_embeddings,emb


def get_res(db_name='hybrid_fin0',
            file_path = "docs/LuC_213_L00_DL_edited_oncobox_ru.pdf",
            model_translate='qwen3:8b', model_main='qwen3:8b',
            n_papers=5,ct_pages=1,
            ct_criteria = \
            ["Does the trial focus on patients with '{fin_condition}'?",
             "Does the trial examine the use or sensitivity of '{treatment}' among main treatments?"],
            papers_criteria=\
            ["Does the study focus on patients/models/cells with '{fin_condition}'?",
             "Does the study examine the use/effect/sensitivity of '{treatment}' among main treatments?", 
             "Does the study describe the effect of '{treatment}' treatment?"],
            fields=['{treatment} effectiveness | string | The outcome of treating {fin_condition} with {treatment}',
                   "Type of study | string | Is the study in vitro, in vivo, clinical trial or others",
                   "Num participants | string | How many patients with {fin_condition} are treated with {treatment} in the study",
            ],
            ct_screen_thresh=[1,0],
            paper_screen_thresh = [1,1,0],
            save_files=True, n_treats=1, skip_ct=False
           ):    

    start = time.time()
    start_all = start
    
    # translation (as a task) is simple => we use a smaller llm to get results quicker
    os.environ["MODEL_NAME"] = model_translate
    logging.info(os.getenv('MODEL_NAME'))
    # get main info from .pdf
    logging.info('GETTING INFO FROM FILE')

    #TODO: add rus-eng table for translations
    fin_condition,treatments_eng,df2,fin_condition_ru,treatments_ru = extract.info_from_doc(file_path, with_ru=True)
    
    logging.info(f'{fin_condition}, {treatments_eng[:20]}')
    end = time.time()
    logging.info(end-start)
    start = end
    
    # using a larger llm for better results
    #os.environ["MODEL_NAME"] ='qwen3:14b'
    os.environ["MODEL_NAME"] = model_main
    if n_treats == 'all':
        idx_list = len(treatments_eng)
    else:
        idx_list = n_treats

    #clientqh,vectore_store_qh,cname,sparse_embeddings,emb = create_vector_store(db_name)
    if not os.path.exists(f"res_files/{fin_condition.replace(' ','_')}/"):
        os.makedirs(f"res_files/{fin_condition.replace(' ','_')}/")

    the_path = f"res_files/{fin_condition.replace(' ','_')}"
    
    for treatment in treatments_eng[0:idx_list]:
        logging.info(treatment)
        try:
            if not os.path.exists(f"{the_path}/ctrials_res_df_{treatment.replace(' ','_')}.csv") and (ct_pages>0):
                # ________________Get clinical trials
                logging.info(f'\nGETTING CLINICAL TRIALS for {treatment}')
                
                if not os.path.exists(f"{the_path}/ctrials_all_df_{treatment.replace(' ','_')}.csv"):
                    all_studies = extract.get_clinicaltrials(f'''"{fin_condition}"''', 
                                                          #' OR '.join(treatments_eng), 
                                                           treatment,  
                                                          max_studies=ct_pages)#.iloc[:2]
                    logging.info(all_studies.shape)
                    # if there are any clinical trials
                    if (all_studies.shape[0]):
                        if ('briefTitle' in all_studies.columns) & ('briefSummary' in all_studies.columns):
                            all_text = all_studies.briefTitle.fillna("") + ": " + all_studies.briefSummary.fillna("")
                        else:
                            all_text = pd.Series([""]*all_studies.shape[0])
                        api = CTScreening()
                        end = time.time()
                        logging.info(end-start)
                        start = end
                        
                        logging.info('SCREENING CLINICAL TRIALS')
                        ec_pred = api.run(
                            llm = os.getenv("MODEL_NAME"),
                            criteria = [i.format(fin_condition=fin_condition,
                                                 treatment=treatment) for i in ct_criteria],
                            papers = all_text.values.tolist(), # make for the top-100 for demo
                        )
                        all_studies[['s1','s2']] = [i.evaluations for i in ec_pred]
                        all_studies[['r1','r2']] = [i.rationale for i in ec_pred]
                        if save_files and all_studies.shape[0]:
                            all_studies.to_csv(f"{the_path}/ctrials_all_df_{treatment.replace(' ','_')}.csv", 
                                      index=False)
                        end = time.time()
                        logging.info(end-start)
                        start = end
                else:
                    all_studies = pd.read_csv(f"{the_path}/ctrials_all_df_{treatment.replace(' ','_')}.csv")
                    all_studies['outcomeMeasures'] = all_studies['outcomeMeasures'].apply(lambda x: ast.literal_eval(x))
                if (all_studies.shape[0]):
                    logging.info('EXTR RESULTS FROM CLINICAL TRIALS')
                    ctrials_fin, chosen_cts = extract.ctrials_res(treatment,
                                                                  all_studies[['s1','s2']].values, 
                                                                  all_studies, ct_screen_thresh)
                    
                    #chosen_cts['res']= [i.model_dump_json() for i in ctrials_fin]
                    if save_files and chosen_cts.shape[0]:
                        chosen_cts.to_csv(f"{the_path}/ctrials_res_df_{treatment.replace(' ','_')}.csv", 
                                  index=False)
                    logging.info(str(ctrials_fin)[:1000])
                    end = time.time()
                    logging.info(end-start)
                    start = end
    
            if (n_papers > 0) and (not os.path.exists(f"{the_path}/papers_res_df_{treatment.replace(' ','_')}.csv")):
                clientqh,vectore_store_qh,cname,sparse_embeddings,emb = create_vector_store(f"hybrid_one_db_{treatment.replace(' ','_')}")
                if not os.path.exists(f"{the_path}/papers_all_df_{treatment.replace(' ','_')}.csv"):
                    # ________________Get papers
                    logging.info('GETTING PAPERS')
                    search_api = PubmedAPIWrapper()
                        # page_size is the max number of records to return!!!! not pages!
                    tmp_inputs = {
                            "page_size": n_papers,
                            "keyword_map": {'conditions':[fin_condition], 
                                            'treatments':[treatment]
                                           },
                            "keywords": {
                                "OPERATOR": 'AND'
                            }
                    }
                    response = search_api.build_search_query_and_get_pmid(tmp_inputs, 
                                                                          api_key=os.getenv("PUBMED_API_KEY"))
        
                    logging.info(search_api._build_query(tmp_inputs, ''))
                    logging.info(f'{response[0],len(response[0])}')
                    df_papers = pmid2papers(pmid_list=response[0], 
                                            api_key=os.getenv("PUBMED_API_KEY"))
        
                    if df_papers[0] is not None:
                        papers = df_papers[0]["Title"] + ": " + df_papers[0]["Abstract"].fillna("")
                        papers = papers.tolist()
                        end = time.time()
                        logging.info(end-start)
                        start = end
                            # screening
                        logging.info('SCREENING PAPERS')
                        
                        api = LiteratureScreening()
                        ec_predP = api.run(
                            population = f"Patients with {fin_condition} undergoing treatment with {treatment}",
                            intervention = f"{treatment}",
                            comparator = "",
                            outcome = "",
                            llm = os.getenv("MODEL_NAME"),
                            criteria = [i.format(fin_condition=fin_condition,
                                                 treatment=treatment) for i in papers_criteria],
                            papers = papers, 
                        )
                
                        evalsP = [i.evaluations for i in ec_predP]
                        word2int = {"YES": 1, "UNCERTAIN": 0, "NO": -1}
                        new_evalsP = []
                        for one_e in evalsP:
                            new_evalsP.append([word2int.get(item, 0) for item in one_e ])
                        new_evalsP = np.array(new_evalsP)  
                        df_p_e = df_papers[0]
                        df_p_e[['s1','s2','s3']] = new_evalsP
                        df_p_e[['r1','r2','r3']] = [i.rationale for i in ec_predP]
                        if save_files and df_p_e.shape[0]:
                            df_p_e.to_csv(f"{the_path}/papers_all_df_{treatment.replace(' ','_')}.csv", 
                                      index=False)
                else:
                    df_p_e = pd.read_csv(f"{the_path}/papers_all_df_{treatment.replace(' ','_')}.csv").iloc[:100] # earlier n_papers was set to 150
                    
                if df_p_e.shape[0]:
                    chosen_df = df_p_e[(df_p_e['s1']>=paper_screen_thresh[0]
                                       )&(df_p_e['s2']>=paper_screen_thresh[1]
                                         )&(df_p_e['s3']>=paper_screen_thresh[2])]
                    logging.info(chosen_df.shape[0])
                    end = time.time()
                    logging.info(end-start)
                    start = end
            
                    #end6=time.time()
                    # if there are papers left AFTER screening
                    if chosen_df.shape[0]:
                        # ________________To RAG
                            # full texts
                        pmid_list = chosen_df.PMID.values.tolist()
                        #['41213063',#'26451310',]
                        logging.info('GETTING FULL TEXT PAPERS')
                        res = pmid2fulltext(pmid_list, api_key=os.getenv("PUBMED_API_KEY"))
                        res = [parse_bioc_xml(r) for r in res]
            
                        # transform the parsed xml into paper content
                        papers0 = []
                        for parsed in res:
                            paper_content = []
                            for parsed_ in parsed["passage"]:
                                paper_content.append(parsed_['content'])
                            paper_content = "\n".join(paper_content)
                            papers0.append(paper_content)
            
                        chosen_df['FullText'] = ''
                        chosen_df['FullText'] = papers0
            
                        pmid_list = chosen_df.PMID.values.tolist()
                        papers_ch = chosen_df.FullText.values
                        docs =  [Document(page_content=i, 
                                          metadata={"source": str(j)}
                                         ) for i,j in zip(papers_ch,pmid_list)]
                        end = time.time()
                        logging.info(end-start)
                        start = end
                        
                        logging.info('EMBEDDING...')
            
                        # embedding papers which are not in the DB
                        paps=0
                        for one_doc, doc_id in zip(docs,pmid_list):
                            searched = vectore_store_qh.similarity_search(
                                                                query=" ",k=1,
                                                                filter=models.Filter(
                                                                    must=[models.FieldCondition(
                                                                            key="metadata.source",
                                                                            match=models.MatchValue(value=str(doc_id)),
                                                                        )
                                                                    ],
                                                                ),
                                                            )
                            
                            if len(searched) == 0:
                                paps+=1
                                all_splits = text_splitter.split_documents([one_doc])
                                _ = vectore_store_qh.add_documents(documents =all_splits)
                        logging.info(f"papers embedded and new: {paps}/{len(pmid_list)}")
                        #vector_store.add_documents(documents=all_splits)
                        logging.info('FIN EMBEDDING')
                        end = time.time()
                        logging.info(end-start)
                        start = end
                        

                        ii =len(pmid_list) #4
                        logging.info('EXTR RESULTS FROM PAPERS')
                        api = StudyCharacteristicsExtraction()
                        res_extracted = api.run(
                            papers_inp=[pmid_list[:ii],papers_ch[:ii]],
                            #fields=[f'The effectiveness of treating {fin_condition} with {treatments_eng[0]}',
                            #       ],
                            fields=[i.format(fin_condition=fin_condition,
                                             treatment=treatment) for i in fields],
                            llm=os.getenv("MODEL_NAME"),
                            chunk_size=0,
                            chunk_overlap=0,
                            thinking=False,
                            vector_store = clientqh,
                        )


                        ok_papers = list(res_extracted.keys())
                        results_per_field = list(res_extracted.values())
                        # to size (n_papers, n_fields)
                        np_rr = np.array(results_per_field)
                        #np_rr = np.array(res_extracted).reshape(-1,len(fields))
                        
                        # unravelling results for the df
                        full_pr = []
                        for paper_ress in np_rr:
                            paper_list = []
                            for one_res in paper_ress:
                                paper_list+=[one_res[-1].fieldresult[0].value, 
                                             one_res[-1].fieldresult[0].source_id,
                                             one_res[-1].fieldresult[0]._cited_blocks,
                                             one_res[-1].model_dump_json()]
                            full_pr.append(paper_list)
                        # creating columns     
                        cols=[]
                        for idx in range(len(fields)):
                            cols+= [f'result_{idx}', f'idxs_{idx}', 
                                    f'citations_{idx}', f'class_{idx}']
                        
                        papers_fin_df = pd.DataFrame.from_records(full_pr, 
                                                                  columns=cols)
                        papers_fin_df['id'] = ok_papers
                        
                        logging.info(f'to save paper res {save_files}, {papers_fin_df.shape[0]}')
                        if save_files and papers_fin_df.shape[0]:
                            papers_fin_df.to_csv(f"{the_path}/papers_res_df_{treatment.replace(' ','_')}.csv", 
                                                 index=False)
                        end = time.time()
                        logging.info(end-start)
                        start = end
                        end = time.time()
                        logging.info(end-start)
                        start = end
            clientqh.close()    
        except Exception as e:   
            logging.info(f'exception:  {e}')
            try:
                clientqh.close()
            except Exception as e2: 
                logging.info(f'cannot close client {e2}')
            pass
    
    end = time.time()
    logging.info(f'FUll time: {end - start_all}')
    #clientqh.close()
    return treatments_eng, fin_condition, fin_condition_ru


def fill_pdf(treatments_eng, fin_condition, fin_condition_ru,
             model_main='qwen3:8b', model_translate='qwen3:8b', n_treats=1, 
             path_file='', fields=[], show_ct=False):
    user_css = """
    h1, h2, h3  { font-size: 14px; } 
    h1 { text-align: center;} 
    h3 { font-weight: normal;}
    body { font-size: 12px; text-justify: inter-word; text-align:justify;}
    table {border-collapse: collapse;} 
    table, th, td {border: .1px solid black; font-size: 8px}
    td, th {padding: 3px;}
    """
    os.environ["MODEL_NAME"] = model_main
    if '16c' in model_main:
        os.environ["CONTEXT_SIZE"] = "16384"
        
    ru_en_drugs = pd.read_csv('docs/ru_en_drugs.csv', usecols=[0,1])
    fin_glossary = f"\nENG {fin_condition.lower()} -> RUS {fin_condition_ru.lower()}"
    
    aux_g = pd.read_csv('docs/aux_glossary.csv')
    aux_g_items = ''.join(('\nENG '+aux_g['eng']+' -> RUS '+aux_g['rus']).values)
    fin_glossary += aux_g_items
    logging.info(fin_glossary)
    
    class index2paper:
        def __init__(self,papers):

            self.dict_index_paper = {}
            self.last_index = 0
            self.index_start=0
            self.papers=papers
        def replacement_match(self,match):
            found_id = match.groups()[0]
            self.last_index+=1
            #logging.info(self.last_index)
            try:
                #logging.info(found_id, papers[papers.PMID==int(found_id)].Title.values[0])
                self.dict_index_paper[self.last_index]=[self.papers[self.papers.PMID==int(found_id)
                                                ].Title.values[0],
                                    'https://pubmed.ncbi.nlm.nih.gov/'+str(found_id)]
            except IndexError:
                pass
            return f'[{self.last_index}]'

    class FieldResult(BaseModel):
        #name: Literal[fields[0].split(',')[0]] = Field( description='Field name that accurately represents the content of the field based on its description.')
        name: str = Field(description='Field name that accurately represents the content of the field based on its description.',
         max_length=100)
        value: str = Field(description='Extracted information from the text based on the field description.',
                          max_length=200)
        source_id: conlist(int) = Field(description='Cited document IDs.')
    class Results(BaseModel):
        fieldresult: list[FieldResult] = Field(min_length=1, max_length=1) 
        
    n_criteria = len(criteria_text)
    logging.info(f'in reportgen {n_criteria}')
    class SentenceEval(BaseModel):
        enhanced_ver: str = Field(description="A rationale for each criteria evaluation",
                                      max_length=300)
        evaluations: conlist(Literal['YES', 'NO', 'UNCERTAIN', '1','2','3','4','5'], 
                             min_length=n_criteria, max_length=n_criteria
                            ) = Field(description="Evaluations for criteria")
        rationale: conlist(str,min_length=n_criteria, 
                           max_length=n_criteria
                          ) = Field(description="A rationale for each criteria evaluation")

    
    pdf = MarkdownPdf(toc_level=4, optimize=True)
    eng_diagnosis_header = f"""# Diagnosis: {fin_condition}\n"""
    eng_text2 = eng_diagnosis_header
    
    ru_diagnosis_header = f"""# Диагноз: {fin_condition_ru}\n"""
    ru_text2 = ru_diagnosis_header
    
    all_lit = ''
    final_lit_idx=0
    
    top_summ_file = f"res_files/{fin_condition.replace(' ','_')}/top_{fin_condition.replace(' ','_')}.csv"
    if not os.path.exists(top_summ_file):
        top_df = pd.DataFrame([['','','']], 
                              columns=['treatment', 'eng','rus'])
        #top_df.to_csv(top_summ_file, index=False)
    else:
        top_df = pd.read_csv(top_summ_file).fillna('')
    
    if n_treats == 'all':
        idx_list = len(treatments_eng)
    else:
        idx_list = n_treats
        
    the_path = f"res_files/{fin_condition.replace(' ','_')}"    
    n_fields = len(fields)    
    add_to_top_df = {}
    
    for treat_idx, treatment in enumerate(treatments_eng[:idx_list]):
        try:
            ru_treat_df = ru_en_drugs[ru_en_drugs.drug==treatment]['name']
            if ru_treat_df.shape[0]:
                ru_treatment = ru_treat_df.values[0]
            else:
                ru_treatment = treatment
    
            papers = pd.read_csv(f"{the_path}/papers_all_df_{treatment.replace(' ','_')}.csv").iloc[:100]
            papers_res0 = pd.read_csv(f"{the_path}/papers_res_df_{treatment.replace(' ','_')}.csv").iloc[:100]
            
            papers_res = papers_res0[papers_res0['result_0']!='NP']
            all_res = papers_res[[f'class_{i}' for i in range(n_fields)]].values#.apply(lambda x: "{x}")
            class_ress = [Results.model_validate_json(one_res) for paper_ress in all_res for one_res in paper_ress ]
                
            all_class_res = np.array(class_ress).reshape(-1,n_fields)

            
            # _________TRIALS
            if os.path.exists(f"{the_path}/ctrials_res_df_{treatment.replace(' ','_')}.csv") and show_ct:
                trials = pd.read_csv(f"{the_path}/ctrials_res_df_{treatment.replace(' ','_')}.csv")
                # filtering
                trials['ct_coverage']=0
                trials['ct_pop']=0
                trial_res = trials['res'].apply(lambda x: ClinicalTrialOutcomes.model_validate_json(x)).values
                
                for idx,trial_mess in enumerate(trial_res):
                    #print(trial_mess.groups_outcomes)
                    pop_all = []
                    n_with_vals = []
                    for group_res in trial_mess.groups_outcomes:
                        
                        pop_all.append(group_res.population_size)
                        measures=np.array([group_res.complete_response,
                                           group_res.partial_response,
                                         group_res.objective_response_rate,
                                           group_res.stable_disease,
                                         group_res.disease_control_rate,
                                           group_res.progression_free_survival,
                                         group_res.overall_survival])
                        n_with_vals.append(measures[measures!=0].shape[0])
                        
                    trials.iloc[idx,
                            trials.columns.get_loc('ct_pop')] = sum(pop_all)
                    trials.iloc[idx,
                            trials.columns.get_loc('ct_coverage')] = max(n_with_vals)
                # taking only cts with at least 1 value
                trials = trials[trials.ct_coverage>0]
                # top 10 is just choosing first 10
                trials = trials.sort_values(by=['ct_pop','ct_coverage'], 
                                            ascending=False).iloc[:10]
                
                trial_res = trials['res'].apply(lambda x: ClinicalTrialOutcomes.model_validate_json(x)).values

                
                eng_treat_ct = f"""\n## Treatment {treat_idx+1}: {treatment}\n\n### Chosen clinical trials:\n"""
                ru_treat_ct = f"""\n## Препарат {treat_idx+1}: {ru_treatment}\n\n### Выбранные клинические испытания:\n"""
                
                table_ct="||Title|Phase|Group|Population|CR|PR|ORR|SD|DCR|PFS|OS|\n"+\
                "|--|--|--|--|--|--|--|--|--|--|--|--|\n"
                ru_table_ct="||Название|Этап|Группа|Популяция|CR|PR|ORR|SD|DCR|PFS|OS|\n"+\
                "|--|--|--|--|--|--|--|--|--|--|--|--|\n"
                num = 1
                for i,trial_mess in zip(trials.values,trial_res):
                    ct_idx = num 
                    ct_title = i[2]
                    ct_phase = f"{i[3].lower()} {' '.join(ast.literal_eval(i[7])).lower()}"
                    ct_link = f'[{i[1]}](https://clinicaltrials.gov/study/{i[1]})'
                    print(len(trial_mess.groups_outcomes))
                    if len(trial_mess.groups_outcomes)>1:
                        table_row=''
                        for idx_g, group_res in enumerate(trial_mess.groups_outcomes):
                            print(group_res.group_name)
                            if idx_g==0:
                                table_row = f"{ct_idx}|{ct_title} ({ct_link})|{ct_phase}|{group_res.group_name}|{group_res.population_size}|{group_res.complete_response}|{group_res.partial_response}|{group_res.objective_response_rate}|{group_res.stable_disease}|{group_res.disease_control_rate}|{group_res.progression_free_survival}|{group_res.overall_survival}|\n"
                            else:
                                table_row = f"||||{group_res.group_name}|{group_res.population_size}|{group_res.complete_response}|{group_res.partial_response}|{group_res.objective_response_rate}|{group_res.stable_disease}|{group_res.disease_control_rate}|{group_res.progression_free_survival}|{group_res.overall_survival}|\n"
                
                            table_ct += table_row
                            ru_table_ct += table_row
                    else:
                        group_res = trial_mess.groups_outcomes[0]
                        table_row = f"{ct_idx}|{ct_title} ({ct_link})|{ct_phase}|{group_res.group_name}|{group_res.population_size}|{group_res.complete_response}|{group_res.partial_response}|{group_res.objective_response_rate}|{group_res.stable_disease}|{group_res.disease_control_rate}|{group_res.progression_free_survival}|{group_res.overall_survival}|\n"
                        table_ct += table_row
                        ru_table_ct += table_row
                    
                    num+=1
                #print(table_ct)
                
                eng_treat_ct += table_ct    
                ru_treat_ct += ru_table_ct
                
            else:
                eng_treat_ct = f"""\n## Treatment {treat_idx+1}: {treatment}\n\n"""
                ru_treat_ct = f"""\n## Препарат {treat_idx+1}: {ru_treatment}\n\n"""
                
            # ________PAPERS
            k = index2paper(papers=papers)
            k.last_index=final_lit_idx
            k.index_start=final_lit_idx

            start = time.time()
    
            # _____ enhancing and evaluating
            top_n = 10
            logging.info(f'top {top_n} '+treatment)
            if 'enhanced' not in papers_res.columns:
                eval_ress, papers_res = extract.eval_results(all_class_res, papers_res, 
                                              fin_condition, treatment, fields)
                # there may be nan
                papers_with_enh = papers_res[papers_res['enhanced'].str.len()>3]
                logging.info('save '+treatment)
                logging.info(papers_res.head(1))
                papers_res.to_csv(f"{the_path}/papers_res_df_{treatment.replace(' ','_')}.csv",index=False)
            else:
                papers_with_enh = papers_res[papers_res['enhanced'].str.len()>3]
                eval_ress = [SentenceEval.model_validate_json(one_res) \
                             for one_res in papers_with_enh['enhanced'].values]
            end = time.time()    
            logging.info(f'enhancing {end-start}')

            
            # filtering
            ctx_size = int(os.getenv("CONTEXT_SIZE", 8192))
            logging.info(f'ctx_size {ctx_size}')
            fin_p = RESULTS_TOP_PROMPT.format(text_chunks='',top_n=top_n,
                                  fin_condition=fin_condition,
                                  treatment=treatment)
            char_t=3
            logging.info(f'fin resulttop prompt {len(fin_p)}')
            if 'enh_html_len_cs' not in papers_with_enh.columns:
                evals = [i.evaluations for i in eval_ress]
                word2int = {"YES": 1, "UNCERTAIN": 0,"NO": -1, 
                            '1':1,'2':2,'3':3,'4':4,'5':5}
                new_evals = []
                for one_e in evals:
                    new_evals.append([word2int.get(item, 0) for item in one_e ])
                new_evals = np.array(new_evals)  

                html_part = len("<source id=99></source>")
                papers_with_enh['enh_html_len'] = [len(i.enhanced_ver)+html_part \
                                                   for i in eval_ress]
                papers_with_enh[[f'e_{i}' for i in range(len(evals[0]))]] = new_evals
                papers_with_enh = papers_with_enh.sort_values(by=
                                          ['e_0','e_1','e_2','e_4','e_3'], 
                                          ascending=[True,False,False,True,False]
                               )
                papers_with_enh['enh_html_len_cs'] =  \
                                papers_with_enh['enh_html_len'].cumsum()
                papers_eng_enough = papers_with_enh[papers_with_enh['enh_html_len_cs']< \
                                    (ctx_size-len(RESULTS_TOP_PROMPT))/char_t]
                (papers_res0.merge(papers_with_enh[['id', 'enh_html_len', 
                                                     'e_0', 'e_1', 'e_2', 'e_3', 'e_4', 'enh_html_len_cs'
                                                    ]], how='left',on='id'
                                 )
                    ).to_csv(f"{the_path}/papers_res_df_{treatment.replace(' ','_')}.csv",
                             index=False)
            else:
                papers_with_enh = papers_with_enh.sort_values(by=
                                         ['e_0','e_1','e_2','e_4','e_3'], 
                                          ascending=[True,False,False,True,False]
                               )
                papers_eng_enough = papers_with_enh[papers_with_enh['enh_html_len_cs']< \
                                    (ctx_size-len(RESULTS_TOP_PROMPT))/char_t]

            eval_ress = [SentenceEval.model_validate_json(one_res) \
                             for one_res in papers_eng_enough['enhanced'].values]
            
            paper_ids = papers_eng_enough.id.values
            
            start = time.time()
            # choosing top10
            if 'top_reason' not in papers_res.columns:
                resp_top, top_text, \
                    top_papers = extract.choose_top_n(fin_condition, treatment,
                                                      eval_ress,paper_ids, top_n)
                papers_res['top_reason']=''
                papers_res['top_order']=999
                logging.info(f'responce with reason {resp_top}')
                for order_i, (one_top_paper, one_chosen_id) in \
                            enumerate(zip(top_papers,  resp_top.chosen_ids)):
                    one_top_reason = [j for j in resp_top.result_eval \
                                      if j.id==one_chosen_id]
                    papers_res.loc[papers_res.id==one_top_paper,
                                'top_reason'] = one_top_reason[0].model_dump_json()
                    papers_res.loc[papers_res.id==one_top_paper,
                                'top_order'] = order_i            
                    #logging.info(order_i, one_top_paper, one_top_reason, one_chosen_id)
                logging.info('save '+treatment)
                papers_res.to_csv(f"{the_path}/papers_res_df_{treatment.replace(' ','_')}.csv",index=False)
            else:
                best_paper_ress = papers_res[papers_res['top_order']<=top_n
                                            ].nsmallest(top_n,'top_order')   
                if best_paper_ress.shape[0]:
                    top_text = [SentenceEval.model_validate_json(one_res).enhanced_ver \
                                for one_res in best_paper_ress.enhanced.values]
                    top_papers = best_paper_ress.id.values
                else:
                    top_text, top_papers = [],[]
            end = time.time()    
            logging.info(f'choosing top10 {end-start}')

            start = time.time()
            # ________ creating a paragraph eng
            chosen_top_df = top_df[top_df.treatment==treatment.replace(' ','_')]
            if (chosen_top_df.shape[0]) and (chosen_top_df['eng'].values[0]!=''):
                logging.info('reading treat summary')
                par_result_orig_id = chosen_top_df['eng'].values[0]
                add_summ = re.sub(r'\[\[(\d+)\]\]', k.replacement_match, 
                                  par_result_orig_id)
            else:
                logging.info('making treat summary')
                #par_result_orig_id = extract.combine_res(fin_condition, treatment, top_text, top_papers)
                par_result_orig_id = []
                for one_sentence, one_id in zip(top_text, top_papers):
                    if one_sentence[-1]=='.':
                        one_sentence = one_sentence[:-1]+f' [[{one_id}]].'
                    else:
                        one_sentence+=f' [[{one_id}]].'
                    par_result_orig_id.append(one_sentence)
                par_result_orig_id = ' '.join(par_result_orig_id)   
                
                add_to_top_df[treatment.replace(' ','_')] = [par_result_orig_id, '']
                add_summ = re.sub(r'\[\[(\d+)\]\]', k.replacement_match, 
                                  par_result_orig_id)
                
            end = time.time()    
            logging.info(f'creating paragraph {end-start}')
            #logging.info('add_summ '+add_summ)    
            eng_papers_summ = f"""\n### Chosen scientific papers:\n\n{add_summ}\n\n"""
            eng_text2 += eng_treat_ct+eng_papers_summ
            
            k.last_index=final_lit_idx
            k.index_start=final_lit_idx
            # ________ creating a paragraph rus
            # if there's eng info BUT no russian info
            if (chosen_top_df.shape[0]) and (chosen_top_df['rus'].values[0] != ''):
                logging.info('reading rus treat summary')
                ru_par_result_orig_id = chosen_top_df['rus'].values[0]
                ru_summ = re.sub(r'\[\[(\d+)\]\]', k.replacement_match,
                                 ru_par_result_orig_id)
                #logging.info(k.dict_index_paper)
                ru_papers_summ = f"""\n### Выбранные научные статьи:\n\n{ru_summ}\n\n"""
                logging.info('ru_treat_ct '+ru_treat_ct)
                if ru_treat_ct!=f"\n## Препарат {treat_idx+1}: {ru_treatment}\n\n":
                    '''
                    logging.info('translate')
                    ru_treat_ct_fin = translate_ru(ru_en_drugs, ru_treat_ct, 
                                               model_translate, fin_glossary)
                    '''
                    ru_treat_ct_fin = ru_treat_ct
                else:
                    ru_treat_ct_fin = ru_treat_ct
                treat_all_info_ru = ru_treat_ct_fin+ru_papers_summ
            else:
                logging.info('making rus treat summary')
                if par_result_orig_id!='':
                    ru_par_result_orig_id = translate_ru(ru_en_drugs, par_result_orig_id, 
                                                     model_translate, fin_glossary)
                else:
                    ru_par_result_orig_id = ''
                #logging.info('ru_par_result_orig_id '+ru_par_result_orig_id)
                add_to_top_df[treatment.replace(' ','_')] = [par_result_orig_id, 
                                                             ru_par_result_orig_id]
                ru_summ = re.sub(r'\[\[(\d+)\]\]', k.replacement_match,
                                 ru_par_result_orig_id)
                logging.info('ru_summ '+ru_summ)
                ru_papers_summ = f"""\n### Выбранные научные статьи:\n\n{ru_summ}\n\n"""
                if ru_treat_ct!=f"\n## Препарат {treat_idx+1}: {ru_treatment}\n\n":
                    '''
                    ru_treat_ct_fin = translate_ru(ru_en_drugs, ru_treat_ct, 
                                               model_translate, fin_glossary)
                    '''
                    ru_treat_ct_fin = ru_treat_ct
                else:
                    ru_treat_ct_fin = ru_treat_ct
                treat_all_info_ru = ru_treat_ct_fin+ru_papers_summ
                
            logging.info(f'K FIRST and LAST INDEX, {k.index_start}, {k.last_index}')        
            #logging.info('BEFORE\n'+eng_papers_summ)
            #logging.info('AFTER\n'+ru_papers_summ)
            ru_text2 += treat_all_info_ru
            
            logging.info(k.dict_index_paper)
            lit_list = '\n\n'.join([f"[{i}]  {k.dict_index_paper.get(i, '--')[0]} [{k.dict_index_paper.get(i, '--')[1]}]({k.dict_index_paper.get(i, '--')[1]})" for i in range(k.index_start+1,k.last_index+1)])+'\n\n'
    
            all_lit += lit_list
            logging.info('lit_list!:\n '+lit_list) 
            #logging.info('all_lit!:\n '+all_lit) 
            final_lit_idx = k.last_index
            
        except ValueError as e:
            logging.info('exception', e)
            pass

    # if new paragraphs were saved 
    if add_to_top_df:
        new_top_df_part = pd.DataFrame(add_to_top_df).T.reset_index()
        new_top_df_part.columns = ['treatment', 'eng','rus'] 
        logging.info(f'adding {new_top_df_part}')
        top_df = pd.concat([top_df, new_top_df_part])
        top_df.to_csv(top_summ_file, index=False)
        
    #logging.info('all_lit fin!:\n '+all_lit)    
    pdf.add_section(Section(eng_text2+"\n### Literature:\n\n"+all_lit), 
                    user_css=user_css)
    pdf.save(path_file+f"ENG_doc_{len(treatments_eng)}_{fin_condition.replace(' ','_')}.pdf")
    

    pdf = MarkdownPdf(toc_level=4, optimize=True)
    pdf.add_section(Section(ru_text2+"\n### Список источников:\n\n"+all_lit), 
                    user_css=user_css)
    pdf.save(path_file+f"RU_doc_{len(treatments_eng)}_{fin_condition.replace(' ','_')}.pdf")
    


def translate_ru(ru_en_drugs, eng_text2, model_translate, zero_glossary):
    logging.info('eng_text2'+ eng_text2)
    en_treats = ru_en_drugs.drug.values
    found_all = re.findall("|".join(en_treats), eng_text2, re.IGNORECASE)
    found_treats = list(set(
        [i.lower() for i in found_all]
    ))

    glossary = ru_en_drugs[ru_en_drugs.drug.str.lower().isin(found_treats)].values
    fin_glossary = "\n".join([f"ENG {pair[0]} -> RUS {pair[1].lower()}" for pair in glossary])
    fin_glossary += zero_glossary
    logging.info(fin_glossary)
    
    prompt = TRANSLATE_PROMPT
    logging.info(f"len of orig text: {len(eng_text2)}")
    logging.info(f"len of prompt with text: {len(prompt.format(fin_glossary=fin_glossary))}")

    messages=[{'role':'system',
               'content':prompt.format(fin_glossary=fin_glossary)
              },
              {'role':'user',
               'content':eng_text2
              },
             ]

    fint= extract.use_llm(model=model_translate, messages=messages)
    return fint