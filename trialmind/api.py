import json
import re
import tempfile
import os
import io
from typing import List, Dict, Union, Optional
from bs4 import BeautifulSoup
import pandas as pd
import requests
from .llm import (
    call_llm, 
    batch_call_llm,
    batch_function_call_llm
)
from .retrievers import (
    combine_blocks_text,
)
from .prompts.search_query import (
    PRIMARY_TERM_EXTRACTION, 
    SEARCH_TERM_EXTRACTION
)

from .prompts.extraction import (
    STUDY_RESULTS_FORMATTING,
    STUDY_FIELDS_EXTRACTION_3,
STUDY_FIELDS_EXTRACTION_3n,
    STUDY_RESULTS_STANDARDIZATION,
    RESULT_TABLE_TEMPLATE,
    )
from .prompts.screening import LITERATURE_SCREENING_FC,CT_SCREENING_FC
from .prompts.screen_criteria import SCREENING_CRITERIA_GENERATION, SCREENING_CRITERIA_CT_GENERATION
from .pubmed import ReqPubmedFull, ReqPubmedID
#from .sandbox import E2BSandbox
import httpx
from pydantic import BaseModel, validator, Field,field_validator, conlist  
from typing_extensions import Literal
from typing import Dict

from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams
from langchain_openai import OpenAIEmbeddings

#from pydantic_classes import Results
import logging
logger = logging.getLogger().setLevel(logging.INFO)
# Silence other loggers
for log_name, log_obj in logging.Logger.manager.loggerDict.items():
    if log_name != 'my_module_name':
        log_obj.disabled = True
        

def extract_json(input_text):
    # Pattern to match content between ```json and ```
    #json_pattern = r"```json([\s\S]*?)```"
    json_pattern = r"\`\`\`json([\s\S]*?)\`\`\`"
    match = re.search(json_pattern, input_text)

    if match:
        # Extract JSON content between ```json and ```
        return match.group(1)
    
    
    elif re.search(r"\`\`\`json([\s\S]*?)", input_text):
        input_text=input_text+'```'
        match = re.search(json_pattern, input_text)
        if match:
            # Extract JSON content between ```json and ```
            return match.group(1)
    else:
        # Pattern to match content between {{ and }}
        curly_pattern = r"\{\{([\s\S]*?)\}\}"
        match = re.search(curly_pattern, input_text)
        if match:
            output = match.group(1)
            return output
        else:
            # Attempt to parse the entire input as JSON
            try:
                json.loads(input_text)
                # If no exception is raised, the entire input is valid JSON
                return input_text
            except json.JSONDecodeError:
                # Input is not valid JSON
                return None


def parse_json_outputs(outputs: List[str]) -> List[Dict]:
    parsed_outputs = []
    for output in outputs:
        output = extract_json(output.strip('<think>\n\n</think>\n\n'))
        try:
            output = json.loads(output)
        except:
            
            output = {}
        parsed_outputs.append(output)
    return parsed_outputs

def extract_code(_raw_output: str) -> str:
    # Using 'html.parser' to parse the content
    soup = BeautifulSoup(_raw_output, "html.parser")
    try:
        _raw_output = soup.find("code").text
    except:
        pass
    if "```python:" in _raw_output:
        pattern = r"```python\n{(.*?)}\n```"
        match = re.search(pattern, _raw_output, re.DOTALL)
        if match:
            return match.group(1)
        else:
            return _raw_output
    else:
        return _raw_output
        

    
class StudyCharacteristicsExtraction:
    """
    Extract the structured data from a clinical study, based on user's request if provided.

    Args:
        papers (list[str or list[str]]): A list of clinical study papers' raw content in text or in a list of string.
            If the input is a list of string, each string is a text block from the paper.
            If the input is a single string, the API will try to cut the text into blocks,
            which is for generating the citations from the paper.
        fields (List[str]): The fields to extract from the clinical study papers. 
           Each element is natural language description of which information this field is about.
           It is suggested to be in the format of "[Field Name], [Data Type], [Description]".
        llm (str): The language model to use for the extraction. Default is "gpt-4".
        batch_size (int): The batch size for the batch call. Default is 5.
            Too large batch size may cause the request failed.
        semantic_filtering (bool): Whether to use semantic ranking to only keep the most relevant blocks
            so to reduce the number of blocks to be processed. Default is False.
        semantic_filtering_top_k (int): The top k blocks to keep after the semantic filtering. Default is 20.
    """
    DEFAULT_FIELDS = [
        "Study Name, string, the study's alias, usually be in the format of FirstAuthorYear",
        "Study Type, string, if the study is randomized controlled trial, observational study, or others",
        "Study Year, date, the study's year",
        "Location: which countries the study was conducted in",
        "Phase, string: in which phase this clinical trial is in, e.g., phase 1, phase 2, phase 3, or phase 4",
        "Conditions, list of string, the conditions or diseases the study is investigating",
        "Treatments, list of string, the primary treatment or intervention used in the study",
        "Comparison, list of string, the comparison treatment or intervention used in the study",
        "Num Patients, int, how many participants are in the study",
        "Mean Age, continuous, the average age of the participants",
        "Age Range, string, the age range of the participants",
        ]
    def __init__(self):
        pass
    
    
    def search(self, query, doc_id, n_to_retrieve=20, with_rerank=True):
        if 'hybrid' in self.vector_store.collection_name:
            filter_q = models.Filter(must=[models.FieldCondition(
                                        key="metadata.source",
                                        match=models.MatchValue(value=str(doc_id)),
                                    ),
                                ]
                            )

            dense_vectors = self.emb.embed_query(query)
            sparse_vectors = self.sparse_embeddings.embed_query(query)
            prefetch = [
                    models.Prefetch(
                        query=dense_vectors,
                        using="dense",
                        limit=round(n_to_retrieve),
                        filter=filter_q
                    ),
                    models.Prefetch(
                        query=models.SparseVector(**sparse_vectors.model_dump()),
                        using="sparse",
                        limit=round(n_to_retrieve),
                        filter=filter_q
                    ),
            ]
            # here vector_store == client
            results = self.vector_store.query_points(
                     self.vector_store.collection_name,
                    prefetch=prefetch,
                    query=models.FusionQuery(fusion=models.Fusion.RRF  ),
                    with_payload=True,
                    limit=n_to_retrieve,
            )
            retrieved_docs = [i.payload for i in results.points]

            
        else:
            retrieved_docs = self.vector_store.similarity_search(query, k=n_to_retrieve,
                                                             filter={"source": doc_id})
        return retrieved_docs
        
    def rerank(self, query, retrieved_docs):
        url = "http://localhost:8080/v1/rerank"
        if 'hybrid' in self.vector_store.collection_name:
            docs2 = [i['page_content'] for i in retrieved_docs]
        else:
            docs2 = [i.page_content for i in retrieved_docs]
        payload = {
            "model": 'reranker-model',
            "query": query,
            "top_n": 10,
            "documents": docs2,
            "return_documents": False
        }

        #logging.info(f"number of letters in retr.30 docs: {len(''.join(docs2))}") # ~ 6400
        
        response = requests.post(url, json=payload)
        try:
            contents = [docs2[i['index']] for i in response.json()['results']]
        except Exception as e:   
            print('exception: ', e)
            contents = []
        #print(contents)
        return contents
        
    def run(self,
        papers_inp: list[Union[str,list[str]]],
        fields: list[str]=[],
        llm: str="gpt-4",
        batch_size: int = None,
        semantic_filtering: bool = False,
        semantic_filtering_top_k: int = 20,
        chunk_size: int = 1000,
        chunk_overlap: int = 20,
        thinking: bool = False,
        vector_store = '',
        ):
        
        self.vector_store = vector_store
        # get the fields
        fields = fields if len(fields) > 0 else self.DEFAULT_FIELDS
        fields_info = '\n'.join([f"<field id={idx+1}>\"{field}\"</field>" for idx, field in enumerate(fields)])
        #print(fields_info)
        pmid_list, papers = papers_inp
        #print(pmid_list)
        
        class FieldResult(BaseModel):
            name: str = Field( description='Field name that accurately represents the content of the field based on its description; UNDER 100 characters',
             max_length=100)
            value: str = Field(description='Extracted information from the text based on the field description; UNDER 200 characters',
                              max_length=200)
            source_id: conlist(int, min_length=0, max_length=3) = Field(description='Cited document IDs; MAX 3 items.')
            
            @field_validator('source_id', mode='before')
            @classmethod
            def truncate(cls, v):
                return v[:3]
                
            @field_validator('name', mode='before')
            @classmethod
            def truncate_n(cls, v):
                return v[:100]
                
            @field_validator('value', mode='before')
            @classmethod
            def truncate_v(cls, v):
                return v[:200]
                
        class Results(BaseModel):
            fieldresult: list[FieldResult] = Field(min_length=1, max_length=1)
        
        
        if 'hybrid' in self.vector_store.collection_name:
            self.sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    
            self.emb = OpenAIEmbeddings(model='Qwen/Qwen3-Embedding-4B-GGUF',
                                      base_url='http://localhost:8080/v1/',
                                    api_key=os.getenv("OPENAI_API_KEY"),
                                   check_embedding_ctx_length =False,
                                   #encoding_format ='float',
                                    http_client=httpx.Client(verify=False)
                                  )
        
        all_outputs = []
        dict_splitted = {}
        skip_papers = []
        dict_papers = {}
        
        for i, one_field in enumerate(fields):
            analyzed_papers = [x for x in pmid_list if x not in skip_papers]
            logging.info(f"field {one_field.split('|')[-1]}")
            batch_inputs = []
            unique_splited_docs = []
        
            all_found_parts_e=[]
            dict_parts = {}
        
            # separate "for" loops to lower the number of switching between models 
            # on llama-server 
            for paper_id in analyzed_papers: 
                #print('search',paper_id)
                # embedding
                found_parts_e = self.search(one_field.split('|')[-1], paper_id)
                if paper_id in dict_parts:
                    dict_parts[paper_id].append([one_field,found_parts_e])
                else:
                    dict_parts[paper_id] = [[one_field,found_parts_e]]
                all_found_parts_e.append(found_parts_e)
                    
            for paper_id in analyzed_papers:   
                #print('rerank',paper_id)
                # [[field1,field_res],[field2,field_res],...]
                all_fields_parts = dict_parts[paper_id]
                # choosing [field_one,field_res]
                prev_found_part = [f_parts[1] for f_parts in all_fields_parts \
                                   if f_parts[0]==one_field][0]
                # reranking
                found_parts = self.rerank(one_field.split('|')[-1], prev_found_part)
                combined = combine_blocks_text(found_parts)
                #print(len(combined))
                if paper_id in dict_splitted:
                    dict_splitted[paper_id].append([one_field,found_parts])
                else:
                    dict_splitted[paper_id] = [[one_field,found_parts]]
                batch_inputs.append({
                    "paper_content": combined,
                    "fields": f"<field id={i+1}>\"{one_field}\"</field>"
                })
            #print(STUDY_FIELDS_EXTRACTION_3)
            outputs = batch_function_call_llm(STUDY_FIELDS_EXTRACTION_3, 
                                              batch_inputs, 
                                     [Results],
                                     llm=llm, batch_size=batch_size)

            
            for paper_id, output in zip(analyzed_papers, outputs):
                if (one_field==fields[0]) and (output.fieldresult[0].value=='NP'):
                    logging.info(f"bad output {output}; removing paper {paper_id}")
                    skip_papers.append(paper_id)
                else:
                    if paper_id in dict_papers:
                        dict_papers[paper_id].append([one_field,output])
                    else:
                        dict_papers[paper_id] = [[one_field,output]]
                        
            #print(skip_papers)
            all_outputs.append(outputs)
        
        analyzed_papers = [x for x in pmid_list if x not in skip_papers]

        #all_outputs = []
        for paper_id in analyzed_papers:
            # [[field1,output],[field2,output],...]
            field_outputs = dict_papers[paper_id]
            # [[field1,chunks],[field2,chunks],...]
            unique_splited_docs = dict_splitted[paper_id]
            
            for i, output in enumerate(field_outputs):
                # -1, since unique_splited_docs[i] is [field_name,field_chunks]
                blocks = unique_splited_docs[i][-1]
                # -1, since output is [field_name,field_output]
                for field_output in output[-1].fieldresult:
                    src_ids = field_output.source_id
                    cited = []
                    #logging.info(f'{len(blocks), src_ids}')
                    for src_id in src_ids:
                        src_id = int(src_id) # we have id as str!
                        if src_id < len(blocks):
                            cited.append(blocks[src_id])
                    field_output._cited_blocks = cited
                #all_outputs.append(field_outputs)
        
        return dict_papers#all_outputs


class LiteratureScreening:
    """Pass the papers through the screening criteria to determine if they are relevant to the research question.
    The input contains a list of criteria for screening the papers, and the papers to be screened.

    Args:
        papers: A list of clinical study papers' raw content in text, to be screened.
        criteria: A list of screening criteria for the papers.
        llm: The language model to use for the screening. Default is "gpt-4".
    """
    def __init__(self):
        pass

    def run(self,
        population: str,
        intervention: str,
        comparator: str,
        outcome: str,
        papers: list[str],
        criteria: list[str],
        llm: str="gpt-4",
        batch_size: int = None,
        ):

        # build the criteria text with index
        criteria_text = [f"{idx+1}. {c}" for idx, c in enumerate(criteria)]
        n_criteria = len(criteria_text)

        # build batch inputs
        batch_inputs = []
        for paper in papers:
            batch_inputs.append({
                "P": population,
                "I": intervention,
                "C": comparator,
                "O": outcome,
                "paper_content": paper,
                "criteria_text": criteria_text,
                "num_criteria": n_criteria
            })

        # call llm
        #from langchain.pydantic_v1 import BaseModel, validator, Field, conlist
        
        class PaperEvaluation(BaseModel):
            evaluations: conlist(Literal['YES', 'NO', 'UNCERTAIN'], min_length=n_criteria, max_length=n_criteria) = Field(description=f"Evaluations for {n_criteria} criteria")
            rationale: conlist(str,min_length=n_criteria, max_length=n_criteria) = Field(description="A rationale for each criteria evaluation")
        #outputs = batch_function_call_llm(LITERATURE_SCREENING_FC, batch_inputs, [PaperEvaluation.model_json_schema()], llm=llm, batch_size=batch_size)    
        
        outputs = batch_function_call_llm(LITERATURE_SCREENING_FC, batch_inputs, [PaperEvaluation], llm=llm, batch_size=batch_size)
        #print('\nOUTPUTS: ', outputs)
        #outputs = parse_json_outputs(outputs)
        #print('\nOUTPUTS: ', outputs)
        # try to fix the predictions if not met the output format
        #parsed_outputs = self._check_outputs(outputs, n_criteria)
        return outputs
    
    def _check_outputs(self, outputs, n_criteria):
        # check if the outputs are in the correct format
        parsed_outputs = []
        for output in outputs:
            try:
                evaluations = output.get("evaluations", [])
                if len(evaluations) != n_criteria:
                    evaluations = ["UNCERTAIN"] * n_criteria
                else:
                    evaluations = [e.upper() for e in evaluations]
                    evaluations = [e if e in ["YES", "NO", "UNCERTAIN"] else "UNCERTAIN" for e in evaluations]
            except:
                evaluations = ["UNCERTAIN"] * n_criteria
            parsed_outputs.append(evaluations)
        return parsed_outputs

    
class CTScreening:
    """Pass the papers through the screening criteria to determine if they are relevant to the research question.
    The input contains a list of criteria for screening the papers, and the papers to be screened.

    Args:
        papers: A list of clinical study papers' raw content in text, to be screened.
        criteria: A list of screening criteria for the papers.
        llm: The language model to use for the screening. Default is "gpt-4".
    """
    def __init__(self):
        pass

    def run(self,
        papers: list[str],
        criteria: list[str],
        llm: str="gpt-4",
        batch_size: int = None,
        ):

        # build the criteria text with index
        criteria_text = [f"{idx+1}. {c}" for idx, c in enumerate(criteria)]
        n_criteria = len(criteria_text)

        # build batch inputs
        batch_inputs = []
        for paper in papers:
            batch_inputs.append({
                "paper_content": paper,
                "criteria_text": criteria_text,
                "num_criteria": n_criteria
            })

        # call llm
        #from langchain.pydantic_v1 import BaseModel, validator, Field, conlist
        class PaperEvaluation(BaseModel):
            evaluations: conlist(Literal['YES', 'NO', 'UNCERTAIN'], min_length=n_criteria, max_length=n_criteria) = Field(description=f"Evaluations for {n_criteria} criteria")
            rationale: conlist(str,min_length=n_criteria, max_length=n_criteria) = Field(description="A rationale for each criteria evaluation") 
            
        #print(CT_SCREENING_FC)    
        outputs = batch_function_call_llm(CT_SCREENING_FC, batch_inputs, 
                                          [PaperEvaluation], llm=llm, batch_size=batch_size)
        #print('\nOUTPUTS: ', outputs)
        #outputs = parse_json_outputs(outputs)
        #print('\nOUTPUTS after json extr: ', outputs)
        # try to fix the predictions if not met the output format
        #parsed_outputs = self._check_outputs(outputs, n_criteria)
        return outputs
    
    def _check_outputs(self, outputs, n_criteria):
        # check if the outputs are in the correct format
        parsed_outputs = []
        for output in outputs:
            try:
                evaluations = output.get("evaluations", [])
                if len(evaluations) != n_criteria:
                    evaluations = ["UNCERTAIN"] * n_criteria
                else:
                    evaluations = [e.upper() for e in evaluations]
                    evaluations = [e if e in ["YES", "NO", "UNCERTAIN"] else "UNCERTAIN" for e in evaluations]
            except:
                evaluations = ["UNCERTAIN"] * n_criteria
            parsed_outputs.append(evaluations)
        return parsed_outputs
    

class StudyResultStandardization:
    """Given the raw extracted results, standardize the extracted results into a structured format.

    Args:
        population: The population of the research question.
        intervention: The intervention of the research question.
        comparator: The comparator of the research question.
        outcome: The target outcome measurement.
        data_type: The data type of the outcome measurement. It can be "binary", "continuous", "o-minus-e", or "generic".
        results: A list of extracted results from the papers.
        sandbox_id: The ID of the sandbox to connect to. If not found, this API will not be able to execute the code.
        llm: The language model to use for the standardization. Default is "gpt-4".
    """
    # step 1: detect the variables from the input
    # step 2: write the python code to create the target table
    # step 3: write some examples to help LLMs understand what to do here.
    def __init__(self):
        pass

    def run(self,
        population: str,
        intervention: str,
        comparator: str,
        outcome: str,
        data_type: str, # ["binary", "continuous", "o-minus-e", "generic"],
        results: list[str],
        sandbox_id: Optional[str]=None,
        llm: str="gpt-4"
        ):
        # connect to the E2B sandbox with the given sandbox id
        if sandbox_id is not None and len(sandbox_id) > 0:
            self.sandbox = E2BSandbox(sandbox_id=sandbox_id)
        else:
            logger.warning("No sandbox id is provided. The API will not be able to execute the code!")
            self.sandbox = None

        # get the initial table
        outputs = self._run_initial_table_extraction(population, intervention, comparator, outcome, results, llm=llm)

        # run the standard table extraction
        output_code = self._run_standard_table_extraction_code_gen(
            population, intervention, comparator, outcome, outputs, data_type, llm
        )

        # run the generated python code to get the standard table results
        output_data = {}
        if self.sandbox is not None:
            output_data = self._execute_code_to_get_standard_table(
                outputs, 
                output_code
                )

        # build the final outputs
        # each output has: result, code, data
        # could be all none, or partial none
        final_outputs = []
        for index, extracted_ in enumerate(outputs):
            code = output_code.get(index, None)
            data = output_data.get(index, None)
            final_outputs.append({
                "raw_data": extracted_,
                "code": code,
                "standardized_data": data
            })
        return final_outputs

    def _run_standard_table_extraction_code_gen(self,
        population: str,
        intervention: str,
        comparator: str,
        outcome: str,
        results: list[dict],
        data_type: str,
        llm: str="gpt-4"
        ):
        data_structure = RESULT_TABLE_TEMPLATE.get(data_type, None)
        if data_structure is None:
            raise ValueError(f"data_type {data_type} is not supported.")
        else:
            target_output = data_structure["table"]
            target_desc = data_structure["desc"]

        # build batch inputs
        batch_inputs = []
        batch_input_indices = []
        for i, result in enumerate(results):
            if result is not None:
                batch_input_indices.append(i)
                # formulate the result better
                result_txt = self._build_result_text(result)
                batch_inputs.append(
                {
                    "population": population,
                    "intervention": intervention,
                    "comparator": comparator,
                    "outcome": outcome,
                    "raw_data": result_txt,
                    "target_output": target_output,
                    "desc": target_desc
                }
        )
        if len(batch_inputs) > 0:
            outputs = batch_call_llm(STUDY_RESULTS_FORMATTING, batch_inputs, llm=llm)
            # parse the outputs to extract the python code
            output_codes = []
            for output_code in outputs:
                try:
                    output_code = extract_code(output_code)
                except:
                    output_code = output_code
                output_codes.append(output_code)
            
            return {i: o for i, o in zip(batch_input_indices, output_codes)}
        else:
            return {}
        
    def _run_initial_table_extraction(self,
        population: str,
        intervention: str,
        comparator: str,
        outcome: str,
        results: list[str],
        llm: str="gpt-4"
        ):
        batch_inputs = []
        for result in results:
            batch_inputs.append({
                "population": population,
                "intervention": intervention,
                "comparator": comparator,
                "outcome": outcome,
                "results": result
            })
        outputs = batch_call_llm(STUDY_RESULTS_STANDARDIZATION, batch_inputs, llm=llm)
        
        # parse outputs
        outputs = parse_json_outputs(outputs)
        return outputs
    
    def _build_result_text(self, result):
        try:
            values = []
            for r in result:
                values_ = []
                for k, v in r.items():
                    values_.append(v)
                values.append(values_)
            columns = list(result[0].keys())
            df = pd.DataFrame(values, columns=columns)
            result_text = df.to_markdown()
            return result_text
            
        except:
            # if failed, return the original result
            return result
        
    def _execute_code_to_get_standard_table(self,
        results,
        codes,
        ):
        def _upload_to_sandbox(df):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmpfile:
                df.to_csv(tmpfile.name, index=False)
                # upload the dataframe to the sandbox
                remote_path = self.sandbox.upload_file(tmpfile.name)
            os.remove(tmpfile.name)
            return remote_path
        
        # execute the code to get the standard table
        output_data = {}
        for index, result in enumerate(results):
            code = codes.get(index, {})
            output_filename = "result_table_{}.csv".format(index)
            save_output_code = f"df.to_csv('{output_filename}', index=False)"
            if code is None:
                continue
            
            # parse the result
            try:
                columns = list(result[0].keys())
                values = []
                for r in result: values.append(r.values())
                df = pd.DataFrame(values, columns=columns)
                filepath = _upload_to_sandbox(df)
                code = f"""import pandas as pd\ndf = pd.read_csv('{filepath}')\n{code}\n{save_output_code}"""
                stdout, stderr, artifacts = self.sandbox.run_python(code)

                # try to download the result from the artifacts
                for artifact in artifacts:
                    if artifact.file_name == output_filename:
                        # load the csv file from bytes content
                        csv_file_like_object = io.BytesIO(artifact.content)
                        data = pd.read_csv(csv_file_like_object)
                        output_data[index] = data.to_dict(orient='records')
                        break
                        
            except:
                continue
                
        # return the output data
        return output_data