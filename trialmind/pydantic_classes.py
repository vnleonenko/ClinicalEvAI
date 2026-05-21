from pydantic import BaseModel, validator, Field, field_validator, conlist  
from typing_extensions import Literal
from typing import Dict

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


class GroupOutcomes(BaseModel):
    reasoning: str = Field(description='Explain: where and why each value was chosen, how each percentage was calculated?')

    group_name: str = Field(description='Description of the analyzed group based on a measure title field, a population description field and a group title field; UNDER 100 characters' ,max_length=100)
    population_size: int = Field(description='Population size of the analyzed group; number of participants; value GREATER or EQUAL TO 0', ge=0)
    
    complete_response: float = Field(description='Complete Response (CR); percentage of participants; value UNDER or EQUAL TO 100.0', le=100.0)
    partial_response: float = Field(description='Partial Response (PR); percentage of participants; value UNDER or EQUAL TO 100.0', le=100.0)
    objective_response_rate: float = Field(description='Objective Response Rate (ORR); ORR = CR + PR; percentage of participants; value UNDER or EQUAL TO 100.0', le=100.0)
    
    stable_disease: float = Field(description='Stable Disease (SD); percentage of participants; value UNDER or EQUAL TO 100.0', le=100.0)
    disease_control_rate: float = Field(description='Disease Control Rate (DCR); DCR = CR + PR + SD; percentage of participants; value UNDER or EQUAL TO 100.0', le=100.0)
    
    progression_free_survival: float = Field(description='Progression Free Survival (PFS); number of months')
    overall_survival: float = Field(description='Overall Survival (OS); number of months')

    @field_validator('complete_response','partial_response','objective_response_rate',
                     'stable_disease','disease_control_rate','progression_free_survival',
                     'overall_survival', mode='before')
    @classmethod
    def truncate_f(cls, v):
        return round(min(v,100.0),2)


class ClinicalTrialOutcomes(BaseModel):
    main_reasoning: str = Field(description='Explain: what was included/excluded and why? List final chosen groups')
    groups_outcomes: list[GroupOutcomes] = Field(description='Extracted measurements for each chosen group.')




    
