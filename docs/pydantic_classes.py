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


from pydantic import BaseModel, validator, Field, conlist  # This is the new version
    class PaperEvaluation(BaseModel):
        evaluations: conlist(Literal['YES', 'NO', 'UNCERTAIN'], min_length=n_criteria, max_length=n_criteria) = Field(description=f"Evaluations for {n_criteria} criteria")
        rationale: conlist(str,min_length=n_criteria, max_length=n_criteria) = Field(description="A rationale for each criteria evaluation")