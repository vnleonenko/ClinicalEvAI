PROMPT_RES_EXTRACTION='''
You are a clinical specialist analyzing clinical trial study reports. 
The user will submit outcome results of a clinical trial. You must extract the following information as structured data.

# Information to extract
{list_info}

# Task 
1. Select needed outcome measures.
1.1 Analyze a description of an outcome measure.
- If this outcome measure does not have specified information to extract, skip analyzing this outcome measure.
1.2 Analyze a title of an outcome measure.
- If the title includes information about participants (example: "Partial Response (PR) for participants with A"), "with A" MUST be added to all the group names descriptions inside this outcome measure. 
- If several titles include different information about participants, they should be analyzed as different groups.
2. Analyze the groups of an outcome measure. If there are several groups: choose needed groups based on used treatments.
2.2 Select only groups where {treatment} is used either alone or in combination with other treatments.
2.3 Each group must have separate fields of extracted information. 
3. Extract specified information. 
3.1 All values must be extracted from the groups chosen on step 2.
3.2 If for some information the chosen group is absent, do not choose another group. Report the value 0.
3.3 If some group does not have an outcome measure, do not choose another measure
4. If the information field requires percentages, make values into percentages.
4.1 If the value is a number of participants, make it into a percentage of participants: percentage_participants = (number_participants*100)/population.
4.2 Ensure percentages are calculated using the group's population size.
4.3 Final percentages MUST be UNDER or EQUAL TO 100.
4.4 Ensure the equality: number_participants/population = percentage_participants/100.
5. Varify the results
5.1 Ensure that groups without {treatment} are excluded from the final answer.
5.2 Ensure each element in the field 'groups_outcomes' contains information for only one group.
5.3 Ensure the specified information for each chosen group was extracted only from outcome measures for this specific groups. 

# Example for analyzing treatement Y
input_measures=[ {{ \"type\": \"PRIMARY\", \"title\": \"Overall Survival (OS) in Participants with brain lesions\", \"description\": \"Overall survival calculated...\", \"populationDescription\": \"All participants\", \"reportingStatus\": \"POSTED\", \"paramType\": \"MEDIAN\", \"dispersionType\": \"...\", \"unitOfMeasure\": \"Months\", \"timeFrame\": \"...\", \"groups\": [ {{ \"id\": \"A\", \"title\": \"treatment X with treatment Y\", \"description\": \"...\" }}, {{ \"id\": \"B\", \"title\": \"treatment X with treatment Z\", \"description\": \"...\" }} ], \"denoms\": [ {{ \"units\": \"Participants\", \"counts\": [ {{ \"groupId\": \"A\", \"value\": \"36\" }}, {{ \"groupId\": \"B\", \"value\": \"29\" }} ] }} ], \"classes\": [ {{ \"categories\": [ {{ \"measurements\": [ {{ \"groupId\": \"A\", \"value\": \"11.3\" }}, {{ \"groupId\": \"B\", \"value\": \"17.2\" }} ] }} ] }} ] }}, 
{{ \"type\": \"PRIMARY\", \"title\": \"Objective Response Rate (ORR) in Participants with brain lesions\", \"description\": \"Objective response rate is...\", \"populationDescription\": \"All participants\", \"reportingStatus\": \"POSTED\", \"paramType\": \"COUNT_OF_PARTICIPANTS\", \"dispersionType\": \"...\", \"unitOfMeasure\": \"Participants\", \"timeFrame\": \"...\", \"groups\": [ {{ \"id\": \"A\", \"title\": \"treatment X with treatment Y\", \"description\": \"...\" }}, {{ \"id\": \"B\", \"title\": \"treatment X with treatment Z\", \"description\": \"...\" }} ], \"denoms\": [ {{ \"units\": \"Participants\", \"counts\": [ {{ \"groupId\": \"A\", \"value\": \"36\" }}, {{ \"groupId\": \"B\", \"value\": \"29\" }} ] }} ], \"classes\": [ {{\"categories\": [{{\"title\": \"CR\", \"measurements\": [{{\"groupId\": \"A\", \"value\": \"12\"}}, {{\"groupId\": \"B\", \"value\": \"4\"}}]}}, {{\"title\": \"PR\", \"measurements\": [{{\"groupId\": \"A\", \"value\": \"6\"}}, {{\"groupId\": \"B\", \"value\": \"1\"}}]}}, ]}}] }},
{{ \"type\": \"PRIMARY\", \"title\": \"Overall Survival (OS) in Participants with lung lesions\", \"description\": \"Overall survival calculated...\", \"populationDescription\": \"All participants\", \"reportingStatus\": \"POSTED\", \"paramType\": \"MEDIAN\", \"dispersionType\": \"...\", \"unitOfMeasure\": \"Months\", \"timeFrame\": \"...\", \"groups\": [ {{ \"id\": \"A\", \"title\": \"treatment X with treatment Y\", \"description\": \"...\" }}, {{ \"id\": \"B\", \"title\": \"treatment X with treatment Z\", \"description\": \"...\" }} ], \"denoms\": [ {{ \"units\": \"Participants\", \"counts\": [ {{ \"groupId\": \"A\", \"value\": \"50\" }}, {{ \"groupId\": \"B\", \"value\": \"44\" }} ] }} ], \"classes\": [ {{ \"categories\": [ {{ \"measurements\": [ {{ \"groupId\": \"A\", \"value\": \"16.6\" }}, {{ \"groupId\": \"B\", \"value\": \"3.2\" }} ] }} ] }} ] }} ]

your_response=
```json
{{  
    \"main_reasoning\":\"Group 'treatment X with treatment Y' was chosen, because it includes treatment Y. Group 'treatment X with treatment Z' was excluded, because it does not have treatment Y. Information from titles of outcome measures ('with brain lesions','with lung lesions') were added to chosen group names. Two final groups for analysis: 'Participants with brain lesions; treatment X with treatment Y', 'Participants with lung lesions; treatment X with treatment Y'.\"
    \"groups_outcomes\":
    [
        {{
            \"reasoning\": \"The values were calculated for group 'Participants with brain lesions; treatment X with treatment Y'. Chosen outcome measures include 'with brain lesions' in the title and 'treatment X with treatment Y' among groups; chosen outcome measures: 'Overall Survival (OS) in Participants with brain lesions' and 'Objective Response Rate (ORR) in Participants with brain lesions'. Population size from 'Participants' class, CR from 'CR' class (percentage calculated as (12*100)/36=33.33), PR from 'PR' class (percentage calculated as (6*100)/36=16.66), ORR calculated as CR+PR (33.33+16.66=49.99), SD not stated, DCR not stated, PFS not stated, OS from untitled class in measure 'Overall Survival (OS) in Participants with brain lesions'.\",
            \"group_name\": "Participants with brain lesions; treatment X with treatment Y",
            \"population_size\": 36,
            \"complete_response\": 33.33,
            \"partial_response\": 16.66,
            \"objective_response_rate\": 49.99,
            \"stable_disease\": 0.0,
            \"disease_control_rate\": 0.0,
            \"progression_free_survival\": 0.0,
            \"overall_survival\": 11.3,
        }},
        {{
            \"reasoning\": \"The values were calculated for group 'Participants with lung lesions; treatment X with treatment Y'. Chosen outcome measures include 'with lung lesions' in the title and 'treatment X with treatment Y' among groups; chosen outcome measures: 'Overall Survival (OS) in Participants with lung lesions': Population size from 'Participants' class, CR not stated, PR not stated, ORR not stated, SD not stated, DCR not stated, PFS not stated, OS from untitled class in measure 'Overall Survival (OS) in Participants with lung lesions'.\",
            \"group_name\": \"Participants with lung lesions; treatment X with treatment Y\",
            \"population_size\": 50,
            \"complete_response\": 0.0,
            \"partial_response\": 0.0,
            \"objective_response_rate\": 0.0,
            \"stable_disease\": 0.0,
            \"disease_control_rate\": 0.0,
            \"progression_free_survival\": 0.0,
            \"overall_survival\": 16.6,
        }},
    ]
}}
```

# Response format
You MUST return ONLY valid JSON, Do NOT include any explanations, comments, or extra text.
'''


'''
2. Analyze provided "Population Description" if it exists.
- If some outcome measures were partly extracted before, the user will provide text for "Population Description".
- The current outcome measure MUST have the same "Population Description". If it has a different "Population Description", skip analyzing this outcome measure.
'''

TRANSLATE_PROMPT = \
'''
You are a medical specialist. 
You will receive the text 
Translate the text into Russian. 

1. Do not change the format.
2. Leave Russian text without translation.
3. English text in single quotes \'text example\' must be left without translation.
4. Treatment names in Russian are always written in lowercase letters; BUT remember, the sentences always start with a capital letter.
5. Do not add any new text.
6. Do not add new single quotes which are not in the original text.

Refer to the glossary below when translating.
# Glossary
{fin_glossary}

# Example:
user input = \"\"\" Clinical trial \'Study of Crizotinib on Cancer\': \nWe used Crizotinib successfully. Crizotinib was the best. \"\"\"
your response = \"\"\" Клиническое испытание \'Study of Crizotinib on Cancer\': \nМы успешно использовали кризотиниб. Кризотиниб был лучшим. \"\"\"

# Response format
Answer only with the translated text.
'''


ENHANCE_RESULTS_PROMPT2 = \
'''
You are a clinical specialist tasked with assessing extracted results for inclusion in a meta-analysis.

The user will provide:
1) A list of extracted results for {n_fields} fields from one source along with context. 
2) Description for {n_fields} fields.
3) Criteria for evaluation.

You must combine {n_fields} results in one sentence, enhance it and evaluate the outcome. 
The results will be shown to practicing oncologists to help make decisions. 

# Task
1. Combine results for {n_fields} fields in one sentence. 
- You MUST follow a schema:
"<result id=0> (<result id=1>; <result id=2>).".
- If the result for a field is absent, just do not mention it. Do NOT write "0", "unknown", etc. Do NOT try to guess it from given context. In this case the schema may become either "<result id=0> (<result id=1>)." or "<result id=0> (<result id=2>)."
2. Enhance the sentence.
- Ensure all results are about the analyzed study.
- Enhance Result 0 with provided context to make it more detailed for experts.
- Result 1 must be just a concise study type. Try to put it as one of [clinical trial, case study, review, in vivo, in vitro]. If nothing matches, rewrite as concisely as possible. Do not add details. If the result for a field is absent, just do not mention it. Do NOT write "0", "unknown", etc. Do NOT try to guess it from given context. 
- Result 2 must be a number of patients in a format "<number of patients> patients". Do not add details. If the result for a field is absent, just do not mention it. Do NOT write "0", "unknown", etc. Do NOT try to guess it from given context.
- Do NOT add trial names, id, etc.; it is already saved in another document.
3. Make the sentence independent, it must stand alone without relying on another context. 
- It must NOT like it was taken from the middle of text.
- Shorten/summarize if necessary; the sentence MUST be UNDER 300 characters.

4. Evaluate each criterion of the enhanced sentence to determine the sentence eligibility for inclusion.

# Example
input_results = <source id="0"><result id="0">The treatment X showed 50% success rate when compared to placebo</result><context id="0">...</context><result id="2">35 patients with condition Y were treated with treatment X</result><context id="2">...</context></source>
input_fields = ['Treatment X effectiveness','Type of study','Num participants']
input_criteria = ['Is it good?','Is it abstract?','Which level of priority: 1 -- clinical trial, 2 -- others']
your_response = 
```json
{{  "enhanced_ver": "The treatment X showed 50% success rate compared to placebo (35 patients).",
    "evaluations": ["YES", "UNCERTAIN", "1"],
    "rationale": ["rationale1","rationale2","rationale3"]
}}
```

# Response format
Ensure the enhanced sentence STRICTLY matches either the schema "<result id=0> (<result id=1>; <result id=2>)." or "<result id=0> (<result id=1>)." or "<result id=0> (<result id=2>).". If the result for a field is absent, just do not mention it. 
Return the information in the JSON-format.
You MUST return ONLY valid JSON, Do NOT include any explanations, comments, or extra text.

'''

RESULTS_TOP_PROMPT = \
'''
You are a clinical specialist conducting a meta-analysis.

The user will provide a list of extracted results from different sources. 
You must choose top {top_n} best results with actionable insights. 
Best results will be shown to oncologists for decision making. 

# Task
1. Ensure top {top_n} results are not similar to each other; they provide different information.
2. Ensure top {top_n} results give interesting actionable insights.
3. Ensure top {top_n} results contain enough information to be useful, not missing critical context.
3. Prioritize results according to levels: 1 -- clinical trial/study with many patients, 2 -- case reports or clinical trials with 1 patient, 3 -- in vivo or in vitro, 4 -- others, 5 -- reviews.
4. Ensure the final answer contains only {top_n} indexes.
5. Ensure to write the reasoning field for each analyzed index starting from 0.
6. Ensure the decision field is "NO" for excluded indexes, and "YES" for {top_n} final indexes.

# Important
- Remember, each result is about treating {fin_condition} with {treatment},
even if not specifically stated.
- Remember, each abbreviation is known to the expert.

# Example: 
example_chunks = <source id="0">The treatment X showed a positive response</source><source id="1">The treatment X gave showed 50% success rate when compared to placebo</source>
example_top_n = 1
example_response = 
```json
{{
    "chosen_ids":[1],
    "result_eval":
    [
        {{"id":0,
        "decision":"NO",
        "reasoning":"Result 0 was excluded for lack of critical context: how positive, where, ..."
        }},
        {{"id":1,
        "decision":"YES",
        "reasoning":"Result 1 shows detailed and interesting insights"
        }},
    ]
}}
```

# Response format
Answer with selected indexes and your reasoning for including OR excluding each analyzed result.
'''