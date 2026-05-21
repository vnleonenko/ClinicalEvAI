criteria_text = ['Is the result about {treatment} effectiveness on {fin_condition}?',
                'Is the sentence about {treatment} effectiveness, not general information?',
                'Does the sentence contain enough information to be useful, not missing critical context?',
                'Does the sentence offer interesting actionable insights?',
                'What is the level of priority according to levels: 1 -- clinical trial/study with many patients, 2 -- case reports or clinical trials with 1 patient, 3 -- in vivo or in vitro, 4 -- others, 5 -- reviews?'
                ]

#The outcome of treating Colorectal cancer with Panitumumab is not explicitly stated in the provided text.
criteria_text2 = ['Is the sentence a meta comment on how the effectiveness was not described in the text? Not about observed low effectiveness.',
                'Is the sentence specific enough for experts (the mention of study type and number of patients do NOT matter here)? Not a statement you may give to a general audience.',
                'Does the sentence offer interesting actionable insights for practicing oncologists even if the result is preclinical? Not an insight to continue this line of research.',
                'What is the level of priority of the study type according to levels: 1 -- clinical trial / clinical study with a number of patients greater than 1, 2 -- case report or clinical trial with 1 patient, 3 -- in vivo or in vitro, 4 -- others, 5 -- reviews?'
                ]

criteria_text3 = ['Is the sentence a meta comment on how the effectiveness was not described in the text? Not about observed low effectiveness.',
                  'Is the sentence about the effectiveness of {treatment} on {fin_condition}? Not on a specific another condition',
                'Is the sentence specific enough for experts (besides the mention of study type and number of patients)? Not a statement you may give to a general audience.',
                'Does the sentence offer interesting actionable insights for practicing oncologists for {fin_condition} even if the result is preclinical? Not an insight to continue this line of research.',
                'What is the level of priority of the study type according to levels: 1 -- clinical trial / clinical study with a number of patients greater than 1, 2 -- case report or clinical trial with 1 patient, 3 -- in vivo or in vitro, 4 -- others, 5 -- reviews?'
                ]
'''

6
enhanced_ver='Crizotinib, when combined with EpAb2-6, significantly inhibits tumor progression and prolongs survival in colon cancer animal models (in vivo; 35 patients).'
13
enhanced_ver='Crizotinib blocked the HGF/STAT3/SOX13/c-MET axis, significantly inhibiting SOX13-mediated CRC migration, invasion, and metastasis (clinical trial; 753 patients).' evaluations=['NO', 'YES', 'YES', '3']
'The study is in vitro or preclinical, which corresponds to priority level 3.'

4 
enhanced_ver='Crizotinib showed limited activity in unselected heavily pre-treated patients with advanced solid tumours (clinical trial).' evaluations=['YES', 'YES', 'NO', '1']
'The sentence directly comments on the lack of effectiveness of crizotinib in the described population.'
17
enhanced_ver='Combination binimetinib/crizotinib showed poor tolerability with no objective responses observed in RASMT advanced CRC patients (clinical trial; 36 patients).' evaluations=['YES', 'YES', 'NO', '1']
'The sentence directly comments on the lack of effectiveness of the treatment as stated in the text.'

25
enhanced_ver='Her abdominal pain resolved, and her CEA decreased from 9.5 to 4.8 (clinical trial; 2 patients with CRC were treated with crizotinib).'
7
enhanced_ver='(S)-crizotinib effectively suppressed tumour growth in animal models and showed strong antiproliferative effects on human cancer cell lines (in vivo; 0 patients).'
18
enhanced_ver='Crizotinib restored cetuximab sensitivity in CRC cell lines and is considered a promising therapeutic agent for CRCs with high IRG risk scores (in vitro; 0 patients).'
20
enhanced_ver='Crizotinib selectively inhibited the growth of ARID1A-deficient CRC cells (in vitro and in vivo; 0 patients).'

26
enhanced_ver='Molecularly targeted treatment with crizotinib induced a rapid and sustained partial response, but disseminated tumor progression occurred after 15 months (clinical trial; 40,589 patients).'
'''



'''
15
nhanced_ver='Crizotinib is mentioned as a drug in the context of P:R fusion-positive CRC, but no specific effectiveness outcome is described (in vitro and in vivo).' evaluations=['YES', 'YES', 'YES', '3']
16
enhanced_ver='Crizotinib was one of the targeted drugs used in the treatment of CRC patients, but the specific effectiveness of crizotinib is not mentioned in the text (clinical trial).' evaluations=['YES', 'YES', 'NO', '1']
31
enhanced_ver='Crizotinib is highly effective against ROS1-rearranged lung cancer but its effectiveness in colorectal cancer is not explicitly mentioned (In vitro and in vivo studies).' evaluations=['YES', 'YES', 'YES', '3']
'''












