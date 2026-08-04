from pathlib import Path
from pypdf import PdfReader
import re

R=Path(__file__).parent; O=R/'data/processed'; O.mkdir(parents=True,exist_ok=True)
def y(s): return '---\n'+s.strip()+'\n---\n\n'
def score(): return '''#### Score

- Overall band: 5
- Task Achievement: Not separately provided
- Coherence and Cohesion: Not separately provided
- Lexical Resource: Not separately provided
- Grammatical Range and Accuracy: Not separately provided
'''
s=y('''title: IELTS General Training Writing Sample Candidate Responses and Examiner Comments
document_type: sample_candidate_responses
skill: writing
test_type: general_training
contains_candidate_responses: true
contains_examiner_comments: true
source_organization: IELTS
official: true
language: en''')+'''# IELTS General Training Writing Sample Candidate Responses and Examiner Comments

## General Training Writing Sample

### Writing Task 1

#### Question

[NEEDS_MANUAL_REVIEW: The Task 1 question is not included in this PDF; the PDF contains only Sample Script A and its examiner comment.]

#### Candidate Response 1

Dear Sir/Madam,

I am writing to express my dissatisfaction with my room-mate. As you know we share one room, I can not study in the room at all any more if I still stay there.

She always has friend visiting and has parties in the room. They make lots of noise and switch on the radio very loudly, for me this environment is very difficult to study and I need a quiet room. Even borrows my things without asking, it is very impolite.

I request you can give me a new room next term because I have been asked her has parties in other place many times they still have parties in the room. I really can not stay in the same room with her.

I would be grateful if you could change me a single room.

Your faithfully,

Catherine

'''+score()+'''\n#### Official Examiner Comment

The answer is below the word limit and there is some repetition of the task rubric. (Length is a common problem in General Training scripts.) Answers that are short lose marks because of inadequate content and may also lose marks because there is insufficient material in the answer for the examiner to give credit for accuracy and coherence. Despite these problems, the introduction to the letter is appropriate and the purpose of the writer is clear. The points are not always linked together well and punctuation is sometimes faulty. The sentences are kept quite simple and mistakes occur as soon as more complex structures are attempted.

### Writing Task 2

#### Question

[NEEDS_MANUAL_REVIEW: The Task 2 question is not included in this PDF; the PDF contains only Sample Script A and its examiner comment.]

#### Candidate Response 1

Who should be responsible for our people.

It is true that the old Peoples situation gets worse in the many countries. The first question must be what they want’s and what they needs? Especially their necessity are more benefit more respect more quiet life.

If they have been working for a long time in the any company or in the Public Sector and when they get old that’s means during their retire’s time company or Government must be responsible of their welfare, it is just my opinion. They should take care of them.

In addition to company or Government. If they have good money they can look after themselves. We can do something to make easier their life for example an organization or a voluntary association, unions.

The families or Relative’s responsibility depends on their wealthy situations.

If they could do they should do anything.

Government’s or their former place could supply them with life insurance and a good Social Security Policy. The Social community center or old age pensioner like in the Britain are very useful for them.

For all of them life is hard and gets harder, in the their old ages. They expect more attention and good life.

The old people, if don’t want lost them. We should do anything that what we able to do.

I.Bozyil

'''+score()+'''\n#### Official Examiner Comment

There are quite a lot of relevant ideas in the answer but they are not always well supported and sometimes they are unclear. There are some areas in the answer where the organisation becomes weak and the reader finds the message difficult to follow. Nevertheless, the writer’s view is apparent and there is a logical flow to the points given. There are a lot of mistakes in the answer and some parts, such as the conclusion, are very hard to follow because of these errors. Although there is some appropriate vocabulary, sentence control is very weak.
'''
(O/'general_training_writing_samples.md').write_text(s,encoding='utf8')

def spans(page):
 a=[]; last=[0,0]
 def v(t,cm,tm,font,size):
  if not t.strip(): return
  x,z=tm[4],tm[5]
  if x==z==0: x,z=last
  else: last[:]=[x,z]
  a.append((x,z,size,t.strip()))
 page.extract_text(visitor_text=v); return a
def cell(a):
 s=' '.join(q[3] for q in a); s=re.sub(r'\s+',' ',s); s=re.sub(r'\s+([,.;:?!])',r'\1',s); s=re.sub(r'([A-Za-z])\s+-\s+([A-Za-z])',r'\1-\2',s)
 return '\n'.join('- '+q for q in re.split(r'(?<=[.!?])\s+(?=[A-Z(])',s) if q)
def rows(pdf,pages,cols):
 d={}; rr=PdfReader(pdf)
 for pn in pages:
  grouped={}; band=None
  for q in spans(rr.pages[pn]):
   if q[0]<45 and q[2]>15 and q[3].isdigit(): band=int(q[3]); continue
   if band is None or q[2]>=10: continue
   for i,(x1,x2) in enumerate(cols):
    if x1<=q[0]<x2: grouped.setdefault((band,i),[]).append(q); break
  for k,v in grouped.items(): d[k]=cell(v)
 return d

cols=[(45,260),(260,485),(485,710),(710,1000)]
d=rows(R/'data/ielts_speaking_band_descriptors.pdf',[1,2,3],cols)
m=y('''title: IELTS Speaking Band Descriptors
document_type: band_descriptor
skill: speaking
source_organization: IELTS
official: true
language: en''')+'''# IELTS Speaking Band Descriptors

## Scoring Criteria for Academic and General Training Tests

A candidate must fully fit the positive features of the descriptor at a particular level.

A candidate will be rated on their average performance across all parts of the test.
'''
cs=['Fluency and Coherence','Lexical Resource','Grammatical Range and Accuracy','Pronunciation']
for b in range(9,-1,-1):
 m+=f'\n## Band {b}\n'
 if b==0: m+='\n- Does not attend.\n'; continue
 for i,c in enumerate(cs): m+=f'\n### {c}\n\n{d.get((b,i),"[NEEDS_MANUAL_REVIEW: Descriptor cell could not be extracted.] ")}\n'
(O/'ielts_speaking_band_descriptors.md').write_text(m,encoding='utf8')

cols=[(45,360),(360,545),(545,740),(740,1000)]
d1=rows(R/'data/ielts_writing_band_descriptors.pdf',[2,3,4],cols)
d2=rows(R/'data/ielts_writing_band_descriptors.pdf',[6,7,8],[(45,330),(330,545),(545,740),(740,1000)])
m=y('''title: IELTS Writing Band Descriptors
document_type: band_descriptor
skill: writing
test_types:
  - academic
  - general_training
source_organization: IELTS
official: true
language: en''')+'''# IELTS Writing Band Descriptors

A script must fully fit the positive features of the descriptor at a particular level. Bolded text indicates negative features that will limit a rating.
'''
for task,d,first in [('Writing Task 1',d1,'Task Achievement'),('Writing Task 2',d2,'Task Response')]:
 m+=f'\n## {task}\n'
 for b in range(9,-1,-1):
  m+=f'\n### Band {b}\n'
  if b==0:
   m+='\n- Should only be used where a candidate did not attend or attempt the task in any way, used a language other than English throughout, or where there is proof that a candidate’s answer has been totally memorised.\n'; continue
  for i,c in enumerate([first,'Coherence and Cohesion','Lexical Resource','Grammatical Range and Accuracy']): m+=f'\n#### {c}\n\n{d.get((b,i),"[NEEDS_MANUAL_REVIEW: Descriptor cell could not be extracted.] ")}\n'
  m+=f'\n[FORMATTING_NEEDS_REVIEW: Bold emphasis in the source table for {task}, Band {b} could not be identified reliably from the PDF text layer; wording is preserved.]\n'
(O/'ielts_writing_band_descriptors.md').write_text(m,encoding='utf8')

titles=['How to Set IELTS Scores','Welcome and Supporting Your Success','The Benefits of Using IELTS','Understanding IELTS','IELTS Quality and Fairness','Test Security and Accessibility','The Four Communication Skills','Assessing Listening and Reading Skills','Assessing Writing and Speaking Skills','Understanding Applicants’ Capabilities','IELTS Band Score Descriptors','IELTS Scores and CEFR Levels','Key IELTS Resources','Five Steps to Setting IELTS Scores','Step 1 — Form a Decision-Making Group','Step 1 — Stakeholder Perspectives','Step 2 — Review Your Existing Scores','Step 2 — Guidance for Educational Institutions','Step 2 — IELTS Score Guidance','Step 2 — Impacts of Inconsistent Requirements','Step 3 — Run a Score-Setting Workshop','Step 3 — IELTS 2024 Test Results','Step 4 — Communicate Your Test Scores','Step 4 — Institutional Perspective','Step 5 — Use the IELTS Results Service','Check Your Applicant’s IELTS Results','The World’s Most Trusted English Test']
m=y('''title: Guide to IELTS Scores 2025
document_type: score_guide
year: 2025
source_organization: IELTS
official: true
language: en''')+'# Guide to IELTS Scores 2025\n'
for n,p in enumerate(PdfReader(R/'data/guide-to-ielts-scores-2025.pdf').pages,1):
 ls=[]
 for q in (p.extract_text() or '').splitlines():
  q=re.sub(r'\s+',' ',q.replace('\x04',' ').replace('\xa0',' ')).strip()
  if q and not q.isdigit() and q not in ('Understanding IELTS','Setting IELTS Scores'): ls.append(q)
 m+=f'\n## {titles[n-1]}\n\n### Source Page {n}\n\n'+'\n'.join(ls)+'\n'
(O/'guide_to_ielts_scores_2025.md').write_text(m,encoding='utf8')
