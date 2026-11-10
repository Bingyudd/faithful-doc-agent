from typing import Literal
from pydantic import BaseModel,ConfigDict,Field,model_validator


class Strict(BaseModel):
    model_config=ConfigDict(extra='forbid')


class Citation(Strict):
    source_id:str
    quote:str=Field(min_length=8)


class Statement(Strict):
    text:str=Field(min_length=1)
    target_document:str
    citations:list[Citation]=Field(min_length=1)


class Section(Strict):
    id:str
    title:str
    status:Literal['supported','missing','conflict','review_required']
    statements:list[Statement]=Field(default_factory=list)
    notes:list[str]=Field(default_factory=list)

    @model_validator(mode='after')
    def consistent(self):
        if self.status=='supported' and not self.statements:
            raise ValueError('Supported section requires statements')
        if self.status=='missing' and self.statements:
            raise ValueError('Missing section must not assert facts')
        if self.status=='conflict' and len({c.source_id for s in self.statements for c in s.citations})<2:
            raise ValueError('Conflict requires two distinct evidence locations')
        return self


class Report(Strict):
    title:str
    target_document:str
    sections:list[Section]


class Critique(Strict):
    issues:list[str]=Field(default_factory=list)


TEMPLATES={
 'overview':[
   {'id':'identity','title':'产品与用途','query':'device description intended use indications trade name subject device'},
   {'id':'design','title':'设计与技术特征','query':'subject device design technical characteristics diameter working length field of view components'},
   {'id':'limitations','title':'使用条件与限制','query':'single use reusable sterilization warnings contraindications limitations patient population'}],
 'testing':[
   {'id':'tests','title':'测试项目与依据','query':'performance testing bench verification validation standards ISO IEC ASTM'},
   {'id':'results','title':'测试结果与结论','query':'test results acceptance criteria passed performance safety effectiveness conclusion'},
   {'id':'gaps','title':'证据局限与待补项','query':'limitations clinical testing not performed unavailable data animal studies comparison'}]
}
