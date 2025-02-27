#pip install pydantic==2.9.2  (downgrade pydantic to 2.9.2)
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_experimental.tabular_synthetic_data.openai import create_openai_data_generator
from langchain_experimental.tabular_synthetic_data.prompts import SYNTHETIC_FEW_SHOT_PREFIX, SYNTHETIC_FEW_SHOT_SUFFIX
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from read_properties import get_property_value

# 設定 OpenAI API 金鑰
api_key = get_property_value("openai_api_key")

# 生成數據要有想像力，把temperature調高(最高2.0)
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=api_key, temperature=0.8, max_tokens=None, timeout=None, max_retries=2)

# 定義資料模型
class MedicalBilling(BaseModel):
    patient_id: int  #患者ID
    patient_name: str  #患者姓名
    diagnosis_code: str  #診斷代碼
    procedure_code: str  #處置代碼
    total_charge: float  #總費用
    insurance_claim_amount: float  #保險理賠金額

# 提供一些樣本資料給AI
examples = [
    {
        "example": "患者ID: 12345, 患者姓名: 張三, 診斷代碼: A123, 處置代碼: B456, 總費用: $1000, 保險理賠金額: $800",
    },{
        "example": "患者ID: 67890, 患者姓名: 李四, 診斷代碼: C789, 處置代碼: D012, 總費用: $2000, 保險理賠金額: $1600",
    },{
        "example": "患者ID: 54321, 患者姓名: 王五, 診斷代碼: E345, 處置代碼: F678, 總費用: $3000, 保險理賠金額: $2400",
    }
]

# 提示模板，用來指導AI生成數據
example_prompt = PromptTemplate(input_variables=["example"], template="{example}")

prompt_template = FewShotPromptTemplate(
    prefix=SYNTHETIC_FEW_SHOT_PREFIX,
    suffix=SYNTHETIC_FEW_SHOT_SUFFIX,
    examples=examples,
    example_prompt=example_prompt,
    input_variables=["subject", "extra"]  #自定義
)

generator = create_openai_data_generator(
    output_schema=MedicalBilling,
    llm=llm,
    prompt=prompt_template
)

result = generator.generate(
    subject="醫療帳單",  #指定生成數據的主題
    extra="名字是隨機的，但是要使用比較生僻且看起來像是真實的人名",  #額外的提示
    runs=10  #生成10筆數據
)
for medicalBilling in result:
    print(medicalBilling)
