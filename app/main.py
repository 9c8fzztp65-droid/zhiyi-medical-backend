import os
from typing import Any, Literal
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

bearer_scheme = HTTPBearer(auto_error=False)APP_API_TOKEN = os.getenv("APP_API_TOKEN", "")
# Provider credentials are server-side only. Never ship these values in the iOS app.
DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", "deepseek").lower()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
BAILIAN_API_KEY = os.getenv("BAILIAN_API_KEY", "")
BAILIAN_BASE_URL = os.getenv("BAILIAN_BASE_URL", "").rstrip("/")
BAILIAN_MODEL = os.getenv("BAILIAN_MODEL", "qwen3.7-plus")
ALLOWED_ORIGINS = [x.strip() for x in os.getenv("ALLOWED_ORIGINS", "").split(",") if x.strip()]

app = FastAPI(title="智医病历医学AI服务", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

Task = Literal["medical_record", "exam_structuring", "diagnosis_suggestions", "medication_review"]

class ClinicalRequest(BaseModel):
    provider: Literal["deepseek", "bailian"] | None = None
    task: Task
    department: str = Field(default="普通外科/中西医结合外科", max_length=100)
    document_type: str | None = Field(default=None, max_length=100)
    clinical_data: dict[str, Any]
    output_language: str = "zh-CN"

SYSTEM_PROMPT = """ 你是临床医生使用的病历辅助模型，服务于普通外科及中西医结合外科。 你只能辅助整理和分析，不能替代医生诊断、处方或临床决策。输出必须为简体中文 JSON。 最高优先级安全规则： 1. 绝不捏造患者未提供的症状、体征、生命体征、舌象、脉象、检验、影像、病理、用药或过敏信息。 2. 病史不是体格检查。不得根据病史推断并写成已经实施的阳性/阴性查体结果。 3. 对未提供或未检查的体征，写入“待检查/待补充”，不得默认正常；如无法确定，明确写“资料不足，暂不能判断”。 4. 将患者自述、医生实际查体、客观检查报告、AI推断严格区分，并尽可能保留来源。 5. 诊断仅为“建议”，逐项列出支持依据、反对/不足依据、鉴别诊断和需要核实的资料；不能把疑似写成确诊。 6. 中医辨证必须基于实际提供的四诊资料。缺少舌象、脉象等关键资料时明确指出，不可虚构。 7. 用药分析必须核对过敏史、药物名称/剂量/途径/频次/疗程；资料不全时不得声称安全或合理。 8. 对急危重症线索或严重药物过敏风险，明确提示立即由临床医生按院内流程处理。 9. 不输出自动开立医嘱，不建议未经医生审核直接执行治疗。 10. 返回可解析 JSON，不要使用 Markdown 代码围栏。 JSON 结构： { "summary": "简要说明", "medical_record": "病历正文或空字符串", "exam_items": [{"item":"项目","result":"已提供结果/待检查","source":"来源","status":"provided|missing|conflict"}], "tcm_diagnosis_suggestions": [{"name":"病名/证型","supporting_evidence":[],"missing_evidence":[],"confidence":"低/中/较高","caveat":""}], "western_diagnosis_suggestions": [{"name":"诊断建议","supporting_evidence":[],"missing_evidence":[],"differential":[],"confidence":"低/中/较高","caveat":""}], "medication_review": [{"drug":"药物","purpose":"用途/类别","allergy_check":"核对结果","issues":[],"recommendation":"供医生核查的建议"}], "quality_warnings": [], "needs_doctor_confirmation": true } 只填充与当前任务相关的字段；其余字段使用空数组或空字符串。 """

TASK_INSTRUCTIONS = {
    "medical_record": "请按医院住院病历书写风格整理原始资料。保留资料来源；缺失项目用[待补充：项目]；不要补写阴性体征。若请求包含诊断资料，可另列诊断建议，但不得自动确诊。",
    "exam_structuring": "请整理医生已经实际完成并提供的体格检查。根据病史只能生成“建议检查项目清单”，不能生成检查结果。每项标明已提供/待检查/冲突。",
    "diagnosis_suggestions": "请依据全部已提供资料分别提出中医与西医初步诊断建议，附逐条依据、缺失证据、鉴别诊断和不确定性。不得把建议写成最终诊断。",
    "medication_review": "逐项核对医嘱药物与过敏史、适应证、剂量途径频次疗程及重复用药/相互作用风险。信息不足时明确说明无法判断，不得擅自修改医嘱。"
}

def provider_config(provider: str):
    if provider == "deepseek":
        return DEEPSEEK_BASE_URL, DEEPSEEK_API_KEY, DEEPSEEK_MODEL
    if provider == "bailian":
        return BAILIAN_BASE_URL, BAILIAN_API_KEY, BAILIAN_MODEL
    raise HTTPException(status_code=400, detail="不支持的模型服务商")

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "default_provider": DEFAULT_PROVIDER,
        "providers": {
            "deepseek": bool(DEEPSEEK_API_KEY),
            "bailian": bool(BAILIAN_API_KEY and BAILIAN_BASE_URL),
        },
        "ant_af": {"mode": "external_only", "direct_api_configured": False},
    }

@app.get("/api/v1/providers")
async def providers(authorization: str | None = Header(default=None)):
    check_app_token(authorization)
    return {
        "default_provider": DEFAULT_PROVIDER,
        "available": {
            "deepseek": bool(DEEPSEEK_API_KEY),
            "bailian": bool(BAILIAN_API_KEY and BAILIAN_BASE_URL),
        },
        "external_tools": [{
            "id": "ant_af",
            "name": "蚂蚁阿福医生版",
            "mode": "external_only",
            "direct_api": False,
            "note": "当前未核实到面向第三方App开放的通用模型API；请在阿福医生版独立使用，不要通过模拟登录或抓取页面接入。"
        }]
    }

def check_app_token(authorization: str | None):
    if not APP_API_TOKEN:
        raise HTTPException(status_code=503, detail="服务未配置 APP_API_TOKEN，拒绝开放调用")
    if authorization != f"Bearer {APP_API_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.post("/api/v1/clinical/assist")
async def clinical_assist(
    payload: ClinicalRequest,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme)
):
    authorization = (
        f"{credentials.scheme} {credentials.credentials}"
        if credentials else None
    )
    check_app_token(authorization)
    provider = payload.provider or DEFAULT_PROVIDER
    base_url, api_key, model_name = provider_config(provider)
    if not api_key or not base_url or not model_name:
        raise HTTPException(status_code=503, detail=f"尚未配置 {provider} 模型服务，请检查服务端环境变量")
    user_content = {
        "task": payload.task,
        "department": payload.department,
        "document_type": payload.document_type,
        "instruction": TASK_INSTRUCTIONS[payload.task],
        "clinical_data": payload.clinical_data,
        "output_language": payload.output_language,
    }
    body = {
        "model": model_name,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": __import__("json").dumps(user_content, ensure_ascii=False)}
        ],
        "response_format": {"type": "json_object"}
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(f"{base_url}/chat/completions", headers=headers, json=body)
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            import json
            parsed = json.loads(content)
            parsed["needs_doctor_confirmation"] = True
            return {"ok": True, "provider": provider, "model": model_name, "result": parsed}
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"模型服务返回错误：HTTP {e.response.status_code}")
    except (httpx.HTTPError, KeyError, ValueError, IndexError):
        raise HTTPException(status_code=502, detail="模型调用失败或返回格式无法解析，请稍后重试")
