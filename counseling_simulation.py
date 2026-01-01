# counseling_simulation.py
"""
심리상담 시뮬레이션 코드
- 상담사: vLLM 서버 (로컬)
- 내담자: OpenAI API
"""

import json
import os
import time
import random
import re
import requests
from openai import OpenAI
from typing import Optional, List, Dict, Any

# ==================== 설정 ====================
# OpenAI API 설정 (내담자용)
OPENAI_API_KEY = "your-key"  # 실제 키로 교체
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# vLLM 서버 설정 (상담사용)
VLLM_BASE_URL = "http://localhost:8090/v1"
vllm_client = OpenAI(base_url=VLLM_BASE_URL, api_key="dummy")

# 모델 설정
COUNSELOR_MODEL = "komit_think"  # vLLM에 로드된 모델명
#COUNSELOR_MODEL = "naive_sft/checkpoint-75"

CLIENT_MODEL = "gpt-5-mini"  # 내담자 시뮬레이션용

# 대화 설정
MAX_DIALOGUE_TURNS = 10
DELAY_BETWEEN_CALLS = 0.5

# 입출력 파일
INPUT_JSON_FILE = "mi_original_situations.json"
OUTPUT_FILE = "simulation_results_ours.jsonl"


# ==================== 상태 및 행동 정의 ====================
STATE_EXPLAIN = {
    "Precontemplation": "내담자는 행동 문제를 부정하거나 경시하며, 변화의 필요성을 인지하지 못하거나 과소평가합니다",
    "Contemplation": "내담자는 자신의 행동 문제를 인정하지만, 종종 내담자의 프로필에 포함된 신념(beliefs) 때문에 변화를 주저합니다",
    "Preparation": "내담자가 행동 변화를 위한 계획을 세우기 시작하는 단계입니다. 내담자는 행동할 준비가 되었으며, 변화를 위한 구체적인 단계를 계획합니다",
    "Termination": "상담의 마지막 단계로, 내담자는 점차 대화를 종료합니다"
}

ACTION_EXPLAIN = {
    "Precontemplation": "- Inform: 내담자의 배경, 경험 또는 감정에 대한 세부 정보를 공유합니다.\n- Engage: 상담사에게 인사, 감사, 또는 질문을 하는 등 정중하게 상호작용합니다.\n- Deny: 자신의 행동이 문제가 있거나 변화가 필요하다는 것을 직접적으로 거부합니다.\n- Downplay: 자신의 행동이나 상황의 중요성 또는 영향을 경시합니다.\n- Blame: 스트레스가 많은 생활이나 타인과 같은 외부 요인에 문제를 귀인시킵니다",
    "Contemplation": "- Inform: 내담자의 배경, 경험 또는 감정에 대한 세부 정보를 공유합니다.\n- Hesitate: 변화에 대한 양가감정을 나타내며 불확실성을 보입니다.\n- Doubt: 제안된 변화의 실용성이나 성공 가능성에 대해 회의감을 표출합니다.\n- Acknowledge: 변화의 필요성을 인정하거나 변화의 중요성, 이점, 또는 자신감을 강조합니다",
    "Preparation": "- Inform: 내담자의 배경, 경험 또는 감정에 대한 세부 정보를 공유합니다.\n- Reject: 제안된 계획이 부적절하다고 판단하여 거부합니다.\n- Accept: 제안된 행동 계획을 채택하는 데 동의합니다.\n- Plan: 변화 계획을 제안하거나 구체적인 단계를 설명합니다",
    "Termination": "- Terminate: 마무리 대화를 합니다"
}

ACTION_DICT = {
    "inform": "내담자의 배경, 경험 또는 감정에 대한 세부 정보를 공유합니다",
    "engage": "상담사에게 인사, 감사, 또는 질문을 하는 등 정중하게 상호작용",
    "deny": "자신의 행동이 문제가 있거나 변화가 필요하다는 것을 직접적으로 거부",
    "downplay": "자신의 행동이나 상황의 중요성 또는 영향을 경시",
    "blame": "스트레스가 많은 생활이나 타인과 같은 외부 요인에 문제를 귀인",
    "hesitate": "변화에 대한 양가감정을 나타내며 불확실성을 보입니다",
    "doubt": "제안된 변화의 실용성이나 성공 가능성에 대해 회의감을 표출",
    "acknowledge": "변화의 필요성을 인정하거나 변화의 중요성, 이점, 또는 자신감을 강조",
    "reject": "제안된 계획이 부적절하다고 판단하여 거부",
    "accept": "제안된 행동 계획을 채택하는 데 동의",
    "plan": "변화 계획을 제안하거나 구체적인 단계를 설명",
    "terminate": "현재 상태를 강조하고, 현 세션을 끝내고 싶다는 의사를 표현"
}


# ==================== API 호출 함수 ====================
def call_openai_api(messages: List[Dict], max_tokens: int = 200, temperature: float = 0.7) -> Optional[str]:
    """OpenAI API 호출 (내담자용)"""
    try:
        response = openai_client.chat.completions.create(
            model=CLIENT_MODEL,
            messages=messages,
#            max_tokens=max_tokens,
#            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI API 오류: {e}")
        return None


def call_vllm_api(messages: List[Dict], max_tokens: int = 500, temperature: float = 0.7) -> Optional[str]:
    """vLLM 서버 API 호출 (상담사용)"""
    try:
        response = vllm_client.chat.completions.create(
            model=COUNSELOR_MODEL,
            messages=messages,
#            max_tokens=max_tokens,
#            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"vLLM API 오류: {e}")
        return None


def extract_answer(response: str) -> str:
    """<answer></answer> 태그 사이의 내용 추출"""
    match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response  # 태그가 없으면 전체 반환


# ==================== 상태 전이 판단 ====================
PRE_TO_CON_PROMPT = """
당신은 MI(동기 강화 상담) 전문가입니다. 주어진 상담 대화 기록을 분석하여, 내담자가 다음 단계로 전이하는 조건에 있는지 알려주세요.

현재 내담자의 상태는 'Precontemplation' 입니다.
상태 전이 조건: 상담사가 내담자의 동기와 관련된 특정 이유를 언급하고 이를 통해 내담자가 동기 부여를 받을 때 Contemplation으로 진입합니다.

대화를 보았을 때, 내담자의 상태가 Contemplation으로 전이한 상태인가요?
YES or NO 로 대답하세요.
"""

CON_TO_PREP_PROMPT = """
당신은 MI(동기 강화 상담) 전문가입니다. 주어진 상담 대화 기록을 분석하여, 내담자가 다음 단계로 전이하는 조건에 있는지 알려주세요.

현재 내담자의 상태는 'Contemplation' 입니다.
상태 전이 조건: 내담자의 주저하는 이유(신념)가 상담사에 의해 적절하게 다루어졌을 때 Preparation으로 진입합니다.

대화를 보았을 때, 내담자의 상태가 Preparation으로 전이한 상태인가요?
YES or NO 로 대답하세요.
"""

PREP_TO_TER_PROMPT = """
당신은 MI(동기 강화 상담) 전문가입니다. 주어진 상담 대화 기록을 분석하여, 내담자가 다음 단계로 전이하는 조건에 있는지 알려주세요.

현재 내담자의 상태는 'Preparation' 입니다.
상태 전이 조건: 내담자가 선호하는 변화 계획(preferred change plan)에 대해 논의했을 때 Termination으로 진입합니다.

대화를 보았을 때, 내담자의 상태가 Termination으로 전이한 상태인가요?
YES or NO 로 대답하세요.
"""


def judge_state_transition(current_state: str, conversation_history: List[Dict]) -> str:
    """내담자의 상태 전이 판단"""
    if current_state == "Precontemplation":
        system_prompt = PRE_TO_CON_PROMPT
        next_state = "Contemplation"
    elif current_state == "Contemplation":
        system_prompt = CON_TO_PREP_PROMPT
        next_state = "Preparation"
    elif current_state == "Preparation":
        system_prompt = PREP_TO_TER_PROMPT
        next_state = "Termination"
    else:
        return current_state
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"대화 기록:\n{json.dumps(conversation_history, ensure_ascii=False, indent=2)}"}
    ]
    
    response = call_openai_api(messages, max_tokens=10, temperature=0.1)
    
    if response and "yes" in response.lower():
        print(f"  → 상태 전이: {current_state} → {next_state}")
        return next_state
    return current_state


# ==================== 행동 선택 ====================
def get_possible_actions(state: str) -> List[str]:
    """상태에 따른 가능한 행동 목록 반환"""
    if state == "Precontemplation":
        return ["Deny", "Engage", "Inform", "Downplay", "Blame"]
    elif state == "Contemplation":
        return ["Inform", "Hesitate", "Doubt", "Acknowledge"]
    elif state == "Preparation":
        return ["Plan", "Inform", "Accept", "Reject"]
    else:
        return ["Terminate"]


ACTION_SELECT_PROMPT = """
당신은 MI(동기 강화 상담) 전문가입니다.
내담자의 현재 상태: {state}
상태 설명: {state_explain}
내담자의 고민 상황: {summary}

주어진 대화 맥락을 고려하여 내담자의 다음 행동을 선택하세요.
내담자의 목표는 상담을 통해 문제를 해결하는 것입니다.

가능한 행동들:
{actions}

다음 중 하나만 선택하세요 (행동 이름만 반환):
"""


def select_client_action(current_state: str, conversation_history: List[Dict], summary: str) -> str:
    """내담자의 다음 행동 선택"""
    action_types = get_possible_actions(current_state)
    
    system_prompt = ACTION_SELECT_PROMPT.format(
        state=current_state,
        state_explain=STATE_EXPLAIN.get(current_state, ""),
        summary=summary,
        actions=ACTION_EXPLAIN.get(current_state, "")
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"대화 기록:\n{json.dumps(conversation_history, ensure_ascii=False, indent=2)}"}
    ]
    
    response = call_openai_api(messages, max_tokens=20, temperature=0.3)
    
    if response:
        for action in action_types:
            if action.lower() in response.lower():
                return action
    
    return random.choice(action_types)


# ==================== 내담자 응답 생성 ====================
CLIENT_SYSTEM_PROMPT = """
당신은 MI 상담을 받는 내담자 역할입니다.
당신은 [내담자 배경 기억]을 바탕으로, 상담사의 마지막 질문에 자연스럽게 응답해야 합니다.

[내담자 배경 기억]
{initial_utterance}

[현재 상태]
변화 단계: {current_state}
행동 유형: {action_type}

[규칙]
- 상담사의 마지막 질문에 간결한 1-2문장의 자연스러운 구어체 한국어로 응답하세요.
- 당신의 배경 기억을 응답에 자연스럽게 반영하세요.
- 행동 유형에 맞는 발화를 하세요.
"""


def generate_client_response(
    initial_utterance: str,
    current_state: str,
    action_type: str,
    conversation_history: List[Dict]
) -> Optional[str]:
    """내담자 응답 생성"""
    system_prompt = CLIENT_SYSTEM_PROMPT.format(
        initial_utterance=initial_utterance,
        current_state=STATE_EXPLAIN.get(current_state, current_state),
        action_type=ACTION_DICT.get(action_type.lower(), action_type)
    )
    
    messages = [{"role": "system", "content": system_prompt}]
    
    for conv in conversation_history:
        role = "assistant" if conv["role"] == "counselor" else "user"
        messages.append({"role": role, "content": conv["utterance"]})
    
    return call_openai_api(messages, max_tokens=150, temperature=0.7)


# ==================== 상담사 응답 생성 ====================
def generate_counselor_response(conversation_history: List[Dict]) -> Optional[str]:
    """vLLM을 통한 상담사 응답 생성"""
    # 대화 맥락 구성
    context = ""
    for conv in conversation_history:
        role_name = "상담사" if conv["role"] == "counselor" else "내담자"
        context += f"{role_name}: {conv['utterance']}\n"
    
    messages = [
        {"role": "user", "content": context.strip()}
    ]
    
    response = call_vllm_api(messages, max_tokens=500, temperature=0.7)
    
    if response:
        return extract_answer(response)
    return None


# ==================== 사연 요약 ====================
SUMMARIZATION_PROMPT = """
당신은 동기부여 상담 전문가입니다. 내담자가 상담 신청 시 작성한 긴 사연을 읽고, 핵심 문제와 상황을 1-2문장으로 요약해주세요.
"""


def summarize_story(story_text: str) -> str:
    """사연 요약"""
    messages = [
        {"role": "system", "content": SUMMARIZATION_PROMPT},
        {"role": "user", "content": story_text}
    ]
    
    summary = call_openai_api(messages, max_tokens=200, temperature=0.3)
    return summary if summary else story_text[:200] + "..."


# ==================== 시뮬레이션 메인 함수 ====================
def run_simulation(entry: Dict, index: int, total: int) -> Optional[List[Dict]]:
    """단일 대화 시뮬레이션 실행"""
    print(f"\n{'='*60}")
    print(f"시뮬레이션 [{index+1}/{total}] 시작")
    print(f"{'='*60}")
    
    # 원본 사연 추출
    initial_utterance = entry.get("원본", "")
    if not initial_utterance:
        print("원본 사연이 없습니다. 건너뜁니다.")
        return None
    
    # 사연 요약
    summary = summarize_story(initial_utterance)
    print(f"사연 요약: {summary[:100]}...")
    time.sleep(DELAY_BETWEEN_CALLS)
    
    # 대화 초기화
    conversation = []
    client_state = "Precontemplation"
    
    # 상담사 첫 인사
    counselor_greeting = "무엇을 도와드릴까요?"
    conversation.append({
        "turn": 0,
        "role": "counselor",
        "utterance": counselor_greeting
    })
    print(f"\n[Turn 0] 상담사: {counselor_greeting}")
    
    termination = False
    
    # 대화 진행
    for turn in range(1, MAX_DIALOGUE_TURNS * 2):
        if turn % 2 == 1:  # 내담자 턴
            # 상태 전이 판단
            new_state = judge_state_transition(client_state, conversation)
            client_state = new_state
            time.sleep(DELAY_BETWEEN_CALLS)
            
            # 행동 선택
            action_type = select_client_action(client_state, conversation, summary)
            print(f"\n[Turn {turn}] 내담자 (상태: {client_state}, 행동: {action_type})")
            time.sleep(DELAY_BETWEEN_CALLS)
            
            # 내담자 응답 생성
            client_response = generate_client_response(
                initial_utterance, client_state, action_type, conversation
            )
            
            if not client_response:
                print("내담자 응답 생성 실패")
                return None
            
            conversation.append({
                "turn": turn,
                "role": "client",
                "state": client_state,
                "action_type": action_type,
                "utterance": client_response
            })
            print(f"  → {client_response}")
            time.sleep(DELAY_BETWEEN_CALLS)
            
            if termination:
                break
            
            if client_state == "Termination":
                termination = True
                
        else:  # 상담사 턴
            print(f"\n[Turn {turn}] 상담사")
            
            counselor_response = generate_counselor_response(conversation)

            print(f'원본 ㅡㅡㅡ{counselor_response}ㅡㅡㅡ')
            
            if 'user' in counselor_response:
                counselor_response = counselor_response[:counselor_response.find('user')]
            elif '<answer>' in counselor_response:
                counselor_response = counselor_response[counselor_response.find('<answer>')+len('<answer>'):counselor_response.find('</answer>')]
            
            if not counselor_response:
                print("상담사 응답 생성 실패")
                return None
            
            conversation.append({
                "turn": turn,
                "role": "counselor",
                "utterance": counselor_response
            })
            print(f"  → {counselor_response}")
            time.sleep(DELAY_BETWEEN_CALLS)
    
    print(f"\n시뮬레이션 [{index+1}/{total}] 완료")
    return conversation


# ==================== 데이터 로드 ====================
def load_data(file_path: str) -> List[Dict]:
    """데이터 로드 및 필터링"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"데이터 로드 오류: {e}")
        return []
    
    filtered = []
    for item in data:
        original = item.get("원본")
        if isinstance(original, list) and len(original) > 0:
            item["원본"] = original[0]
            filtered.append(item)
        elif isinstance(original, str) and original.strip():
            filtered.append(item)
    
    print(f"총 {len(filtered)}개의 유효한 데이터 로드됨")
    return filtered


# ==================== 메인 ====================
def main():
    print("=" * 60)
    print("심리상담 시뮬레이션 시작")
    print(f"상담사 모델: {COUNSELOR_MODEL} (vLLM)")
    print(f"내담자 모델: {CLIENT_MODEL} (OpenAI)")
    print("=" * 60)
    
    # 데이터 로드
    data = load_data(INPUT_JSON_FILE)
    if not data:
        print("데이터가 없습니다.")
        return
    
    # 시뮬레이션 실행
    results = []
    for idx, entry in enumerate(data):
        
        # 원하는 범위만 실행 (예: 처음 10개)
        if idx <1000:
            continue
            
        result = run_simulation(entry, idx, len(data))
        
        if result:
            results.append({
                "index": idx,
                "conversation": result
            })
            
            # 중간 저장
            with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                json.dump({"index": idx, "conversation": result}, f, ensure_ascii=False)
                f.write('\n')
    
    print(f"\n{'='*60}")
    print(f"시뮬레이션 완료: {len(results)}개 대화 생성")
    print(f"결과 저장: {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()