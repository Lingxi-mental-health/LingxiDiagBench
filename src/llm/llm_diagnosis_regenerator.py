# -*- coding: utf-8 -*-
"""
基于已有对话重新生成诊断的脚本 - 支持并行处理和OpenRouter

该脚本用于：
1. 读取已生成的医患对话数据（all_conversations.json格式）
2. 去除原有的诊断结果
3. 重新构建对话历史
4. 使用LLM重新生成诊断结果（支持多端口并行 + OpenRouter）
5. 保持原有数据格式输出
"""

import json
import os
import time
import multiprocessing
from multiprocessing import Process, Manager
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.llm import llm_tools_api

# 加载环境变量
load_dotenv()


def api_llm_diagnosis(dialogue_history, patient_template, doctor_persona, model_name, client):
    """
    基于对话历史使用LLM生成诊断结果
    
    Args:
        dialogue_history (str): 构建的对话历史字符串
        patient_template (dict): 患者信息模板
        doctor_persona (str): 医生角色描述
        model_name (str): 模型名称
        client: API客户端对象
        
    Returns:
        str: 诊断结果字符串
    """
    # 使用与 doctor_base.py 相同的诊断提示词
    diagnosis_prompt = (
        f"基于当前的对话历史：{dialogue_history} 以及之前的对话信息：{patient_template.get('cleaned_text', '')}，"
        "请输出本次会诊的最终精神科主诊断（仅从以下 ICD-10 代码中选择其一："
        "F20、F31、F32、F39、F41、F42、F43、F45、F51、F98）。\n\n"
        "你是一名资深精神科主任医师。请严格参考《国际疾病分类 ICD-10 精神与行为障碍诊断标准（中文版）》的临床原始描述，结合患者的症状表现、病程特点和功能受损情况来进行判断，而不是仅仅依赖要点式总结。\n\n"
        "## 诊断范围（含要点提示）\n"
        "- F20 精神分裂症：持续 ≥1 月的妄想、幻觉、思维形式障碍或明显阴性症状，排除器质性和物质原因。\n"
        "- F31 躁郁症（双相情感障碍）：至少一次躁狂或轻躁狂发作，常与抑郁发作交替。\n"
        "- F32 抑郁发作：以情绪低落、兴趣缺乏、快感缺失为核心，伴疲乏、睡眠食欲改变、自罪或无价值感。\n"
        "- F39 情感障碍未特指：临床明确存在情感障碍，但无法明确归入抑郁或双相等具体亚型。\n"
        "- F41 焦虑障碍：包括恐慌发作或广泛性焦虑，表现为持续紧张、担忧及自主神经症状。\n"
        "- F42 强迫障碍：反复出现的强迫思维或强迫行为，个体自知过度，但难以控制，显著影响功能。\n"
        "- F43 应激相关障碍：在重大应激或创伤后出现急性应激反应、创伤后应激障碍或适应障碍。\n"
        "- F45 躯体形式障碍：以难以用医学解释的多种躯体主诉为核心，伴健康焦虑及反复求医。\n"
        "- F51 非器质性睡眠障碍：失眠、过度嗜睡、噩梦、睡眠节律紊乱等，排除器质性原因。\n"
        "- F98 儿童期起病的其他行为与情绪障碍：如遗尿、遗粪、拔毛、习惯性抽动、儿童期睡眠障碍等。\n\n"
        "## 输出要求\n"
        "严格按照以下格式输出唯一诊断结果：icd_code{F20|F31|F32|F39|F41|F42|F43|F45|F51|F98}。\n"
        "除上述花括号内的唯一 ICD 代码外，不要输出任何其他文字、标点或解释。\n"
    )
    diagnosis_prompt = (
            f"基于当前的对话历史：{dialogue_history}以及之前的对话信息：{patient_template.get('cleaned_text','')}，"
            "输出最终的精神科诊断结果。\n\n"
            "你是一名资深精神科主任医师。请严格参考《国际疾病分类 ICD-10 精神与行为障碍诊断标准（中文版）》的临床原始描述，结合患者的症状表现、病程特点和功能受损情况来进行判断，而不是仅仅依赖要点式总结。\n\n"
            "## 诊断范围（含要点提示）\n"
            "F32 忧郁发作：在典型的轻度、中度或重度忧郁症发作中，患者通常有忧郁情绪、体力下降、活动减少，快乐感、兴趣、注意力均减低，稍微活动即可感到疲倦。常伴有睡眠障碍、食欲减低、自我评价与自信降低，有罪恶感与无用感的意念（即便在轻度时亦可存在）。情绪持续低潮，对生活情境反应减弱。部分患者出现身体性症状，如失去兴趣或快乐感、清晨早醒、晨间症状加重、精神运动迟滞或激动、食欲减退、体重下降、性欲降低。根据症状数量与严重度可分轻度、中度、重度，重度者可伴精神病性症状。\n"
            "F41 焦虑障碍：包括恐慌症和广泛性焦虑症。恐慌症的基本特征是反复发作的强烈焦虑（恐慌），发生与情境无关、不可预期，伴心悸、胸痛、窒息感、头晕、不真实感，以及害怕死亡、失控或发狂等。广泛性焦虑症表现为广泛且持续的焦虑，与特定情境无关，常伴紧张、颤抖、肌肉紧张、出汗、头轻飘感、心悸、头晕、上腹不适等，并可能担心自己或亲人的健康或安全。症状可造成明显困扰或功能受损。\n"
            "F32,F41 焦虑抑郁混合障碍：抑郁与焦虑症状同时存在且都达到ICD-10诊断标准。若能够判断主次，应优先选择单一类别（F32 或 F41）。\n"
            "Others：除 F32、F41、F32,F41 之外的其他精神障碍，或当前信息不足以明确归入前三类，也包括无明显精神症状的情形。常见范围（ICD-10）：F00–F09 器质性（含症状性）精神障碍；F10–F19 物质使用所致障碍；F20–F29 精神分裂谱系与妄想性障碍；F30–F39 情感性障碍；F40–F48 神经症性/应激性/躯体形式障碍；F50–F59 伴生理障碍与躯体因素的行为综合征；F60–F69 成人人格与行为障碍；F70–F79 精神发育迟缓；F80–F89 心理发育障碍；F90–F98 儿童青少年期行为与精神障碍；F99 精神障碍NOS。\n"
            "## 判别参考（内部使用，不在输出中展示）\n"
            "1) 优先判断单一类别（F32 或 F41），如一方症状明显占优则直接选该类。\n"
            "2) 仅在两类症状与严重程度接近、抑郁与焦虑症状同时存在且都达到ICD-10诊断标准时，才选 F32,F41。\n"
            "3) 当抑郁和焦虑症状均不足、或症状特征更符合上述其他精神障碍，或呈现无明显精神症状时，选 Others。\n"
            "## 输出要求\n"
            "严格按照以下格式输出诊断结果：icd_code{F32}、icd_code{F41}、icd_code{F32,F41} 或 icd_code{Others}（四选一），不要输出其他任何文字。\n"
    )
        
    diagnosis_prompt = (
        f"基于当前的对话历史：{dialogue_history}以及之前的对话信息：{patient_template.get('cleaned_text', '')}，"
        "输出最终的精神科诊断结果 \n\n" 
        "你是一名资深精神科主任医师。请严格参考《国际疾病分类 ICD-10 精神与行为障碍诊断标准（中文版）》的临床原始描述，结合患者的症状表现、病程特点和功能受损情况来进行判断，而不是仅仅依赖要点式总结。\n\n"
        "## 诊断范围（含要点提示）"
        "F32 忧郁发作：在典型的轻度、中度或重度忧郁症发作中，患者通常有忧郁情绪、体力下降、活动减少，快乐感、兴趣、注意力均减低，稍微活动即可感到疲倦。常伴有睡眠障碍、食欲减低、自我评价与自信降低，有罪恶感与无用感的意念（即便在轻度时亦可存在）。情绪持续低潮，对生活情境反应减弱。部分患者出现身体性症状，如失去兴趣或快乐感、清晨早醒、晨间症状加重、精神运动迟滞或激动、食欲减退、体重下降、性欲降低。根据症状数量与严重度可分轻度、中度、重度，重度者可伴精神病性症状。\n"
        "F41 焦虑障碍：包括恐慌症和广泛性焦虑症。恐慌症的基本特征是反复发作的强烈焦虑（恐慌），发生与情境无关、不可预期，伴心悸、胸痛、窒息感、头晕、不真实感，以及害怕死亡、失控或发狂等。广泛性焦虑症表现为广泛且持续的焦虑，与特定情境无关，常伴紧张、颤抖、肌肉紧张、出汗、头轻飘感、心悸、头晕、上腹不适等，并可能担心自己或亲人的健康或安全。症状可造成明显困扰或功能受损。\n"
        "## 判别参考（内部使用，不在输出中展示）\n"
        "1) 当抑郁症状为主时，选 F32。\n"
        "2) 当焦虑症状为主时，选 F41。\n"
        "3) 如两类症状均存在但一方更为突出，则选该方对应的 ICD 代码。\n"
        "## 输出要求\n"
        "严格按照以下格式输出诊断结果：icd_code{F32} 或 icd_code{F41}（二选一），不要输出其他任何文字。\n"
    )
    
    diagnosis_messages = [
        {"role": "system", "content": doctor_persona},
        {"role": "user", "content": diagnosis_prompt}
    ]
    
    # 带重试机制的诊断生成
    max_retries = 2
    diag_result = None
    
    for retry in range(max_retries):
        try:
            chat_response = client.chat.completions.create(
                model=model_name,
                messages=diagnosis_messages,
                temperature=0.3
            )
            
            # 检查响应有效性
            if chat_response is None or not hasattr(chat_response, 'choices') or not chat_response.choices:
                if retry < max_retries - 1:
                    print(f"[诊断生成] 警告: API返回无效响应，重试第 {retry + 1} 次...")
                    continue
                else:
                    print(f"[诊断生成] 错误: API返回无效响应，已达最大重试次数")
                    return "诊断结束，你的诊断结果为：\n\n无法生成诊断（API返回无效响应）"
            
            # 提取响应内容
            try:
                content = chat_response.choices[0].message.content
                if content is None or content.strip() == "":
                    if retry < max_retries - 1:
                        print(f"[诊断生成] 警告: API返回空内容，重试第 {retry + 1} 次...")
                        continue
                    else:
                        print(f"[诊断生成] 错误: API返回空内容，已达最大重试次数")
                        return "诊断结束，你的诊断结果为：\n\n无法生成诊断（API返回空内容）"
                
                diag_result = content
                break  # 成功获取诊断
                
            except (AttributeError, IndexError) as e:
                if retry < max_retries - 1:
                    print(f"[诊断生成] 警告: 无法提取诊断内容: {e}，重试第 {retry + 1} 次...")
                    continue
                else:
                    print(f"[诊断生成] 错误: 无法提取诊断内容: {e}，已达最大重试次数")
                    return f"诊断结束，你的诊断结果为：\n\n无法生成诊断（格式错误: {str(e)}）"
                    
        except Exception as e:
            if retry < max_retries - 1:
                print(f"[诊断生成] 警告: API调用异常: {e}，重试第 {retry + 1} 次...")
                continue
            else:
                print(f"[诊断生成] 错误: API调用异常: {e}，已达最大重试次数")
                return f"诊断结束，你的诊断结果为：\n\n无法生成诊断（错误: {str(e)}）"
    
    # 返回成功生成的诊断
    diag_result = "诊断结束，你的诊断结果为：\n\n" + diag_result
    return diag_result


def extract_base_model_name(model_path):
    """
    从模型路径中提取基础模型名称
    
    Args:
        model_path (str): 模型路径，可能包含端口信息
        
    Returns:
        str: 基础模型名称
    """
    return model_path.split(':')[0].split('/')[-1]


def construct_dialogue_history(conversation_data):
    """
    构建对话历史字符串，去除最后的诊断结果
    
    Args:
        conversation_data (list): 对话数据列表
        
    Returns:
        str: 格式化的对话历史字符串
    """
    dialogue_history = []
    
    for turn in conversation_data:
        # 跳过包含诊断结果的轮次
        if 'doctor' in turn and '诊断结束' in turn['doctor']:
            break
            
        if 'doctor' in turn and 'patient' in turn:
            dialogue_history.append(f"医生: {turn['doctor']}")
            dialogue_history.append(f"患者: {turn['patient']}")
    
    return "\n".join(dialogue_history)


class DiagnosisRegenerator:
    """诊断重新生成器 - 支持并行处理和OpenRouter"""
    
    def __init__(self, model_name, ports=None, max_parallel_processes=None, raw_data_path=None, model_mode='offline'):
        """
        初始化诊断重新生成器
        
        Args:
            model_name (str): 模型名称（offline模式为路径，openrouter模式为模型ID）
            ports (list): 可用的API端口列表（仅offline模式使用）
            max_parallel_processes (int): 最大并行进程数
            raw_data_path (str): 原始数据文件路径，用于获取cleaned_text
            model_mode (str): 模型模式，'offline' 或 'openrouter'
        """
        self.model_name = model_name
        self.model_mode = model_mode.lower()
        self.ports = ports if isinstance(ports, list) else ([ports] if ports else [])
        
        # 根据模式设置并行进程数
        if max_parallel_processes:
            self.max_parallel_processes = max_parallel_processes
        elif self.model_mode == 'openrouter':
            self.max_parallel_processes = 3  # OpenRouter默认并行数
        else:
            self.max_parallel_processes = len(self.ports) if self.ports else 1
        
        self.raw_data_path = raw_data_path
        
        print(f"诊断重新生成器初始化:")
        print(f"- 模型模式: {self.model_mode.upper()}")
        print(f"- 模型: {self.model_name}")
        if self.model_mode == 'offline':
            print(f"- 端口: {self.ports}")
        print(f"- 最大并行进程数: {self.max_parallel_processes}")
        print(f"- 原始数据文件: {self.raw_data_path}")
        
        # 加载原始数据用于获取cleaned_text
        self.raw_data_dict = {}
        if self.raw_data_path and os.path.exists(self.raw_data_path):
            try:
                with open(self.raw_data_path, 'r', encoding='utf-8') as f:
                    raw_data_list = json.load(f)
                # 构建patient_id到数据的映射字典
                for item in raw_data_list:
                    patient_id = item.get('patient_id')
                    if patient_id:
                        self.raw_data_dict[patient_id] = item
                print(f"- 成功加载 {len(self.raw_data_dict)} 个患者的原始数据")
            except Exception as e:
                print(f"- 警告：加载原始数据失败: {e}")
                self.raw_data_dict = {}
        else:
            print(f"- 警告：原始数据文件不存在或未指定，将使用空的cleaned_text")
        
        # 医生角色描述（用于诊断）
        self.doctor_persona = """# 角色定位
你是一名资深的精神卫生中心临床心理科主任医师，正在根据问诊结果进行精神科诊断。

# 专业要求
- 严格按照《国际疾病分类 ICD-10 精神与行为障碍诊断标准》进行诊断
- 结合患者的症状表现、病程特点和功能受损情况进行综合判断
- 基于充分的临床证据做出准确诊断
"""
    
    def process_single_patient(self, patient_info, shared_dict, process_id, patient_index):
        """处理单个患者的函数，运行在独立进程中"""
        patient_data, patient_idx = patient_info
        patient_id = patient_data.get('patient_id', f'unknown_{patient_idx}')
        
        try:
            print(f"进程 {process_id}: 开始处理患者 {patient_id} (第{patient_idx+1}个)")
            
            # 根据模式初始化客户端
            if self.model_mode == 'openrouter':
                # OpenRouter模式：直接使用模型名称
                client = llm_tools_api.doctor_client_init(self.model_name)
                api_model_name = llm_tools_api.extract_base_model_name(self.model_name)
                print(f"进程 {process_id}: 使用OpenRouter模型 {api_model_name}")
            else:
                # Offline模式：根据患者索引选择端口（负载均衡）
                selected_port = self.ports[patient_idx % len(self.ports)]
                model_with_port = f"{self.model_name}:{selected_port}"
                client = llm_tools_api.doctor_client_init(model_with_port)
                api_model_name = self.model_name  # 不包含端口
                print(f"进程 {process_id}: 使用离线模型 {model_with_port}")
            
            # 提取对话数据
            conversation = patient_data.get('conversation', [])
            
            # 构建对话历史（去除诊断结果）
            dialogue_history = construct_dialogue_history(conversation)
            
            # 从原始数据中获取cleaned_text
            cleaned_text = ""
            if patient_id in self.raw_data_dict:
                cleaned_text = self.raw_data_dict[patient_id].get('cleaned_text', '')
                print(f"进程 {process_id}: 为患者 {patient_id} 找到 cleaned_text，长度: {len(cleaned_text)}")
            else:
                print(f"进程 {process_id}: 警告：患者 {patient_id} 未在原始数据中找到，使用空的 cleaned_text")
            
            # 构建患者模板（包含真实的cleaned_text）
            patient_template = {
                'cleaned_text': cleaned_text,
                'patient_id': patient_id
            }
            
            print(f"进程 {process_id}: 正在为患者 {patient_id} 生成诊断...")
            
            # 重新生成诊断
            new_diagnosis = api_llm_diagnosis(
                dialogue_history, 
                patient_template, 
                self.doctor_persona, 
                api_model_name, 
                client
            )
            
            # 构建新的对话数据
            new_conversation = []
            for turn in conversation:
                if 'doctor' in turn and '诊断结束' in turn['doctor']:
                    break  # 跳过原有的诊断结果
                new_conversation.append(turn)
            
            # 添加新的诊断结果
            new_conversation.append({
                "doctor": new_diagnosis
            })
            
            # 构建输出数据（保持与输入格式一致）
            output_data = {
                'patient_id': patient_id,
                'conversation': new_conversation,
                'label': patient_data.get('label', '')  # 保留原有标签
            }
            
            print(f"进程 {process_id}: 患者 {patient_id} 处理完成")
            
            # 将结果存储到共享字典中
            result_key = f"patient_{patient_id}"
            shared_dict[result_key] = {
                'success': True,
                'process_id': process_id,
                'data': output_data
            }
            if self.model_mode == 'offline':
                shared_dict[result_key]['port'] = selected_port
            
        except Exception as e:
            error_msg = f"处理患者 {patient_id} 时发生错误: {str(e)}"
            print(f"进程 {process_id}: {error_msg}")
            
            result_key = f"patient_{patient_id}"
            shared_dict[result_key] = {
                'success': False,
                'error': str(e),
                'process_id': process_id,
                'patient_id': patient_id
            }
            if self.model_mode == 'offline' and 'selected_port' in locals():
                shared_dict[result_key]['port'] = selected_port
    
    def run_parallel_processing(self, input_file, output_file):
        """
        运行并行处理
        
        Args:
            input_file (str): 输入JSON文件路径（all_conversations.json格式）
            output_file (str): 输出JSON文件路径
        """
        print(f"开始并行重新生成诊断结果...")
        print(f"输入文件: {input_file}")
        print(f"输出文件: {output_file}")
        
        # 读取输入文件
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                patient_data_list = json.load(f)
            print(f"成功加载 {len(patient_data_list)} 个患者数据")
        except FileNotFoundError:
            print(f"错误: 找不到输入文件 {input_file}")
            return 0, 0
        except Exception as e:
            print(f"错误: 加载输入文件失败: {e}")
            return 0, 0
        
        if not patient_data_list:
            print("输入文件为空")
            return 0, 0
        
        print(f"最大并行进程数: {self.max_parallel_processes}")
        
        # 创建患者信息列表（患者数据，索引）
        patient_infos = [(patient_data, i) for i, patient_data in enumerate(patient_data_list)]
        
        # 创建共享字典用于收集结果
        manager = Manager()
        shared_dict = manager.dict()
        
        # 创建进程池
        processes = []
        patient_queue = list(patient_infos)
        
        # 启动初始进程
        for i in range(min(self.max_parallel_processes, len(patient_queue))):
            if patient_queue:
                patient_info = patient_queue.pop(0)
                
                process = Process(
                    target=self.process_single_patient,
                    args=(patient_info, shared_dict, i, i)
                )
                process.start()
                processes.append((process, i))
        
        # 动态分配剩余患者
        while patient_queue or any(p.is_alive() for p, _ in processes):
            # 检查已完成的进程
            for i, (process, process_id) in enumerate(processes):
                if not process.is_alive():
                    process.join()
                    
                    # 如果还有待处理的患者，启动新进程
                    if patient_queue:
                        patient_info = patient_queue.pop(0)
                        new_process = Process(
                            target=self.process_single_patient,
                            args=(patient_info, shared_dict, process_id, patient_info[1])
                        )
                        new_process.start()
                        processes[i] = (new_process, process_id)
                    else:
                        # 没有更多患者，移除已完成的进程
                        processes[i] = None
            
            # 清理已完成的进程
            processes = [p for p in processes if p is not None]
            
            # 短暂等待避免忙等待
            time.sleep(1)
        
        # 等待所有进程完成
        for process, _ in processes:
            if process:
                process.join()
        
        # 收集结果并写入输出文件
        successful_patients = 0
        error_patients = 0
        output_data_list = []
        
        print("\n=== 处理结果统计 ===")
        for key, result in sorted(shared_dict.items()):
            if result['success']:
                successful_patients += 1
                output_data_list.append(result['data'])
                port_info = f" -> 端口 {result['port']}" if 'port' in result else ""
                print(f"✅ 患者 {result['data']['patient_id']}{port_info}")
            else:
                error_patients += 1
                print(f"❌ 患者 {result.get('patient_id', 'unknown')} -> 错误: {result['error']}")
        
        # 写入输出文件
        try:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data_list, f, ensure_ascii=False, indent=2)
            print(f"\n成功写入输出文件: {output_file}")
        except Exception as e:
            print(f"\n错误: 写入输出文件失败: {e}")
        
        print(f"\n=== 并行处理完成 ===")
        print(f"成功处理患者数: {successful_patients}")
        print(f"处理失败患者数: {error_patients}")
        print(f"输出文件: {output_file}")
        
        return successful_patients, error_patients


if __name__ == "__main__":
    # =================== 配置参数 ===================
    
    # 模型模式配置
    MODEL_MODE = os.getenv('MODEL_MODE', 'offline')  # 可选: 'offline' 或 'openrouter'
    
    print("=" * 60)
    print(f"诊断重新生成脚本 - {MODEL_MODE.upper()}模式")
    print("=" * 60)
    
    if MODEL_MODE == 'openrouter':
        # ========== OpenRouter模式配置 ==========
        DOCTOR_MODEL = os.getenv('OPENROUTER_DOCTOR_MODEL', 'qwen/qwen3-8b')
        MAX_PARALLEL = int(os.getenv('OPENROUTER_MAX_PARALLEL', '3'))
        API_PORTS = None
        
        print(f"OpenRouter Doctor模型: {DOCTOR_MODEL}")
        print(f"并行进程数: {MAX_PARALLEL}")
        
    else:  # offline模式
        # ========== 离线模型模式配置 ==========
        DOCTOR_MODEL = os.getenv('OFFLINE_DOCTOR_MODEL', '../../models/Qwen3-8B')
        
        # 从环境变量读取端口配置
        doctor_ports_str = os.getenv('OFFLINE_DOCTOR_PORTS', '9041')
        API_PORTS = [int(p.strip()) for p in doctor_ports_str.split(',')]
        MAX_PARALLEL = len(API_PORTS)
        
        print(f"离线Doctor模型: {DOCTOR_MODEL}")
        print(f"API端口: {API_PORTS}")
        print(f"并行进程数: {MAX_PARALLEL}")
    
    # 原始数据文件路径（用于获取cleaned_text）
    RAW_DATA_PATH = os.getenv('RAW_DATA_PATH', './raw_data/SMHC_LingxiDiag-16K_validation_data_100samples.json')
    
    # 输入输出文件配置
    INPUT_FILE = './DataSyn/all_conversations.json'
    OUTPUT_FILE = './DataSyn/all_conversations_regenerated.json'
    
    print(f"原始数据文件: {RAW_DATA_PATH}")
    print(f"输入文件: {INPUT_FILE}")
    print(f"输出文件: {OUTPUT_FILE}")
    print("=" * 60)
    
    # 创建诊断重新生成器
    regenerator = DiagnosisRegenerator(
        model_name=DOCTOR_MODEL,
        ports=API_PORTS,
        max_parallel_processes=MAX_PARALLEL,
        raw_data_path=RAW_DATA_PATH,
        model_mode=MODEL_MODE
    )
    
    # 执行并行处理
    successful, failed = regenerator.run_parallel_processing(INPUT_FILE, OUTPUT_FILE)
    
    print("\n" + "=" * 60)
    print("🎉 诊断重新生成完成！")
    print(f"成功: {successful}, 失败: {failed}")
    print("=" * 60)
