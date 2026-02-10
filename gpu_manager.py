import json
import os
import time
import re
import multiprocessing
from multiprocessing import Process, Queue, Manager
from tqdm import tqdm
import inspect

def extract_label_from_conversation(conversation):
    """从对话中提取诊断标签（ICD代码）"""
    # 遍历对话，找到最后一条医生消息
    for item in reversed(conversation):
        if 'doctor' in item:
            doctor_msg = item['doctor']
            # 使用正则表达式提取<box>标签中的内容
            match = re.search(r'<box>(.*?)</box>', doctor_msg)
            if match:
                return match.group(1)
    return ""

class GPUManager:
    """GPU任务分发管理器 - 纯粹的并行处理器，不关心具体使用哪个版本的类"""
    
    def __init__(self, max_parallel_processes, doctor_prompt_path, diagtree_path, num_rounds=1, 
                 doctor_model_name=None, patient_model_name=None, 
                 doctor_port=None, patient_port=None,
                 doctor_class=None, patient_class=None, 
                 doctor_version='base', enable_chief_complaint=True,
                 output_filename='all_conversations.json',
                 output_filename_with_classification='all_conversations_with_classification.json'):
        """
        初始化GPU管理器
        
        Args:
            max_parallel_processes: 最大并行进程数
            doctor_prompt_path: 医生配置文件路径
            diagtree_path: 诊断树配置文件路径
            num_rounds: 对话轮数
            doctor_model_name: Doctor模型名称
            patient_model_name: Patient模型名称
            doctor_port: Doctor端口列表
            patient_port: Patient端口列表
            doctor_class: Doctor类（由main_parallel.py传入）
            patient_class: Patient类（由main_parallel.py传入）
            doctor_version: Doctor版本信息（'v1', 'v2', 'base'）
            enable_chief_complaint: 是否启用患者主诉功能
            output_filename: 输出文件名（正常版本），默认为 'all_conversations.json'
            output_filename_with_classification: 输出文件名（带分类版本），默认为 'all_conversations_with_classification.json'
        """
        # 基本参数
        self.max_parallel_processes = max_parallel_processes
        self.doctor_model_name = doctor_model_name
        self.patient_model_name = patient_model_name
        self.doctor_ports = doctor_port or [9041]
        self.patient_ports = patient_port or [9040]
        
        self.doctor_prompt_path = doctor_prompt_path
        self.diagtree_path = diagtree_path
        self.num_rounds = num_rounds
        
        # 接收传入的类和版本信息
        self.Doctor = doctor_class
        self.Patient = patient_class
        self.doctor_version = doctor_version
        self.enable_chief_complaint = enable_chief_complaint
        
        # 输出文件名配置
        self.output_filename = output_filename
        self.output_filename_with_classification = output_filename_with_classification
        
        print(f"端口配置: Doctor端口={self.doctor_ports}, Patient端口={self.patient_ports}")
        print(f"并行模式: 每个患者对话中doctor和patient交替工作")
        print(f"最大并行进程数: {self.max_parallel_processes} (处理{self.max_parallel_processes}个患者)")
        
        # Doctor版本信息输出
        version_info_map = {
            'v1': "V1 (传统诊断树)",
            'v2': "V2 (阶段式诊断树)", 
            'base': "基础版本 (无诊断树问诊)"
        }
        version_info = version_info_map.get(self.doctor_version, f"未知版本: {self.doctor_version}")
        
        print(f"GPUManager 初始化完成，当前Doctor版本: {version_info}")
        print(f"使用的Doctor类: {self.Doctor.__name__ if self.Doctor else 'None'}")
        print(f"使用的Patient类: {self.Patient.__name__ if self.Patient else 'None'}")
        print(f"患者主诉生成功能: {'启用' if self.enable_chief_complaint else '禁用'}")
        print(f"Doctor模型配置: {self.doctor_model_name} (端口: {self.doctor_ports})")
        print(f"Patient模型配置: {self.patient_model_name} (端口: {self.patient_ports})")
        print(f"输出文件名配置: 正常版本={self.output_filename}, 带分类版本={self.output_filename_with_classification}")
        
    def process_single_patient(self, patient_template, shared_dict, process_id, output_dir, output_dir_with_classification=None):
        """处理单个患者的函数，运行在独立进程中"""
        # 安全的字符串处理，防止None值
        def safe_strip(text):
            return text.strip() if text is not None else ""
            
        try:
            print(f"进程 {process_id}: 开始处理患者 {patient_template['patient_id']}")
            
            # 从端口列表中选择端口（负载均衡）
            patient_id = patient_template.get('patient_id', 0)
            selected_doctor_port = self.doctor_ports[hash(str(patient_id)) % len(self.doctor_ports)]
            selected_patient_port = self.patient_ports[hash(str(patient_id)) % len(self.patient_ports)]
            
            # 分别为doctor和patient配置模型名称和端口
            # 检测是否为OpenRouter模式（模型名称包含"provider/"表示API格式，如 "moonshotai/kimi-k2"）
            # 注意：需要排除路径格式（以/开头或包含@符号的是vLLM模式）
            is_openrouter_doctor = ('/' in self.doctor_model_name and 
                                   not self.doctor_model_name.startswith('/') and 
                                   '@' not in self.doctor_model_name)
            is_openrouter_patient = ('/' in self.patient_model_name and 
                                    not self.patient_model_name.startswith('/') and 
                                    '@' not in self.patient_model_name)
            
            # 处理 Doctor 模型名称
            if is_openrouter_doctor:
                # OpenRouter模式：不添加端口号
                doctor_model_with_port = self.doctor_model_name
                print(f"进程 {process_id}: Doctor使用模型: {doctor_model_with_port} (OpenRouter API)")
            else:
                # 离线/远程vLLM模式：添加端口号
                # 如果模型名称包含@（远程vLLM），格式为 /path/to/model@host:port
                # 否则为本地vLLM，格式为 /path/to/model:port
                doctor_model_with_port = f"{self.doctor_model_name}:{selected_doctor_port}"
                if '@' in self.doctor_model_name:
                    print(f"进程 {process_id}: Doctor使用模型: {doctor_model_with_port} (远程vLLM)")
                else:
                    print(f"进程 {process_id}: Doctor使用模型: {doctor_model_with_port} (本地vLLM，从端口列表{self.doctor_ports}中选择)")
            
            # 处理 Patient 模型名称
            if is_openrouter_patient:
                # OpenRouter模式：不添加端口号
                patient_model_with_port = self.patient_model_name
                print(f"进程 {process_id}: Patient使用模型: {patient_model_with_port} (OpenRouter API)")
            else:
                # 离线/远程vLLM模式：添加端口号
                patient_model_with_port = f"{self.patient_model_name}:{selected_patient_port}"
                if '@' in self.patient_model_name:
                    print(f"进程 {process_id}: Patient使用模型: {patient_model_with_port} (远程vLLM)")
                else:
                    print(f"进程 {process_id}: Patient使用模型: {patient_model_with_port} (本地vLLM，从端口列表{self.patient_ports}中选择)")
            
            total_output_list = []
            total_output_list_with_classification = []
            total_cost = 0
            
            for i in range(self.num_rounds):
                dialogue_history = []
                output_list = []
                output_list_with_classification = []
                output_dict = {}
                output_dict_with_classification = {}
                
                # 初始化医生和患者，分别使用不同的模型配置
                # Doctor使用doctor_model_with_port连接到doctor端口
                # Patient使用patient_model_with_port连接到patient端口
                doc = self.Doctor(patient_template, 
                                 self.doctor_prompt_path, 
                                 self.diagtree_path, 
                                 doctor_model_with_port, 
                                 True)
                
                # 检查Patient类的__init__方法是否支持enable_chief_complaint参数
                patient_init_signature = inspect.signature(self.Patient.__init__)
                patient_params = patient_init_signature.parameters
                
                # 根据Patient类是否支持enable_chief_complaint参数来创建实例
                if 'enable_chief_complaint' in patient_params:
                    # Patient CoT版本：支持enable_chief_complaint参数
                    pat = self.Patient(patient_template, patient_model_with_port, True, 
                                      enable_chief_complaint=self.enable_chief_complaint)
                else:
                    # Patient V1版本：不支持enable_chief_complaint参数
                    pat = self.Patient(patient_template, patient_model_with_port, True)
                
                if self.enable_chief_complaint:
                    print(f"进程 {process_id}: 患者主诉生成功能已启用")
                else:
                    print(f"进程 {process_id}: 患者主诉生成功能已禁用")
                
                print(f"进程 {process_id}: 患者 {patient_template['patient_id']} 第 {i+1} 轮对话开始")
                
                if self.doctor_version == 'base':
                    # === 基础版本：医生问候 + 患者主诉 + 针对性问诊 ===
                    # 医生首次响应（问候询问）
                    # doctor_response_gen返回：(response, topic, reasoning)
                    doctor_greeting, _, greeting_reasoning = doc.doctor_response_gen(None, None)
                    print(f"进程 {process_id}: 医生问候: {doctor_greeting}")
                    
                    # 患者主诉回复
                    # 检查Patient类是否有api_generate_chief_complaint方法（Patient CoT支持）
                    if hasattr(pat, 'api_generate_chief_complaint'):
                        # 使用患者模块生成主诉（Patient CoT版本）
                        patient_chief_complaint = pat.api_generate_chief_complaint()
                        print(f"进程 {process_id}: 患者主诉（AI生成）: {patient_chief_complaint}")
                    else:
                        # 使用模板中的ChiefComplaint（Patient V1版本）
                        raw_complaint = patient_template.get('ChiefComplaint', '我感觉不太舒服')
                        # 去除"主诉："前缀，使回答更自然
                        patient_chief_complaint = raw_complaint.replace('主诉：', '').replace('主诉:', '').strip()
                        print(f"进程 {process_id}: 患者主诉（模板）: {patient_chief_complaint}")
                    
                    # 第一轮对话：医生问候 + 患者主诉
                    output_dict['doctor'] = safe_strip(doctor_greeting)
                    output_dict_with_classification['doctor'] = safe_strip(doctor_greeting)
                    # 保存医生问候的reasoning
                    if greeting_reasoning:
                        output_dict['doctor_reasoning'] = greeting_reasoning
                        output_dict_with_classification['doctor_reasoning'] = greeting_reasoning
                    
                    output_dict['patient'] = safe_strip(patient_chief_complaint)
                    output_dict_with_classification['patient'] = safe_strip(patient_chief_complaint)
                    # 保存患者主诉的reasoning
                    patient_complaint_reasoning = getattr(pat, '_chief_complaint_reasoning', "")
                    if patient_complaint_reasoning:
                        output_dict['patient_reasoning'] = patient_complaint_reasoning
                        output_dict_with_classification['patient_reasoning'] = patient_complaint_reasoning
                    
                    # 更新对话历史
                    dialogue_history.append(f"医生: {doctor_greeting}")
                    dialogue_history.append(f"患者本人: {patient_chief_complaint}")
                    
                    # 保存第一轮对话
                    output_list.append(output_dict.copy())
                    output_list_with_classification.append(output_dict_with_classification.copy())
                    output_dict = {}
                    output_dict_with_classification = {}
                    
                    # 医生基于患者主诉开始正式问诊
                    doctor_response, current_topic, doctor_reasoning = doc.doctor_response_gen(
                        patient_chief_complaint, dialogue_history)
                    print(f"进程 {process_id}: 医生首个正式问题: {doctor_response}")
                    
                    current_topic = current_topic or '患者的精神状况'
                    
                    # 保存医生首个正式问题（作为第二轮对话的doctor部分）
                    output_dict['doctor'] = safe_strip(doctor_response)
                    output_dict_with_classification['doctor'] = safe_strip(doctor_response)
                    if doctor_reasoning:
                        output_dict['doctor_reasoning'] = doctor_reasoning
                        output_dict_with_classification['doctor_reasoning'] = doctor_reasoning
                
                else:
                    # === V1/V2版本：直接使用诊断树逻辑问诊 ===
                    # 医生首次响应（按诊断树逻辑）
                    # doctor_response_gen返回：(response, topic, reasoning)
                    doctor_response, _, doctor_reasoning = doc.doctor_response_gen(None, None)
                    
                    output_dict['doctor'] = safe_strip(doctor_response)
                    output_dict_with_classification['doctor'] = safe_strip(doctor_response)
                    # 保存医生初始问题的reasoning
                    if doctor_reasoning:
                        output_dict['doctor_reasoning'] = doctor_reasoning
                        output_dict_with_classification['doctor_reasoning'] = doctor_reasoning
                    print(f"进程 {process_id}: 医生初始问题: {doctor_response}")
                    
                    current_topic = '患者的精神状况'
                
                # 对话循环
                while True:
                    # 患者响应 - 所有版本现在统一返回4个值
                    # (response, cost, classification_info, reasoning)
                    patient_response, patient_cost, classification_info, patient_reasoning = pat.patient_response_gen(
                        current_topic, dialogue_history, doctor_response)
                    
                    # 正常版本
                    output_dict['patient'] = safe_strip(patient_response)
                    if patient_reasoning:
                        output_dict['patient_reasoning'] = patient_reasoning
                    
                    # 带分类版本
                    output_dict_with_classification['patient'] = safe_strip(patient_response)
                    if patient_reasoning:
                        output_dict_with_classification['patient_reasoning'] = patient_reasoning
                    if classification_info:
                        output_dict_with_classification['category'] = classification_info['category']
                        # 也保存分类步骤的reasoning（如果有）
                        if 'classification_reasoning' in classification_info and classification_info['classification_reasoning']:
                            output_dict_with_classification['classification_reasoning'] = classification_info['classification_reasoning']
                    
                    # 更新对话历史（base版本使用描述性标识，避免模型学习前缀模式）
                    dialogue_history.append(f"医生: {doctor_response}")
                    dialogue_history.append(f"患者本人: {patient_response}")
                    
                    output_list.append(output_dict.copy())
                    output_list_with_classification.append(output_dict_with_classification.copy())
                    output_dict = {}
                    output_dict_with_classification = {}
                    
                    # 医生后续响应
                    doctor_response, current_topic, doctor_reasoning = doc.doctor_response_gen(
                        patient_response, dialogue_history)
                    
                    # 累计cost（处理None值）
                    if patient_cost is not None:
                        total_cost += patient_cost
                    
                    if '诊断结束' in doctor_response:
                        output_dict = {'doctor': safe_strip(doctor_response)}
                        output_dict_with_classification = {'doctor': safe_strip(doctor_response)}
                        
                        # 保存诊断的reasoning（优先使用返回值，否则尝试获取）
                        if doctor_reasoning:
                            output_dict['doctor_reasoning'] = doctor_reasoning
                            output_dict_with_classification['doctor_reasoning'] = doctor_reasoning
                        else:
                            doctor_diagnosis_reasoning = doc.get_last_reasoning() if hasattr(doc, 'get_last_reasoning') else ""
                            if doctor_diagnosis_reasoning:
                                output_dict['doctor_reasoning'] = doctor_diagnosis_reasoning
                                output_dict_with_classification['doctor_reasoning'] = doctor_diagnosis_reasoning
                        
                        output_list.append(output_dict)
                        output_list_with_classification.append(output_dict_with_classification)
                        print(f"进程 {process_id}: 患者 {patient_template['patient_id']} 第 {i+1} 轮诊断结束")
                        break
                    else:
                        output_dict['doctor'] = safe_strip(doctor_response)
                        output_dict_with_classification['doctor'] = safe_strip(doctor_response)
                        
                        # 保存医生问题的reasoning（优先使用返回值）
                        if doctor_reasoning:
                            output_dict['doctor_reasoning'] = doctor_reasoning
                            output_dict_with_classification['doctor_reasoning'] = doctor_reasoning
                
                total_output_list.append({"conversation": output_list})
                total_output_list_with_classification.append({"conversation": output_list_with_classification})
                
                print(f"进程 {process_id}: 患者 {patient_template['patient_id']} 第 {i+1} 轮对话完成")
            
            # 取第一轮对话（如果有多轮，可以根据需要调整）
            conversation = total_output_list[0]['conversation'] if total_output_list else []
            
            # 提取label（ICD代码）
            label = extract_label_from_conversation(conversation)
            
            # 将结果存储到共享字典中
            shared_dict[patient_template['patient_id']] = {
                'patient_id': str(patient_template['patient_id']),
                'conversation': conversation,
                'label': label,
                'cost': total_cost,
                'process_id': process_id,
                'doctor_port': selected_doctor_port,
                'patient_port': selected_patient_port,
                # 如果需要带分类版本的数据，也保存
                'conversation_with_classification': total_output_list_with_classification[0]['conversation'] if output_dir_with_classification and total_output_list_with_classification else None
            }
            
            print(f"进程 {process_id}: 患者 {patient_template['patient_id']} 处理完成，费用: {total_cost}")
            print(f"进程 {process_id}: 提取的诊断标签: {label}")
            
        except Exception as e:
            print(f"进程 {process_id}: 处理患者 {patient_template['patient_id']} 时发生错误: {str(e)}")
            # 尝试获取已选择的端口，如果还未选择则使用第一个
            try:
                patient_id = patient_template.get('patient_id', 0)
                selected_doctor_port = self.doctor_ports[hash(str(patient_id)) % len(self.doctor_ports)]
                selected_patient_port = self.patient_ports[hash(str(patient_id)) % len(self.patient_ports)]
            except:
                selected_doctor_port = self.doctor_ports[0] if self.doctor_ports else None
                selected_patient_port = self.patient_ports[0] if self.patient_ports else None
                
            shared_dict[patient_template['patient_id']] = {
                'error': str(e),
                'process_id': process_id,
                'doctor_port': selected_doctor_port,  # 实际选择的doctor端口
                'doctor_ports_available': self.doctor_ports,  # 可用的doctor端口列表
                'patient_port': selected_patient_port,  # 实际选择的patient端口
                'patient_ports_available': self.patient_ports  # 可用的patient端口列表
            }
    
    def run_parallel_processing(self, patient_info_list, output_dir, output_dir_with_classification=None):
        """运行并行处理"""
        print(f"开始并行处理 {len(patient_info_list)} 个患者")
        print(f"最大并行进程数: {self.max_parallel_processes} (基于端口配置)")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        if output_dir_with_classification:
            os.makedirs(output_dir_with_classification, exist_ok=True)
            print(f"将生成两个版本: 正常版本 -> {output_dir}, 带分类版本 -> {output_dir_with_classification}")
        else:
            print(f"只生成正常版本 -> {output_dir}")
        
        # 创建共享字典用于收集结果
        manager = Manager()
        shared_dict = manager.dict()
        
        # 创建进程池
        processes = []
        patient_queue = list(patient_info_list)
        
        # 启动初始进程
        for i in range(min(self.max_parallel_processes, len(patient_queue))):
            if patient_queue:
                patient = patient_queue.pop(0)
                
                process = Process(
                    target=self.process_single_patient,
                    args=(patient, shared_dict, i, output_dir, output_dir_with_classification)
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
                        patient = patient_queue.pop(0)
                        new_process = Process(
                            target=self.process_single_patient,
                            args=(patient, shared_dict, process_id, output_dir, output_dir_with_classification)
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
            process.join()
        
        # 收集结果统计并整理数据
        total_cost = 0
        successful_patients = 0
        error_patients = 0
        
        # 用于存储所有患者的对话数据
        all_conversations = []
        all_conversations_with_classification = []
        
        for patient_id, result in shared_dict.items():
            if 'error' in result:
                print(f"患者 {patient_id} 处理失败: {result['error']}")
                error_patients += 1
            else:
                total_cost += result['cost']
                successful_patients += 1
                
                # 收集正常版本的数据
                all_conversations.append({
                    'patient_id': result['patient_id'],
                    'conversation': result['conversation'],
                    'label': result['label']
                })
                
                # 收集带分类版本的数据（如果有）
                if result.get('conversation_with_classification'):
                    all_conversations_with_classification.append({
                        'patient_id': result['patient_id'],
                        'conversation': result['conversation_with_classification'],
                        'label': result['label']
                    })
        
        # 按patient_id排序（转换为整数排序）
        all_conversations.sort(key=lambda x: int(x['patient_id']))
        
        # 保存正常版本到单一JSON文件（使用可配置的文件名）
        output_file_path = os.path.join(output_dir, self.output_filename)
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(all_conversations, f, indent=2, ensure_ascii=False)
        print(f"\n所有对话已保存到: {output_file_path}")
        
        # 保存带分类版本（如果有，使用可配置的文件名）
        if output_dir_with_classification and all_conversations_with_classification:
            all_conversations_with_classification.sort(key=lambda x: int(x['patient_id']))
            output_file_path_with_classification = os.path.join(
                output_dir_with_classification, self.output_filename_with_classification)
            with open(output_file_path_with_classification, 'w', encoding='utf-8') as f:
                json.dump(all_conversations_with_classification, f, indent=2, ensure_ascii=False)
            print(f"带分类版本已保存到: {output_file_path_with_classification}")
        
        print(f"\n=== 并行处理完成 ===")
        print(f"成功处理患者数: {successful_patients}")
        print(f"处理失败患者数: {error_patients}")
        print(f"总费用: {total_cost}")
        
        return total_cost, successful_patients, error_patients


def create_port_specific_model_name(base_model_name, port):
    """创建包含端口信息的模型名称"""
    return f"{base_model_name}:{port}"
