/**
 * 诊断详情管理 - 处理医生备注的自动保存
 */

// 全局变量
let doctorNotesAutosaveTimer = null;
let lastSavedDoctorNotes = "";
let currentDiagnosisData = {};

/**
 * 初始化医生备注自动保存
 */
function initializeDoctorNotesAutosave() {
    const doctorNotesField = document.getElementById('doctorNotesField');
    if (!doctorNotesField) return;
    
    // 监听输入事件
    doctorNotesField.addEventListener('input', function() {
        // 清除之前的定时器
        if (doctorNotesAutosaveTimer) {
            clearTimeout(doctorNotesAutosaveTimer);
        }
        
        // 更新状态为"保存中"
        updateAutosaveStatus('saving');
        
        // 设置新的定时器（2秒后保存）
        doctorNotesAutosaveTimer = setTimeout(() => {
            saveDoctorNotes();
        }, 2000);
    });
    
    console.log('医生备注自动保存已初始化');
}

/**
 * 保存医生备注和结构化详情
 */
async function saveDoctorNotes() {
    const doctorNotesField = document.getElementById('doctorNotesField');
    if (!doctorNotesField) return;
    
    const notes = doctorNotesField.value.trim();
    
    // 如果内容没有变化，不保存
    if (notes === lastSavedDoctorNotes) {
        updateAutosaveStatus('saved');
        return;
    }
    
    try {
        updateAutosaveStatus('saving');
        
        // 准备保存数据
        const saveData = {
            patient_id: selectedDiagnosisPatient ? selectedDiagnosisPatient.patient_id : null,
            diagnosis_id: currentDiagnosisData.diagnosis_id || null,
            conversation: currentPatientConversation || diagnosisConversationRecords || [],
            diagnosis_reason: document.getElementById('diagnosisReasonField')?.value || '',
            diagnosis_conclusion: document.getElementById('diagnosisResultField')?.value || '',
            doctor_notes: notes,
        };
        
        const response = await fetch('/api/diagnosis/save-details', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(saveData)
        });
        
        const data = await response.json();
        
        if (data.success) {
            lastSavedDoctorNotes = notes;
            currentDiagnosisData.diagnosis_id = data.data.diagnosis_id;
            updateAutosaveStatus('saved');
            console.log('医生备注已自动保存:', data.data.timestamp);
        } else {
            throw new Error(data.error || '保存失败');
        }
    } catch (error) {
        console.error('自动保存失败:', error);
        updateAutosaveStatus('error');
        setTimeout(() => updateAutosaveStatus('unsaved'), 3000);
    }
}

/**
 * 更新自动保存状态显示
 */
function updateAutosaveStatus(status) {
    const indicator = document.getElementById('doctorNotesAutosaveStatus');
    if (!indicator) return;
    
    // 移除所有状态类
    indicator.classList.remove('saved', 'saving', 'error');
    
    switch(status) {
        case 'saved':
            indicator.textContent = '已保存';
            indicator.classList.add('saved');
            break;
        case 'saving':
            indicator.textContent = '保存中...';
            indicator.classList.add('saving');
            break;
        case 'error':
            indicator.textContent = '保存失败';
            indicator.classList.add('error');
            break;
        default:
            indicator.textContent = '未保存';
    }
}

/**
 * HTML转义，防止XSS
 */
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * 清空所有详情显示
 */
function clearAllDetails() {
    // 清空医生备注
    const doctorNotesField = document.getElementById('doctorNotesField');
    if (doctorNotesField) {
        doctorNotesField.value = '';
        lastSavedDoctorNotes = '';
    }
    
    updateAutosaveStatus('unsaved');
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    initializeDoctorNotesAutosave();
    console.log('诊断详情模块已加载');
});

