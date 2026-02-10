/**
 * Patient Agent 测试界面 - 前端JavaScript
 * 
 * 主要功能：
 * 1. 患者列表加载和搜索
 * 2. 创建对话会话
 * 3. 实时对话交互
 * 4. 诊断结果生成
 * 5. 会话管理
 */

// 全局变量
let currentPatients = [];
let selectedPatient = null;
let currentSession = null;
let isLoading = false;
let diagnosisList = [];  // 存储诊断列表
let diagnosisGroupedData = [];  // 存储分组的诊断数据
let suggestedQuestions = [];
let isAutoDiagnosing = false;
let isFetchingRecommendation = false;
let lastAutoDiagnosis = null;
let currentUser = null;
let availableUsers = [];
let appInitialized = false;
let activeModule = 'diagnosis';  // 默认打开EverDiagnosis模块
let psychosisSession = null;
let psychosisSelectedPatient = null;
let isPsychosisLoading = false;
let isGeneratingPsychosisQuestion = false;

// ===== 功能开关配置 =====
const FEATURE_FLAGS = {
    ENABLE_RECOMMENDED_QUESTIONS: false,  // 推荐问题功能开关（默认关闭）
};
let psychosisLatestQuestion = null;
let diagnosisConversationRecords = [];
let diagnosisSheetInfo = '';
let isDiagnosisGenerating = false;
let diagnosisPatients = [];
let selectedDiagnosisPatient = null;
let currentPatientConversation = [];
// 医生相关变量
let currentDoctors = [];
let psychosisSelectedDoctor = null;
let psychosisConversationActive = false;
// 诊断版本管理变量
let aiGeneratedDiagnosisReason = '';
let aiGeneratedDiagnosisConclusion = '';
let aiGeneratedTimestamp = '';
let currentDiagnosisId = null;
let autoSaveTimeout = null;
let isAutoSaving = false;
let hasUnsavedChanges = false;

// API 基础配置
const API_BASE = '/api';
const API_ENDPOINTS = {
    health: `${API_BASE}/health`,
    patients: `${API_BASE}/patients`,
    doctors: `${API_BASE}/doctors`,
    sessions: `${API_BASE}/sessions`,
    diagnoses: `${API_BASE}/diagnoses`,
    chat: (sessionId) => `${API_BASE}/sessions/${sessionId}/chat`,
    diagnosis: (sessionId) => `${API_BASE}/sessions/${sessionId}/diagnosis`,
    evaluation: (sessionId) => `${API_BASE}/sessions/${sessionId}/evaluation`,
    sessionInfo: (sessionId) => `${API_BASE}/sessions/${sessionId}`,
    autoDiagnosis: (sessionId) => `${API_BASE}/sessions/${sessionId}/auto-diagnosis`,
    recommendQuestion: (sessionId) => `${API_BASE}/sessions/${sessionId}/recommend-question`,
    sessionEvent: (sessionId) => `${API_BASE}/sessions/${sessionId}/events`,
    conversationLog: (sessionId) => `${API_BASE}/sessions/${sessionId}/conversation-log`,
    authUsers: `${API_BASE}/auth/users`,
    authLogin: `${API_BASE}/auth/login`,
    authRegister: `${API_BASE}/auth/register`,
    doctorQuestion: (sessionId) => `${API_BASE}/sessions/${sessionId}/doctor-question`,
    patientReply: (sessionId) => `${API_BASE}/sessions/${sessionId}/patient-reply`,
    diagnosisImport: `${API_BASE}/diagnosis/import-excel`,
    diagnosisImportJson: `${API_BASE}/diagnosis/import-json`,
    diagnosisGenerate: `${API_BASE}/diagnosis/generate`,
    diagnosisParseConversation: `${API_BASE}/diagnosis/parse-conversation`,
    diagnosisSave: `${API_BASE}/diagnosis/save`,
    diagnosisLoadDefault: `${API_BASE}/diagnosis/load-default-data`,
    diagnosisLoadAnnotation: `${API_BASE}/diagnosis/load-annotation`,
    diagnosisAnnotatedPatients: `${API_BASE}/diagnosis/annotated-patients`,
    diagnosisPatientConversation: `${API_BASE}/diagnosis/patient-conversation`
};

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    console.log('Patient Agent 测试界面加载中...');
    initializeModuleNavigation();
    initializeAuth();
    initializePageUnloadWarning();
    updateVersionBadges(); // 初始化时更新版本显示
});

// 初始化页面离开警告
function initializePageUnloadWarning() {
    window.addEventListener('beforeunload', function(event) {
        // 如果有活跃的医生对话，显示警告
        if (psychosisConversationActive && psychosisSelectedDoctor) {
            const message = `当前正在与${psychosisSelectedDoctor.name}进行对话，离开页面将会中断对话。对话记录将被保存。`;
            event.preventDefault();
            event.returnValue = message; // 标准方式
            return message; // 兼容某些浏览器
        }
    });
}

async function initializeAuth() {
    bindAuthFormEvents();
    await loadUserOptions();
    updateUserDisplay();
    toggleAuthMode('login');
    showAuthOverlay();
}

function initializeModuleNavigation() {
    const navItems = document.querySelectorAll('.app-navigation .nav-item');
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const targetModule = item.dataset.module;
            if (!targetModule || targetModule === activeModule) {
                return;
            }
            switchModule(targetModule);
        });
    });
    updateModuleVisibility();
}

function switchModule(module) {
    // 如果当前在psychiatrist模块且有活跃会话，检查是否完成评测和诊断
    if (activeModule === 'psychiatrist' && currentSession && module !== 'psychiatrist') {
        // 使用异步检查，但由于这是事件处理函数，我们需要阻止立即切换
        checkAndPromptCompletion('切换页面').then(shouldProceed => {
            if (shouldProceed) {
                // 用户确认切换或已完成评测诊断
                performModuleSwitch(module);
            }
            // 否则不做任何操作，留在当前页面
        });
        return;
    }
    
    // 如果当前在psychosis模块且有活跃对话，询问用户是否确认切换
    if (activeModule === 'psychosis' && psychosisConversationActive && module !== 'psychosis') {
        const doctorName = psychosisSelectedDoctor ? psychosisSelectedDoctor.name : '医生';
        const confirmSwitch = confirm(
            `当前正在与${doctorName}进行对话。\n\n` +
            `离开此页面将会中断当前对话，对话记录将被保存。\n\n` +
            `确定要离开吗？`
        );
        
        if (!confirmSwitch) {
            return; // 用户取消切换
        }
        
        // 用户确认切换，结束当前对话
        endPsychosisConversation();
    }
    
    performModuleSwitch(module);
}

function performModuleSwitch(module) {
    activeModule = module;
    updateModuleVisibility();
}

function updateModuleVisibility() {
    const navItems = document.querySelectorAll('.app-navigation .nav-item');
    navItems.forEach(item => {
        item.classList.toggle('active', item.dataset.module === activeModule);
    });

    const modules = document.querySelectorAll('.module-container .app-module');
    modules.forEach(section => {
        const isActive = section.dataset.module === activeModule;
        section.classList.toggle('active', isActive);
        section.style.display = isActive ? 'flex' : 'none';
    });

    updatePsychosisControls();
    updateDiagnosisButtons();
    
    // 如果切换到诊断模块，且用户已登录，自动加载默认数据
    if (activeModule === 'diagnosis' && currentUser) {
        loadDefaultDiagnosisData();
    }
}

function bindAuthFormEvents() {
    const tabs = document.querySelectorAll('.auth-tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            toggleAuthMode(tab.dataset.mode);
        });
    });

    const loginForm = document.getElementById('loginForm');
    const registerForm = document.getElementById('registerForm');
    const loginUserSelect = document.getElementById('loginUserSelect');

    loginForm.addEventListener('submit', handleLoginSubmit);
    registerForm.addEventListener('submit', handleRegisterSubmit);

    loginUserSelect.addEventListener('change', () => {
        const selected = loginUserSelect.value;
        const usernameInput = document.getElementById('loginUsername');
        usernameInput.value = selected;
    });
}

function toggleAuthMode(mode) {
    const loginForm = document.getElementById('loginForm');
    const registerForm = document.getElementById('registerForm');
    const tabs = document.querySelectorAll('.auth-tab');

    tabs.forEach(tab => {
        tab.classList.toggle('active', tab.dataset.mode === mode);
    });

    if (mode === 'register') {
        loginForm.style.display = 'none';
        registerForm.style.display = 'block';
    } else {
        registerForm.style.display = 'none';
        loginForm.style.display = 'block';
    }

    setAuthMessage('');
}

function showAuthOverlay() {
    const overlay = document.getElementById('authOverlay');
    overlay.style.display = 'flex';
}

function hideAuthOverlay() {
    const overlay = document.getElementById('authOverlay');
    overlay.style.display = 'none';
    setAuthMessage('');
}

function showInstructionOverlay(event) {
    if (event) {
        event.preventDefault();
    }
    const modal = document.getElementById('instructionOverlay');
    if (modal) {
        modal.style.display = 'block';
    }
}

function closeInstructionOverlay() {
    const modal = document.getElementById('instructionOverlay');
    if (modal) {
        modal.style.display = 'none';
    }
}

function setAuthMessage(message, type = 'info') {
    const messageEl = document.getElementById('authMessage');
    if (!messageEl) return;
    messageEl.textContent = message;
    messageEl.className = `auth-message ${type}`;
}

async function loadUserOptions() {
    try {
        const response = await fetch(API_ENDPOINTS.authUsers);
        const data = await response.json();
        if (data.success) {
            availableUsers = Array.isArray(data.data) ? data.data : [];
        } else {
            availableUsers = [];
        }
    } catch (error) {
        console.warn('加载用户列表失败:', error);
        availableUsers = [];
    }

    populateUserSelect();
}

function populateUserSelect() {
    const loginUserSelect = document.getElementById('loginUserSelect');
    if (!loginUserSelect) return;

    loginUserSelect.innerHTML = '<option value="">选择已有用户...</option>';
    availableUsers.forEach(user => {
        const option = document.createElement('option');
        option.value = user;
        option.textContent = user;
        loginUserSelect.appendChild(option);
    });
}

async function handleLoginSubmit(event) {
    event.preventDefault();
    const loginUserSelect = document.getElementById('loginUserSelect');
    const usernameInput = document.getElementById('loginUsername');
    const passwordInput = document.getElementById('loginPassword');

    const username = (loginUserSelect.value || usernameInput.value || '').trim();
    const password = passwordInput.value || '';

    if (!username || !password) {
        setAuthMessage('请输入用户名和密码', 'error');
        return;
    }

    try {
        const response = await fetch(API_ENDPOINTS.authLogin, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ username, password })
        });

        const data = await response.json();
        if (!response.ok || !data.success) {
            throw new Error(data.error || '登录失败');
        }

        completeLogin(username);
        passwordInput.value = '';
        usernameInput.value = '';
        loginUserSelect.value = '';
        setAuthMessage('登录成功', 'success');
        hideAuthOverlay();
    } catch (error) {
        console.error('登录失败:', error);
        setAuthMessage(error.message || '登录失败', 'error');
    }
}

async function handleRegisterSubmit(event) {
    event.preventDefault();
    const usernameInput = document.getElementById('registerUsername');
    const passwordInput = document.getElementById('registerPassword');
    const repeatInput = document.getElementById('registerPasswordConfirm');

    const username = (usernameInput.value || '').trim();
    const password = passwordInput.value || '';
    const repeat = repeatInput.value || '';

    if (!username || !password) {
        setAuthMessage('请输入用户名和密码', 'error');
        return;
    }

    if (password !== repeat) {
        setAuthMessage('两次输入的密码不一致', 'error');
        return;
    }

    try {
        const response = await fetch(API_ENDPOINTS.authRegister, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ username, password })
        });

        const data = await response.json();
        if (!response.ok || !data.success) {
            throw new Error(data.error || '注册失败');
        }

        setAuthMessage('注册成功，请使用新账号登录', 'success');
        usernameInput.value = '';
        passwordInput.value = '';
        repeatInput.value = '';

        await loadUserOptions();
        toggleAuthMode('login');
        const loginUserSelect = document.getElementById('loginUserSelect');
        const loginUsernameInput = document.getElementById('loginUsername');
        if (loginUserSelect) {
            loginUserSelect.value = username;
        }
        if (loginUsernameInput) {
            loginUsernameInput.value = username;
        }
    } catch (error) {
        console.error('注册失败:', error);
        setAuthMessage(error.message || '注册失败', 'error');
    }
}

function completeLogin(username) {
    currentUser = username;
    updateUserDisplay();
    if (!appInitialized) {
        initializeApp();
    } else {
        showAlert('success', `欢迎回来，${username}`);
        // 如果已初始化但当前在诊断模块，加载默认数据
        if (activeModule === 'diagnosis') {
            loadDefaultDiagnosisData();
        }
    }
}

function updateUserDisplay() {
    const userDisplay = document.getElementById('currentUserDisplay');
    if (userDisplay) {
        userDisplay.textContent = currentUser || '未登录';
    }
}

/**
 * 初始化应用
 */
async function initializeApp() {
    if (appInitialized) {
        return;
    }

    if (!currentUser) {
        console.warn('未登录用户无法初始化应用');
        return;
    }

    try {
        // 检查后端服务状态
        await checkBackendHealth();
        
        // 加载患者列表
        await loadPatients();

        // 加载医生列表
        await loadDoctors();

        // 加载诊断列表
        await loadDiagnoses();

        // 绑定事件监听器
        bindEventListeners();
        updateRecommendButtonState();
        updatePsychosisControls();
        updateDiagnosisButtons();
        updateUserDisplay();
        appInitialized = true;
        
        // 如果当前模块是诊断模块，加载默认数据
        if (activeModule === 'diagnosis') {
            await loadDefaultDiagnosisData();
        }
        
        console.log('应用初始化完成');
    } catch (error) {
        console.error('应用初始化失败:', error);
        showAlert('error', '系统初始化失败，请检查后端服务是否正常运行');
    }
}

/**
 * 检查后端健康状态
 */
async function checkBackendHealth() {
    try {
        const response = await fetch(API_ENDPOINTS.health);
        const data = await response.json();
        
        if (data.status === 'healthy') {
            console.log('后端服务正常:', data);
            showAlert('success', `系统已连接，共加载 ${data.config.patients_loaded} 个患者数据`);
        } else {
            throw new Error('后端服务状态异常');
        }
    } catch (error) {
        console.error('后端健康检查失败:', error);
        throw new Error('无法连接到后端服务');
    }
}

/**
 * 加载患者列表
 */
async function loadPatients() {
    try {
        showAlert('info', '正在加载患者数据...');
        
        let url = API_ENDPOINTS.patients;
        if (currentUser) {
            url += `?username=${encodeURIComponent(currentUser)}`;
        }
        
        const response = await fetch(url);
        const data = await response.json();
        
        if (data.success) {
            currentPatients = data.data;
            renderPatientList(currentPatients);
            showAlert('success', `成功加载 ${data.total} 位患者数据`);
            const countDisplay = document.getElementById('patientCountDisplay');
            if (countDisplay) {
                countDisplay.textContent = currentPatients.length;
            }
        } else {
            throw new Error(data.error || '加载患者数据失败');
        }
    } catch (error) {
        console.error('加载患者列表失败:', error);
        showAlert('error', '加载患者数据失败: ' + error.message);
        const countDisplay = document.getElementById('patientCountDisplay');
        if (countDisplay) {
            countDisplay.textContent = '--';
        }
    }
}

/**
 * 加载医生列表
 */
async function loadDoctors() {
    try {
        showAlert('info', '正在加载医生数据...');
        
        const response = await fetch(API_ENDPOINTS.doctors);
        const data = await response.json();
        
        if (data.success) {
            currentDoctors = data.data;
            renderDoctorList(currentDoctors);
            showAlert('success', `成功加载 ${data.count} 位医生数据`);
            const countDisplay = document.getElementById('psychosisDoctorCountDisplay');
            if (countDisplay) {
                countDisplay.textContent = currentDoctors.length;
            }
        } else {
            throw new Error(data.error || '加载医生数据失败');
        }
    } catch (error) {
        console.error('加载医生列表失败:', error);
        showAlert('error', '加载医生数据失败: ' + error.message);
        const countDisplay = document.getElementById('psychosisDoctorCountDisplay');
        if (countDisplay) {
            countDisplay.textContent = '--';
        }
    }
}

/**
 * 加载诊断列表
 */
async function loadDiagnoses() {
    try {
        // 加载分组的诊断数据
        const groupedResponse = await fetch(`${API_ENDPOINTS.diagnoses}/grouped`);
        const groupedData = await groupedResponse.json();
        
        if (groupedData.success) {
            diagnosisGroupedData = groupedData.data;
            console.log(`成功加载 ${groupedData.total_categories} 个诊断大类`);
        }
        
        // 同时加载扁平化的诊断列表（用于向后兼容）
        const response = await fetch(API_ENDPOINTS.diagnoses);
        const data = await response.json();
        
        if (data.success) {
            diagnosisList = data.data;
            console.log(`成功加载 ${data.total} 种诊断类型`);
        } else {
            throw new Error(data.error || '加载诊断数据失败');
        }
    } catch (error) {
        console.error('加载诊断列表失败:', error);
        showAlert('error', '加载诊断数据失败: ' + error.message);
    }
}

/**
 * 渲染患者列表
 */
function renderPatientList(patients) {
    renderPatientListForModule(patients, 'psychiatrist');
}

/**
 * 渲染医生列表
 */
function renderDoctorList(doctors) {
    const container = document.getElementById('psychosisDoctorList');
    if (!container) {
        return;
    }

    const countElement = document.getElementById('psychosisDoctorCountDisplay');

    if (doctors.length === 0) {
        container.innerHTML = '<div class="alert alert-info">未找到匹配的医生</div>';
        if (countElement) {
            countElement.textContent = `0/${currentDoctors.length}`;
        }
        return;
    }

    const itemsHtml = doctors.map(doctor => `
        <div class="patient-item" onclick="selectPsychosisDoctor(${doctor.doctor_id})" data-doctor-id="${doctor.doctor_id}">
            <div class="patient-id">${doctor.name}</div>
            <div class="patient-info">${doctor.age} ${doctor.gender}</div>
        </div>
    `).join('');
    // <div class="patient-info">专长: ${doctor.special}</div>
    // <div class="patient-complaint">沟通: ${doctor.commu} | 共情: ${doctor.empathy}</div>

    container.innerHTML = itemsHtml;

    if (countElement) {
        countElement.textContent = doctors.length;
    }

    // 恢复选中状态
    if (psychosisSelectedDoctor !== null) {
        const selectedItem = container.querySelector(`[data-doctor-id="${psychosisSelectedDoctor.doctor_id}"]`);
        if (selectedItem) {
            selectedItem.classList.add('selected');
        }
    }
}

function renderPatientListForModule(patients, module) {
    // psychosis模块现在使用医生列表，不再使用患者列表
    if (module === 'psychosis') {
        return;
    }
    
    const containerId = 'patientList';
    const container = document.getElementById(containerId);
    if (!container) {
        return;
    }

    const countElementId = 'patientCountDisplay';
    const countElement = document.getElementById(countElementId);

    if (patients.length === 0) {
        container.innerHTML = '<div class="alert alert-info">未找到匹配的患者</div>';
        if (countElement) {
            countElement.textContent = `0/${currentPatients.length}`;
        }
        return;
    }

    const handler = 'selectPatient';
    const itemsHtml = patients.map(patient => `
        <div class="patient-item" onclick="${handler}(${patient.patient_id})" data-patient-id="${patient.patient_id}">
            <div class="patient-id">
                患者 #${patient.patient_id}
                ${patient.completed ? '<span style="font-size:12px; color:#28a745; margin-left:5px;">【已完成】</span>' : ''}
                ${patient.annotated ? '<span style="font-size:12px; color:#007bff; margin-left:5px;">【已标注】</span>' : ''}
            </div>
            <div class="patient-info">${patient.age}岁 ${patient.gender}性</div>
        </div>
    `).join('');

    container.innerHTML = itemsHtml;

    if (countElement) {
        countElement.textContent = patients.length;
    }

    const selectedId = selectedPatient ? selectedPatient.patient_id : null;
    if (selectedId !== null) {
        const selectedItem = container.querySelector(`[data-patient-id="${selectedId}"]`);
        if (selectedItem) {
            selectedItem.classList.add('selected');
        }
    }
}

/**
 * 绑定事件监听器
 */
function bindEventListeners() {
    // 搜索框事件
    const searchInput = document.getElementById('searchInput');
    if (searchInput) {
        searchInput.addEventListener('input', handleSearch);
    }

    const psychosisSearchInput = document.getElementById('psychosisSearchInput');
    if (psychosisSearchInput) {
        psychosisSearchInput.addEventListener('input', handleSearch);
    }

    // 消息输入框事件
    const messageInput = document.getElementById('messageInput');
    if (messageInput) {
        messageInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    }

    const psychosisReplyInput = document.getElementById('psychosisReplyInput');
    if (psychosisReplyInput) {
        psychosisReplyInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendPsychosisReply();
            }
        });
    }

    // 推荐问题按钮（受功能开关控制）
    const recommendButton = document.getElementById('recommendButton');
    if (recommendButton && FEATURE_FLAGS.ENABLE_RECOMMENDED_QUESTIONS) {
        recommendButton.style.display = 'inline-block';
        recommendButton.addEventListener('click', fetchRecommendedQuestion);
    } else if (recommendButton) {
        recommendButton.style.display = 'none';
    }

    const psychosisSendButton = document.getElementById('psychosisSendButton');
    if (psychosisSendButton) {
        psychosisSendButton.addEventListener('click', sendPsychosisReply);
    }

    const psychosisNextQuestionButton = document.getElementById('psychosisNextQuestionButton');
    if (psychosisNextQuestionButton) {
        psychosisNextQuestionButton.addEventListener('click', () => requestPsychosisQuestion(false, true));
    }


    // 诊断表单提交事件
    const diagnosisForm = document.getElementById('diagnosisForm');
    if (diagnosisForm) {
        diagnosisForm.addEventListener('submit', handleDiagnosisSubmit);
    }

    const diagnosisFileInput = document.getElementById('diagnosisFileInput');
    if (diagnosisFileInput) {
        diagnosisFileInput.addEventListener('change', handleDiagnosisFileChange);
    }

    const diagnosisAutoButton = document.getElementById('diagnosisAutoButton');
    if (diagnosisAutoButton) {
        diagnosisAutoButton.addEventListener('click', handleDiagnosisAutoGenerate);
    }

    const diagnosisClearButton = document.getElementById('diagnosisClearButton');
    if (diagnosisClearButton) {
        diagnosisClearButton.addEventListener('click', clearDiagnosisFields);
    }

    const diagnosisClearRecordsButton = document.getElementById('diagnosisClearRecordsButton');
    if (diagnosisClearRecordsButton) {
        diagnosisClearRecordsButton.addEventListener('click', clearDiagnosisRecords);
    }

    const diagnosisSaveButton = document.getElementById('diagnosisSaveButton');
    if (diagnosisSaveButton) {
        diagnosisSaveButton.addEventListener('click', saveDiagnosisResult);
    }

    // 添加诊断字段的自动保存监听
    const diagnosisReasonField = document.getElementById('diagnosisReasonField');
    if (diagnosisReasonField) {
        diagnosisReasonField.addEventListener('input', handleDiagnosisFieldChange);
    }

    const diagnosisResultField = document.getElementById('diagnosisResultField');
    if (diagnosisResultField) {
        diagnosisResultField.addEventListener('input', handleDiagnosisFieldChange);
        // 当失去焦点时，自动格式化ICD-10代码
        diagnosisResultField.addEventListener('blur', async function() {
            if (window.icd10Formatter && window.icd10Formatter.autoFormatDiagnosisResult) {
                await window.icd10Formatter.autoFormatDiagnosisResult();
            }
        });
    }
}

/**
 * 处理搜索
 */
function handleSearch(e) {
    const module = e.target.dataset.module || 'psychiatrist';
    const searchTerm = e.target.value.toLowerCase().trim();

    if (!searchTerm) {
        if (module === 'psychosis') {
            renderDoctorList(currentDoctors);
        } else {
            renderPatientListForModule(currentPatients, module);
            const countDisplay = document.getElementById('patientCountDisplay');
            if (countDisplay) {
                countDisplay.textContent = currentPatients.length;
            }
        }
        return;
    }

    if (module === 'psychosis') {
        // 搜索医生
        const filteredDoctors = currentDoctors.filter(doctor => {
            return (
                doctor.name.toLowerCase().includes(searchTerm) ||
                doctor.gender.toLowerCase().includes(searchTerm) ||
                doctor.special.toLowerCase().includes(searchTerm) ||
                doctor.commu.toLowerCase().includes(searchTerm) ||
                doctor.empathy.toLowerCase().includes(searchTerm)
            );
        });
        
        renderDoctorList(filteredDoctors);
    } else {
        // 搜索患者 (移除科室搜索)
        const filteredPatients = currentPatients.filter(patient => {
            return (
                patient.patient_id.toString().includes(searchTerm) ||
                patient.gender.toLowerCase().includes(searchTerm) ||
                patient.chief_complaint.toLowerCase().includes(searchTerm)
            );
        });

        renderPatientListForModule(filteredPatients, module);
        const countDisplay = document.getElementById('patientCountDisplay');
        if (countDisplay) {
            countDisplay.textContent = `${filteredPatients.length}/${currentPatients.length}`;
        }
    }
}

/**
 * 检查是否完成评测和诊断，如果未完成则提示用户
 * @param {string} action - 触发检查的动作描述（如"切换患者"、"切换页面"）
 * @returns {Promise<boolean>} - 返回是否应该继续操作
 */
async function checkAndPromptCompletion(action) {
    // 如果没有活跃会话或对话轮数为0，直接允许操作
    if (!currentSession || !currentSession.session_info || currentSession.session_info.conversation_count === 0) {
        return true;
    }
    
    // 检查是否已完成评测和诊断
    const hasEvaluation = isEvaluationSubmitted;
    const hasDiagnosis = isDiagnosisSaved;
    
    // 如果都已完成，直接允许操作
    if (hasEvaluation && hasDiagnosis) {
        return true;
    }
    
    // 构建提示信息
    let missingItems = [];
    if (!hasEvaluation) {
        missingItems.push('患者评测');
    }
    if (!hasDiagnosis) {
        missingItems.push('诊断结果');
    }
    
    const message = `您尚未完成：${missingItems.join('、')}\n\n建议您先完成这些内容再${action}。\n\n点击"确定"留在当前页面完成，点击"取消"直接${action}。`;
    
    // 显示确认对话框
    const shouldStay = confirm(message);
    
    if (shouldStay) {
        // 用户选择留下完成任务
        // 如果还没有评测，优先打开评测面板
        if (!hasEvaluation) {
            showEvaluationPanel();
        } else if (!hasDiagnosis) {
            showDiagnosisPanel();
        }
        return false; // 不允许继续操作
    }
    
    // 用户选择直接离开
    return true;
}

/**
 * 选择患者
 */
async function selectPatient(patientId) {
    if (isLoading) return;

    // 检查是否有未完成的评测和诊断
    if (currentSession && selectedPatient && selectedPatient.patient_id !== patientId) {
        const shouldProceed = await checkAndPromptCompletion('切换患者');
        if (!shouldProceed) {
            return; // 用户选择留在当前页面
        }
    }

    try {
        isLoading = true;

        // 更新UI选中状态
        document.querySelectorAll('#module-psychiatrist .patient-item').forEach(item => {
            item.classList.remove('selected');
        });
        const targetItem = document.querySelector(`#module-psychiatrist [data-patient-id="${patientId}"]`);
        if (targetItem) {
            targetItem.classList.add('selected');
        }
        
        // 获取患者详细信息
        const response = await fetch(`${API_ENDPOINTS.patients}/${patientId}`);
        const data = await response.json();
        
        if (!data.success) {
            throw new Error(data.error);
        }
        
        selectedPatient = data.data;
        
        // 创建新会话
        await createSession(patientId);
        
    } catch (error) {
        console.error('选择患者失败:', error);
        showAlert('error', '选择患者失败: ' + error.message);
    } finally {
        isLoading = false;
    }
}

async function selectPsychosisDoctor(doctorId) {
    if (isPsychosisLoading) return;

    // 检查是否有活跃对话
    if (psychosisConversationActive && psychosisSelectedDoctor && psychosisSelectedDoctor.doctor_id !== doctorId) {
        const confirmSwitch = confirm(
            `当前正在与${psychosisSelectedDoctor.name}进行对话。\n\n` +
            `切换到其他医生将会中断当前对话，对话记录将被保存。\n\n` +
            `确定要切换医生吗？`
        );
        
        if (!confirmSwitch) {
            return; // 用户取消切换
        }
        
        // 用户确认切换，结束当前对话
        endPsychosisConversation();
    }

    try {
        isPsychosisLoading = true;

        document.querySelectorAll('#module-psychosis .patient-item').forEach(item => {
            item.classList.remove('selected');
        });
        const targetItem = document.querySelector(`#module-psychosis [data-doctor-id="${doctorId}"]`);
        if (targetItem) {
            targetItem.classList.add('selected');
        }

        // 从当前医生列表中找到选中的医生
        psychosisSelectedDoctor = currentDoctors.find(doctor => doctor.doctor_id === doctorId);
        if (!psychosisSelectedDoctor) {
            throw new Error('未找到选中的医生');
        }

        // 立即显示对话界面，不等待会话创建完成
        showPsychosisChatInterfaceImmediately();
        
        // 异步创建会话
        createPsychosisSession(doctorId);

    } catch (error) {
        console.error('选择医生失败:', error);
        showAlert('error', '选择医生失败: ' + error.message);
    } finally {
        isPsychosisLoading = false;
        updatePsychosisControls();
    }
}

function showPsychosisChatInterfaceImmediately() {
    // 立即显示对话界面，不等待会话创建
    clearPsychosisMessages();
    showPsychosisChatInterface();
    
    // 显示医生信息（使用选中的医生信息）
    const detailElement = document.getElementById('psychosisPatientDetail');
    if (detailElement && psychosisSelectedDoctor) {
        detailElement.innerHTML = `
            <h4>${psychosisSelectedDoctor.name} - ${psychosisSelectedDoctor.gender}，${psychosisSelectedDoctor.age}</h4>
            <div class="detail-row">
                <div class="detail-item">
                    <span class="detail-label">正在初始化会话...</span>
                </div>
            </div>
        `;
    }
    
    // 显示初始状态消息
    addPsychosisMessage('system', '医生正在准备首个问题，请稍候...');
    updatePsychosisStatus('正在连接医生...');
    updatePsychosisControls();
}

function endPsychosisConversation() {
    // 结束当前的医生对话
    psychosisConversationActive = false;
    
    if (psychosisSession) {
        // 记录对话结束事件
        console.log(`对话结束: 会话ID ${psychosisSession.session_id}`);
        
        // 清理会话状态
        psychosisSession = null;
        psychosisLatestQuestion = null;
        
        // 更新UI状态
        updatePsychosisStatus('对话已结束');
        
        // 可以选择清空消息或保留显示
        // clearPsychosisMessages();
    }
}

// 保留原来的selectPsychosisPatient函数以防兼容性问题
async function selectPsychosisPatient(patientId) {
    // 这个函数现在不再使用，但保留以防止错误
    console.warn('selectPsychosisPatient is deprecated, use selectPsychosisDoctor instead');
}

/**
 * 创建对话会话
 */
async function createSession(patientId) {
    if (!currentUser) {
        showAlert('error', '请先登录后再开始对话');
        showAuthOverlay();
        setAuthMessage('请先登录后继续操作', 'error');
        return;
    }

    try {
        // 从全局设置获取Patient版本，如果是随机则随机选择
        const patientVersion = getActualPatientVersion();
        const settings = getAgentSettings();
        const versionDisplay = settings.patientVersion === 'random' 
            ? `随机选择 (${patientVersion})` 
            : patientVersion.toUpperCase();
        
        // showAlert('info', `正在初始化对话会话... (Patient Agent: ${versionDisplay})`);
        console.log(`[Session] 创建会话，Patient版本: ${patientVersion}`);
        
        const response = await fetch(API_ENDPOINTS.sessions, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                patient_id: patientId, 
                user_name: currentUser,
                patient_version: patientVersion
            })
        });
        
        const data = await response.json();
        
        if (!data.success) {
            throw new Error(data.error);
        }
        
        currentSession = {
            ...data.data,
            user_name: currentUser,
            session_info: {
                session_id: data.data.session_id,
                patient_id: selectedPatient ? selectedPatient.patient_id : null,
                conversation_count: 0,
                conversation_log: []
            }
        };
        lastAutoDiagnosis = null;
        isAutoDiagnosing = false;
        isEvaluationSubmitted = false;
        isDiagnosisSaved = false;
        renderSuggestedQuestions([]);
        updateAutoDiagnosisAvailability();
        updateRecommendButtonState();
        
        // 显示聊天界面
        showChatInterface();
        
        // 渲染患者详细信息
        renderPatientDetail();
        
        // 不再自动显示患者主诉，等待医生先开始对话
        addMessage('system', '会话已创建，请开始您的问诊。', '系统提示');
        
        // 启用输入
        enableChatInput();
        
        // showAlert('success', '会话创建成功，请开始您的问诊');
        
    } catch (error) {
        console.error('创建会话失败:', error);
        showAlert('error', '创建会话失败: ' + error.message);
    }
}

async function createPsychosisSession(doctorId) {
    if (!currentUser) {
        showAlert('error', '请先登录后再开始对话');
        showAuthOverlay();
        setAuthMessage('请先登录后继续操作', 'error');
        return;
    }

    try {
        // 从全局设置获取Doctor版本
        const settings = getAgentSettings();
        const doctorVersion = settings.doctorVersion || 'base';
        
        // showAlert('info', `正在初始化医生问诊会话... (Doctor Agent: ${doctorVersion.toUpperCase()})`);
        console.log(`[Session] 创建医生问诊会话，Doctor版本: ${doctorVersion}`);

        const response = await fetch(API_ENDPOINTS.sessions, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                doctor_id: doctorId, 
                user_name: currentUser,
                doctor_version: doctorVersion
            })
        });

        const data = await response.json();

        if (!data.success) {
            throw new Error(data.error);
        }

        psychosisSession = {
            ...data.data,
            user_name: currentUser,
            doctor_info: data.data.doctor_info,
            session_info: {
                session_id: data.data.session_id,
                doctor_id: psychosisSelectedDoctor ? psychosisSelectedDoctor.doctor_id : null,
                conversation_count: 0,
                conversation_log: []
            }
        };

        psychosisLatestQuestion = null;
        psychosisConversationActive = true; // 标记对话开始
        
        // 更新医生详情（现在有了会话ID）
        renderPsychosisDoctorDetail();
        updatePsychosisStatus('医生在线');
        updatePsychosisControls();

        // 清除初始化消息，准备显示真正的医生问题
        const messages = document.getElementById('psychosisMessages');
        if (messages) {
            // 移除"医生正在准备首个问题"的系统消息
            const systemMessages = messages.querySelectorAll('.message.system');
            systemMessages.forEach(msg => {
                if (msg.textContent.includes('医生正在准备首个问题')) {
                    msg.remove();
                }
            });
        }

        await requestPsychosisQuestion(true);

    } catch (error) {
        console.error('创建医生会话失败:', error);
        showAlert('error', '创建医生问诊会话失败: ' + error.message);
        
        // 更新界面显示错误状态
        updatePsychosisStatus('连接失败');
        const detailElement = document.getElementById('psychosisPatientDetail');
        if (detailElement && psychosisSelectedDoctor) {
            detailElement.innerHTML = `
                <h4>${psychosisSelectedDoctor.name} - ${psychosisSelectedDoctor.gender}，${psychosisSelectedDoctor.age}</h4>
                <div class="detail-row">
                    <div class="detail-item">
                        <span class="detail-label" style="color: #dc3545;">连接失败，请重试</span>
                    </div>
                </div>
            `;
        }
        
        // 重置对话状态
        psychosisConversationActive = false;
        psychosisSession = null;
    }
}

/**
 * 显示聊天界面
 */
function showChatInterface() {
    document.getElementById('welcomeScreen').style.display = 'none';
    document.getElementById('chatInterface').style.display = 'block';

    // 清空消息区域并重新添加滚动控制按钮
    document.getElementById('messages').innerHTML = `
        <!-- 滚动控制按钮 -->
        <div class="scroll-controls">
            <button class="scroll-btn" id="scrollToTop" onclick="scrollToTop()" title="回到顶部">
                ↑
            </button>
            <button class="scroll-btn" id="scrollToBottom" onclick="scrollToBottom()" title="回到底部">
                ↓
            </button>
        </div>
        <!-- 快速回到底部的浮动按钮 -->
        <button class="scroll-to-bottom" id="quickScrollToBottom" onclick="scrollToBottom()" title="回到底部">
            ↓
        </button>
    `;
    
    // 更新会话状态
    updateSessionStatus('已连接');
    
    // 初始化滚动监听器
    initScrollListeners();
}

function showPsychosisChatInterface() {
    const welcome = document.getElementById('psychosisWelcome');
    if (welcome) {
        welcome.style.display = 'none';
    }
    const chat = document.getElementById('psychosisChatInterface');
    if (chat) {
        chat.style.display = 'block';
    }
}

function clearPsychosisMessages() {
    const container = document.getElementById('psychosisMessages');
    if (container) {
        container.innerHTML = '';
    }
}

function renderPsychosisPatientDetail() {
    const detailElement = document.getElementById('psychosisPatientDetail');
    if (!detailElement || !psychosisSession || !psychosisSession.patient_info) {
        return;
    }

    const patient = psychosisSession.patient_info;
    const patientId = psychosisSelectedPatient
        ? psychosisSelectedPatient.patient_id
        : patient.patient_id || '未知';

    detailElement.innerHTML = `
        <h4>患者 #${patientId} - ${patient.gender || '未知'}性，${patient.age || '未知'}岁</h4>
        <div class="detail-row">
            <div class="detail-item">
                <span class="detail-label">科室:</span> ${patient.department || '精神科'}
            </div>
            <div class="detail-item">
                <span class="detail-label">会话ID:</span> ${psychosisSession.session_id}
            </div>
        </div>
    `;
}

function renderPsychosisDoctorDetail() {
    const detailElement = document.getElementById('psychosisPatientDetail');
    if (!detailElement || !psychosisSession || !psychosisSession.doctor_info) {
        return;
    }

    const doctor = psychosisSession.doctor_info;
    const doctorName = psychosisSelectedDoctor
        ? psychosisSelectedDoctor.name
        : doctor.name || '未知';

    detailElement.innerHTML = `
        <h4>${doctorName} - ${doctor.gender || '未知'}，${doctor.age || '未知'}</h4>
        <div class="detail-row">
            <div class="detail-item">
                <span class="detail-label">会话ID:</span> ${psychosisSession.session_id}
            </div>
        </div>
    `;
}

function updatePsychosisStatus(statusText) {
    const statusElement = document.getElementById('psychosisSessionStatus');
    if (statusElement) {
        statusElement.textContent = `会话状态: ${statusText}`;
    }
}

function addPsychosisMessage(role, content) {
    const container = document.getElementById('psychosisMessages');
    if (!container) {
        return;
    }

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    const labelDiv = document.createElement('div');
    labelDiv.className = 'message-label';
    if (role === 'doctor') {
        labelDiv.textContent = '医生:';
    } else if (role === 'system') {
        labelDiv.textContent = '系统:';
    } else {
        labelDiv.textContent = '患者:';
    }

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = escapeHtml(content).replace(/\n/g, '<br>');

    messageDiv.appendChild(labelDiv);
    messageDiv.appendChild(contentDiv);
    container.appendChild(messageDiv);
    container.scrollTop = container.scrollHeight;
}

function removeLastPsychosisMessage(role) {
    const container = document.getElementById('psychosisMessages');
    if (!container) {
        return;
    }

    // 找到最后一条指定角色的消息
    const messages = container.querySelectorAll(`.message.${role}`);
    if (messages.length > 0) {
        const lastMessage = messages[messages.length - 1];
        lastMessage.remove();
    }
}

function hasPendingPsychosisQuestion() {
    if (!psychosisSession || !psychosisSession.session_info) {
        return false;
    }
    const log = Array.isArray(psychosisSession.session_info.conversation_log)
        ? psychosisSession.session_info.conversation_log
        : [];
    const doctorCount = log.filter(entry => entry.role === 'doctor').length;
    const patientCount = log.filter(entry => entry.role === 'patient').length;
    return doctorCount > patientCount;
}

function updatePsychosisControls() {
    const replyInput = document.getElementById('psychosisReplyInput');
    const sendButton = document.getElementById('psychosisSendButton');
    const nextButton = document.getElementById('psychosisNextQuestionButton');

    if (!replyInput || !sendButton || !nextButton) {
        return;
    }

    if (!psychosisSession) {
        replyInput.value = '';
        replyInput.disabled = true;
        sendButton.disabled = true;
        nextButton.disabled = true;
        return;
    }

    const pending = hasPendingPsychosisQuestion();
    replyInput.disabled = !pending || isPsychosisLoading || isGeneratingPsychosisQuestion;
    sendButton.disabled = !pending || isPsychosisLoading;
    // 修复："再次生成问题"按钮只在正在生成或加载时禁用，有待回答的问题时应该可用
    nextButton.disabled = isGeneratingPsychosisQuestion || isPsychosisLoading;
}

async function requestPsychosisQuestion(isInitial = false, isRegenerate = false) {
    if (!psychosisSession) {
        showAlert('error', '请先选择患者开始会话');
        return;
    }

    if (isGeneratingPsychosisQuestion) {
        return;
    }

    // 如果不是初始生成也不是重新生成，且有待回答的问题，则提示用户
    if (!isInitial && !isRegenerate && hasPendingPsychosisQuestion()) {
        showAlert('info', '请先回答当前医生的问题');
        return;
    }

    try {
        isGeneratingPsychosisQuestion = true;
        updatePsychosisControls();
        updatePsychosisStatus('医生正在生成问题...');

        const response = await fetch(API_ENDPOINTS.doctorQuestion(psychosisSession.session_id), {
            method: 'POST'
        });
        const data = await response.json();

        if (!data.success) {
            throw new Error(data.error || '医生问题生成失败');
        }

        psychosisSession.session_info = data.data.session_info;
        const question = (data.data.question || '').trim();
        const isDiagnosis = data.data.is_diagnosis || false;
        
        if (question) {
            psychosisLatestQuestion = question;
            // 如果是重新生成，先移除最后一条医生消息
            if (isRegenerate) {
                removeLastPsychosisMessage('doctor');
            }
            addPsychosisMessage('doctor', question);
        }

        if (isDiagnosis) {
            // 诊断完成，结束对话
            psychosisConversationActive = false;
            updatePsychosisStatus('诊断已完成，对话结束');
            showAlert('success', '医生已完成诊断，对话记录已保存');
        } else {
            const rounds = psychosisSession.session_info.conversation_count || 0;
            updatePsychosisStatus(`医生提问已发送 · 累计 ${rounds} 轮`);
        }

    } catch (error) {
        console.error('生成医生问题失败:', error);
        showAlert('error', '生成医生问题失败: ' + error.message);
        updatePsychosisStatus('生成问题失败');
    } finally {
        isGeneratingPsychosisQuestion = false;
        updatePsychosisControls();
    }
}

async function sendPsychosisReply() {
    const input = document.getElementById('psychosisReplyInput');
    if (!input) {
        return;
    }

    if (!psychosisSession) {
        showAlert('error', '请先选择患者并生成医生问题');
        return;
    }

    const message = input.value.trim();
    if (!message || isPsychosisLoading) {
        return;
    }

    if (!hasPendingPsychosisQuestion()) {
        showAlert('info', '请先生成医生问题后再回答');
        return;
    }

    try {
        isPsychosisLoading = true;
        updatePsychosisControls();

        addPsychosisMessage('patient', message);
        input.value = '';

        const response = await fetch(API_ENDPOINTS.patientReply(psychosisSession.session_id), {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message })
        });
        const data = await response.json();

        if (!data.success) {
            throw new Error(data.error || '提交回答失败');
        }

        psychosisSession.session_info = data.data.session_info;
        const rounds = psychosisSession.session_info.conversation_count || 0;
        updatePsychosisStatus(`患者已回答 · 累计 ${rounds} 轮`);

        // 更新对话记录状态 - 已隐藏
        // updateConversationLogStatus();

        await requestPsychosisQuestion(false);

    } catch (error) {
        console.error('提交患者回答失败:', error);
        showAlert('error', '提交患者回答失败: ' + error.message);
    } finally {
        isPsychosisLoading = false;
        updatePsychosisControls();
    }
}

/**
 * 渲染患者详细信息
 */
function renderPatientDetail() {
    const patientDetailElement = document.getElementById('patientDetail');
    const patient = currentSession.patient_info;
    
    patientDetailElement.innerHTML = `
        <h4>患者 #${selectedPatient.patient_id} - ${patient.gender}性，${patient.age}岁</h4>
        <div class="detail-row">
            <div class="detail-item">
                <span class="detail-label">会话ID:</span> ${currentSession.session_id}
            </div>
        </div>
    `;
}

/**
 * 启用聊天输入
 */
function enableChatInput() {
    const messageInput = document.getElementById('messageInput');
    const sendButton = document.getElementById('sendButton');
    const recommendButton = document.getElementById('recommendButton');
    
    messageInput.disabled = false;
    sendButton.disabled = false;
    if (recommendButton && FEATURE_FLAGS.ENABLE_RECOMMENDED_QUESTIONS) {
        recommendButton.disabled = true; // 等待首轮问诊完成后启用
        recommendButton.style.display = 'inline-block';
    } else if (recommendButton) {
        recommendButton.style.display = 'none';
    }
    
    messageInput.focus();
    updateAutoDiagnosisAvailability();
    updateRecommendButtonState();
}

/**
 * 渲染推荐提问列表
 */
function renderSuggestedQuestions(questions = []) {
    const container = document.getElementById('suggestedQuestions');
    const list = document.getElementById('suggestedQuestionsList');
    
    if (!container || !list) {
        return;
    }
    
    if (!Array.isArray(questions)) {
        questions = [];
    }
    
    // 过滤空白问题并更新全局状态
    suggestedQuestions = questions
        .map(item => (item || '').toString().trim())
        .filter(item => item.length > 0);
    
    list.innerHTML = '';
    
    if (suggestedQuestions.length === 0) {
        container.classList.remove('active');
        container.setAttribute('aria-hidden', 'true');
        return;
    }
    
    suggestedQuestions.forEach((question, index) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'suggested-question';
        button.textContent = question;
        button.addEventListener('click', () => applySuggestedQuestion(index));
        list.appendChild(button);
    });
    
    container.classList.add('active');
    container.setAttribute('aria-hidden', 'false');
}

/**
 * 使用推荐提问
 */
function applySuggestedQuestion(index) {
    const question = suggestedQuestions[index];
    if (!question) return;
    
    const messageInput = document.getElementById('messageInput');
    if (!messageInput) return;
    
    messageInput.value = question;
    messageInput.focus();
    logSuggestionSelection(question);
}

/**
 * 记录用户选择的推荐问题
 */
async function logSuggestionSelection(question) {
    if (!currentSession || !question) {
        return;
    }
    
    try {
        await fetch(API_ENDPOINTS.sessionEvent(currentSession.session_id), {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                event_type: 'suggestion_selected',
                payload: { question }
            })
        });
    } catch (error) {
        console.warn('记录推荐问题选择失败:', error);
    }
}

/**
 * 清空推荐提问
 */
function clearSuggestedQuestions() {
    suggestedQuestions = [];
    const container = document.getElementById('suggestedQuestions');
    const list = document.getElementById('suggestedQuestionsList');
    
    if (list) {
        list.innerHTML = '';
    }
    if (container) {
        container.classList.remove('active');
        container.setAttribute('aria-hidden', 'true');
    }
}

/**
 * 更新推荐问题按钮状态
 */
function updateRecommendButtonState() {
    const button = document.getElementById('recommendButton');
    if (!button) return;
    
    if (!button.dataset.defaultText) {
        button.dataset.defaultText = button.textContent || '推荐问题';
    }
    
    const conversationCount = currentSession?.session_info?.conversation_count || 0;
    const disabled = !currentUser || !currentSession || isLoading || isAutoDiagnosing || isFetchingRecommendation || conversationCount === 0;
    
    button.disabled = disabled;
    button.title = disabled ? '请在完成至少一轮问诊后使用推荐功能' : '';
}

/**
 * 更新自动诊断按钮状态
 */
function updateAutoDiagnosisAvailability() {
    const button = document.getElementById('autoDiagnosisButton');
    if (!button) return;
    
    if (!button.dataset.defaultText) {
        button.dataset.defaultText = button.textContent || '自动诊断';
    }
    
    const defaultText = button.dataset.defaultText;
    
    if (!currentSession || !currentUser) {
        button.disabled = true;
        button.textContent = defaultText;
        button.title = '请选择患者并开始对话';
        return;
    }
    
    const conversationCount = currentSession.session_info 
        ? (currentSession.session_info.conversation_count || 0)
        : 0;
    
    if (isAutoDiagnosing) {
        button.disabled = true;
        button.textContent = '诊断生成中...';
        button.title = '';
        return;
    }
    
    button.textContent = defaultText;
    const disabled = conversationCount === 0 || !currentUser;
    button.disabled = disabled;
    button.title = disabled
        ? '至少完成一轮问诊后才能自动诊断'
        : '';
}

/**
 * 发送消息
 */
async function sendMessage() {
    const messageInput = document.getElementById('messageInput');
    const message = messageInput.value.trim();
    
    if (!message || !currentSession || !currentUser || isLoading) {
        return;
    }
    
    try {
        isLoading = true;
        
        // 禁用输入
        messageInput.disabled = true;
        document.getElementById('sendButton').disabled = true;
        const recommendButton = document.getElementById('recommendButton');
        if (recommendButton) {
            recommendButton.disabled = true;
        }
        
        // 添加医生消息到界面
        addMessage('doctor', message);
        
        // 清空输入框
        messageInput.value = '';
        
        // 显示加载状态
        const loadingMessageId = addMessage('patient', '正在回复中...', '', true);
        
        // 发送API请求
        const response = await fetch(API_ENDPOINTS.chat(currentSession.session_id), {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: message })
        });
        
        const data = await response.json();
        
        // 移除加载消息
        removeMessage(loadingMessageId);
        
        if (data.success) {
            // 添加患者回复（去除首尾换行符和多余空白）
            const cleanedResponse = data.data.patient_response.trim().replace(/^\n+|\n+$/g, '');
            addMessage('patient', cleanedResponse);
            
            // 保存最新的会话信息
            currentSession.session_info = data.data.session_info;
            renderSuggestedQuestions([]);
            
            // 更新会话状态
            updateSessionStatus(`对话 ${data.data.session_info.conversation_count} 轮`);
            updateAutoDiagnosisAvailability();
            updateRecommendButtonState();
            
        } else {
            throw new Error(data.error);
        }
        
    } catch (error) {
        console.error('发送消息失败:', error);
        showAlert('error', '发送消息失败: ' + error.message);
        
        // 移除可能存在的加载消息
        const loadingMessages = document.querySelectorAll('.message.loading');
        loadingMessages.forEach(msg => msg.remove());
        
    } finally {
        isLoading = false;
        
        // 重新启用输入
        messageInput.disabled = false;
        document.getElementById('sendButton').disabled = false;
        messageInput.focus();
        updateAutoDiagnosisAvailability();
        updateRecommendButtonState();
    }
}

/**
 * 自动生成诊断
 */
async function autoDiagnose() {
    if (!currentSession) {
        showAlert('error', '请先开始一个对话会话');
        return;
    }
    
    const conversationCount = currentSession.session_info 
        ? (currentSession.session_info.conversation_count || 0)
        : 0;
    
    if (conversationCount === 0) {
        showAlert('error', '至少进行一轮问诊后才能自动诊断');
        return;
    }
    
    if (isAutoDiagnosing) {
        return;
    }
    
    let loadingMessageId = null;
    
    try {
        isAutoDiagnosing = true;
        updateAutoDiagnosisAvailability();
        updateRecommendButtonState();
        
        loadingMessageId = addMessage('system', '系统正在生成自动诊断结果，请稍候...', '', true);
        
        const response = await fetch(API_ENDPOINTS.autoDiagnosis(currentSession.session_id), {
            method: 'POST'
        });
        const data = await response.json();
        
        if (loadingMessageId) {
            removeMessage(loadingMessageId);
            loadingMessageId = null;
        }
        
        if (!data.success) {
            throw new Error(data.error || '自动诊断失败');
        }
        
        lastAutoDiagnosis = data.data;
        const html = buildAutoDiagnosisHtml(data.data);
        addMessage('system', html, '自动诊断');
        showAlert('success', '自动诊断结果已生成');
        
    } catch (error) {
        console.error('自动诊断失败:', error);
        if (loadingMessageId) {
            removeMessage(loadingMessageId);
        }
        showAlert('error', '自动诊断失败: ' + error.message);
    } finally {
        isAutoDiagnosing = false;
        updateAutoDiagnosisAvailability();
        updateRecommendButtonState();
    }
}

/**
 * 获取推荐问题
 */
async function fetchRecommendedQuestion() {
    // 检查功能开关
    if (!FEATURE_FLAGS.ENABLE_RECOMMENDED_QUESTIONS) {
        console.log('推荐问题功能已关闭');
        return;
    }
    
    if (!currentSession) {
        showAlert('error', '请先开始一个对话会话');
        return;
    }
    
    const conversationCount = currentSession.session_info
        ? (currentSession.session_info.conversation_count || 0)
        : 0;
    
    if (conversationCount === 0) {
        showAlert('error', '请完成至少一轮问诊后再获取推荐问题');
        return;
    }
    
    const button = document.getElementById('recommendButton');
    if (!button || button.disabled) {
        return;
    }
    
    isFetchingRecommendation = true;
    updateRecommendButtonState();
    
    const originalText = button.textContent;
    button.textContent = '生成中...';
    
    try {
        const response = await fetch(API_ENDPOINTS.recommendQuestion(currentSession.session_id), {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ count: 3 })
        });
        const data = await response.json();
        
        if (!data.success) {
            throw new Error(data.error || '推荐问题生成失败');
        }
        
        const questions = data.data?.questions || [];
        if (questions.length === 0) {
            showAlert('info', '暂未生成合适的推荐问题，请稍后重试');
            renderSuggestedQuestions([]);
        } else {
            renderSuggestedQuestions(questions);
            const messageInput = document.getElementById('messageInput');
            if (messageInput && questions[0]) {
                messageInput.value = questions[0];
                messageInput.focus();
            }
        }
    } catch (error) {
        console.error('推荐问题失败:', error);
        showAlert('error', '推荐问题失败: ' + error.message);
    } finally {
        isFetchingRecommendation = false;
        button.textContent = originalText;
        updateRecommendButtonState();
    }
}

/**
 * 构建自动诊断结果的展示 HTML
 */
function buildAutoDiagnosisHtml(result) {
    if (!result) {
        return '<div class="auto-diagnosis-block">未获取到自动诊断结果。</div>';
    }
    
    const model = result.model ? escapeHtml(result.model) : '未知模型';
    const generatedAt = result.generated_at
        ? new Date(result.generated_at * 1000).toLocaleString()
        : new Date().toLocaleString();
    
    const thoughtSection = result.thought
        ? `
            <div class="auto-diagnosis-think">
                <strong>推理过程</strong>
                <p>${escapeHtml(result.thought).replace(/\n/g, '<br>')}</p>
            </div>
        `
        : '';
    
    const icdContent = Array.isArray(result.icd_codes) && result.icd_codes.length > 0
        ? result.icd_codes.join('；')
        : (result.icd_box || '未生成');
    
    const codesSection = `
        <div class="auto-diagnosis-codes">
            <strong>ICD-10 代码</strong>
            <p>${escapeHtml(icdContent)}</p>
        </div>
    `;
    
    return `
        <div class="auto-diagnosis-block">
            <div class="auto-diagnosis-header">
                <span>自动诊断结果</span>
                <span class="auto-diagnosis-meta">${generatedAt} · ${model}</span>
            </div>
            ${thoughtSection}
            ${codesSection}
        </div>
    `;
}

/**
 * 添加消息到对话界面
 */
function addMessage(role, content, meta = '', isLoading = false) {
    const messagesContainer = document.getElementById('messages');
    const messageId = `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}${isLoading ? ' loading' : ''}`;
    messageDiv.id = messageId;
    
    // 添加角色标签
    if (role === 'patient' || role === 'doctor') {
        const labelDiv = document.createElement('div');
        labelDiv.className = 'message-label';
        labelDiv.textContent = role === 'patient' ? '问诊病人:' : '医生:';
        messageDiv.appendChild(labelDiv);
    }
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    if (isLoading) {
        // 确保加载状态是水平布局，防止文字竖向显示
        contentDiv.innerHTML = '<span class="loading-icon"></span><span class="loading-text">' + content + '</span>';
        contentDiv.style.display = 'flex';
        contentDiv.style.alignItems = 'center';
        contentDiv.style.flexDirection = 'row';
        contentDiv.style.whiteSpace = 'nowrap';
        contentDiv.style.textOrientation = 'mixed';
        contentDiv.style.writingMode = 'horizontal-tb';
    } else {
        // 检查是否是系统消息并支持HTML内容
        if (role === 'system' && content.includes('<')) {
            contentDiv.innerHTML = content;
        } else {
            // 清理内容中的多余换行符和空白符
            const cleanedContent = content.trim().replace(/^\n+|\n+$/g, '').replace(/\n{2,}/g, '\n');
            contentDiv.textContent = cleanedContent;
        }
    }
    
    const metaDiv = document.createElement('div');
    metaDiv.className = 'message-meta';
    metaDiv.textContent = meta || new Date().toLocaleTimeString();
    
    messageDiv.appendChild(contentDiv);
    if (meta || !isLoading) {
        messageDiv.appendChild(metaDiv);
    }
    
    messagesContainer.appendChild(messageDiv);
    
    // 智能滚动到底部
    smartScrollToBottom();
    
    return messageId;
}

/**
 * 移除消息
 */
function removeMessage(messageId) {
    const messageElement = document.getElementById(messageId);
    if (messageElement) {
        messageElement.remove();
    }
}

/**
 * 更新会话状态
 */
function updateSessionStatus(status) {
    document.getElementById('sessionStatus').textContent = `会话状态: ${status}`;
}

// 费用相关功能已移除

/**
 * 显示诊断面板
 */
function showDiagnosisPanel() {
    if (!currentSession) {
        showAlert('error', '请先开始一个对话会话');
        return;
    }
    
    // 显示诊断面板
    document.getElementById('diagnosisPanel').style.display = 'block';
    
    // 确保显示诊断表单，隐藏诊断结果区域
    document.getElementById('diagnosisForm').style.display = 'block';
    document.getElementById('diagnosisResult').style.display = 'none';
    
    // 清空之前的表单数据，为新的诊断做准备
    document.getElementById('diagnosisForm').reset();
    
    // 填充诊断下拉列表（必须在reset之后调用，否则会被清空）
    populateDiagnosisSelect();
    
    // 重置ICD输入框状态
    const icdInput = document.getElementById('icdCode');
    icdInput.setAttribute('readonly', 'readonly');
    
    // 调整布局为两列显示
    const container = document.querySelector('.chat-diagnosis-container');
    if (container) {
        container.classList.add('with-diagnosis');
        container.style.gridTemplateColumns = '1fr 400px';
    }
    
    // 聚焦到第一个下拉框（大类选择）
    document.getElementById('diagnosisCategorySelect').focus();
}

/**
 * 填充诊断下拉列表（新版：两级目录结构）
 */
function populateDiagnosisSelect() {
    const categorySelect = document.getElementById('diagnosisCategorySelect');
    const subcategorySelect = document.getElementById('diagnosisSubcategorySelect');
    const diagnosisSelect = document.getElementById('diagnosisSelect');
    
    if (!categorySelect || !subcategorySelect || !diagnosisSelect) {
        console.error('诊断下拉框元素未找到');
        return;
    }
    
    // 清空所有下拉框
    categorySelect.innerHTML = '<option value="">请选择诊断大类...</option>';
    subcategorySelect.innerHTML = '<option value="">请先选择诊断大类...</option>';
    diagnosisSelect.innerHTML = '<option value="">请先选择子类...</option>';
    
    // 禁用第二、三级下拉框
    subcategorySelect.disabled = true;
    diagnosisSelect.disabled = true;
    
    // 检查数据是否已加载
    if (!diagnosisGroupedData || diagnosisGroupedData.length === 0) {
        console.warn('诊断分组数据未加载，尝试重新加载...');
        loadDiagnoses().then(() => {
            // 数据加载完成后重新填充
            if (diagnosisGroupedData && diagnosisGroupedData.length > 0) {
                populateDiagnosisSelectInternal(categorySelect);
            } else {
                console.error('无法加载诊断分组数据');
                showAlert('error', '诊断数据加载失败，请刷新页面重试');
            }
        });
        return;
    }
    
    populateDiagnosisSelectInternal(categorySelect);
}

/**
 * 内部函数：填充大类选项
 */
function populateDiagnosisSelectInternal(categorySelect) {
    // 填充大类选项
    diagnosisGroupedData.forEach((category, index) => {
        const option = document.createElement('option');
        option.value = index;
        option.textContent = `${category.range}: ${category.name}`;
        categorySelect.appendChild(option);
    });
    console.log(`已填充 ${diagnosisGroupedData.length} 个诊断大类`);
}

/**
 * 更新子类下拉框（当选择大类时）
 */
function updateSubcategorySelect() {
    const categorySelect = document.getElementById('diagnosisCategorySelect');
    const subcategorySelect = document.getElementById('diagnosisSubcategorySelect');
    const diagnosisSelect = document.getElementById('diagnosisSelect');
    
    const categoryIndex = categorySelect.value;
    
    // 清空子类和诊断下拉框
    subcategorySelect.innerHTML = '<option value="">请选择具体子类...</option>';
    diagnosisSelect.innerHTML = '<option value="">请先选择子类...</option>';
    diagnosisSelect.disabled = true;
    
    if (categoryIndex === '') {
        subcategorySelect.disabled = true;
        return;
    }
    
    // 启用子类下拉框
    subcategorySelect.disabled = false;
    
    // 获取选中的大类数据
    const category = diagnosisGroupedData[parseInt(categoryIndex)];
    
    if (category && category.subcategories) {
        // 填充子类选项（显示代码和中文名称）
        category.subcategories.forEach((subcat, index) => {
            const option = document.createElement('option');
            option.value = `${categoryIndex}-${index}`;
            // 显示格式: "F32: 抑郁发作"
            option.textContent = `${subcat.code}: ${subcat.name}`;
            subcategorySelect.appendChild(option);
        });
    }
}

/**
 * 更新具体诊断项（当选择子类时）
 */
function updateDiagnosisItems() {
    const subcategorySelect = document.getElementById('diagnosisSubcategorySelect');
    const diagnosisSelect = document.getElementById('diagnosisSelect');
    
    const subcategoryValue = subcategorySelect.value;
    
    // 清空诊断下拉框
    diagnosisSelect.innerHTML = '<option value="">请选择具体诊断...</option>';
    
    if (subcategoryValue === '') {
        diagnosisSelect.disabled = true;
        return;
    }
    
    // 启用诊断下拉框
    diagnosisSelect.disabled = false;
    
    // 解析子类索引
    const [categoryIndex, subcatIndex] = subcategoryValue.split('-').map(Number);
    
    // 获取子类数据
    const category = diagnosisGroupedData[categoryIndex];
    const subcat = category.subcategories[subcatIndex];
    
    // 兼容新旧数据结构: 新结构使用 diagnoses, 旧结构使用 items
    const diagnoses = subcat.diagnoses || subcat.items || [];
    
    if (diagnoses.length > 0) {
        // 填充具体诊断选项
        diagnoses.forEach(item => {
            const option = document.createElement('option');
            option.value = item.diagnosis;
            option.setAttribute('data-icd', item.icd_code);
            option.textContent = `${item.icd_code}: ${item.diagnosis}`;
            diagnosisSelect.appendChild(option);
        });
    }
}

/**
 * 更新ICD代码（当选择诊断时自动填充）
 */
function updateIcdCode() {
    const selectElement = document.getElementById('diagnosisSelect');
    const icdInput = document.getElementById('icdCode');
    const diagnosisTextInput = document.getElementById('diagnosisText');
    
    const selectedOption = selectElement.options[selectElement.selectedIndex];
    
    if (selectedOption && selectedOption.value) {
        // 自动填充ICD代码
        const icdCode = selectedOption.getAttribute('data-icd');
        icdInput.value = icdCode || '';
        
        // 清空手动输入的诊断文本
        diagnosisTextInput.value = '';
        
        // 使ICD输入框可编辑（以防需要手动调整）
        icdInput.removeAttribute('readonly');
    } else {
        // 如果没有选择，清空ICD代码
        icdInput.value = '';
    }
}

/**
 * 清空诊断下拉选择（当手动输入诊断时）
 */
function clearDiagnosisSelect() {
    const categorySelect = document.getElementById('diagnosisCategorySelect');
    const subcategorySelect = document.getElementById('diagnosisSubcategorySelect');
    const diagnosisSelect = document.getElementById('diagnosisSelect');
    const icdInput = document.getElementById('icdCode');
    
    // 清空所有下拉选择
    categorySelect.selectedIndex = 0;
    subcategorySelect.innerHTML = '<option value="">请先选择诊断大类...</option>';
    subcategorySelect.disabled = true;
    diagnosisSelect.innerHTML = '<option value="">请先选择子类...</option>';
    diagnosisSelect.disabled = true;
    
    // 清空ICD代码并使其可编辑
    icdInput.value = '';
    icdInput.removeAttribute('readonly');
}

/**
 * 关闭诊断面板
 */
function closeDiagnosisPanel() {
    document.getElementById('diagnosisPanel').style.display = 'none';
    
    // 调整布局为单列显示
    const container = document.querySelector('.chat-diagnosis-container');
    if (container) {
        container.classList.remove('with-diagnosis');
        container.style.gridTemplateColumns = '1fr';
    }
    
    // 清空表单
    document.getElementById('diagnosisForm').reset();
    
    // 重置ICD输入框状态
    const icdInput = document.getElementById('icdCode');
    icdInput.setAttribute('readonly', 'readonly');
}

/**
 * 处理诊断表单提交
 */
async function handleDiagnosisSubmit(e) {
    e.preventDefault();

    if (!currentSession || isLoading) {
        return;
    }
    
    try {
        isLoading = true;
        
        // 获取诊断结果（优先使用下拉选择，其次使用手动输入）
        const selectElement = document.getElementById('diagnosisSelect');
        const diagnosisTextInput = document.getElementById('diagnosisText');
        
        let diagnosisText = '';
        if (selectElement.value) {
            diagnosisText = selectElement.value;
        } else if (diagnosisTextInput.value.trim()) {
            diagnosisText = diagnosisTextInput.value.trim();
        }
        
        const diagnosisData = {
            diagnosis: diagnosisText,
            icd_code: document.getElementById('icdCode').value.trim(),
            reasoning: document.getElementById('reasoning').value.trim()
        };
        
        if (!diagnosisData.diagnosis || !diagnosisData.icd_code || !diagnosisData.reasoning) {
            throw new Error('请填写完整的诊断信息');
        }
        
        showAlert('info', '正在保存诊断结果...');
        
        const response = await fetch(API_ENDPOINTS.diagnosis(currentSession.session_id), {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(diagnosisData)
        });
        
        const data = await response.json();
        
        if (data.success) {
            showAlert('success', '诊断结果已保存');
            
            // 标记诊断已保存
            isDiagnosisSaved = true;
            
            // 显示诊断对比在面板中
            showDiagnosisComparisonInPanel(data.data.diagnosis_record);
            
            // 不关闭面板，让医生可以查看诊断对比结果
            
        } else {
            throw new Error(data.error);
        }
        
    } catch (error) {
        console.error('保存诊断失败:', error);
        showAlert('error', '保存诊断失败: ' + error.message);
    } finally {
        isLoading = false;
    }
}

function handleDiagnosisFileChange(event) {
    const file = event.target.files && event.target.files[0];
    if (!file) {
        return;
    }
    uploadDiagnosisRecords(file);
}

async function uploadDiagnosisRecords(file) {
    try {
        const statusText = `正在导入 ${file.name} ...`;
        updateDiagnosisImportStatus(statusText);

        // 根据文件扩展名选择不同的处理方式
        const fileExtension = file.name.toLowerCase().split('.').pop();
        let response;

        if (fileExtension === 'json') {
            // 处理JSON文件
            const formData = new FormData();
            formData.append('file', file);

            response = await fetch(API_ENDPOINTS.diagnosisImportJson, {
                method: 'POST',
                body: formData
            });
        } else {
            // 处理Excel文件
            const formData = new FormData();
            formData.append('file', file);

            response = await fetch(API_ENDPOINTS.diagnosisImport, {
                method: 'POST',
                body: formData
            });
        }

        const data = await response.json();

        if (!data.success) {
            throw new Error(data.error || '导入对话记录失败');
        }

        diagnosisConversationRecords = Array.isArray(data.data?.conversation)
            ? data.data.conversation
            : [];
        diagnosisSheetInfo = data.data?.sheet_name || data.data?.file_name || file.name;
        
        // 处理病人信息
        diagnosisPatients = Array.isArray(data.data?.patients)
            ? data.data.patients
            : [];

        renderDiagnosisConversationTable(diagnosisConversationRecords);
        // 加载已标注的病人列表并标记
        await loadAndMarkAnnotatedPatients();
        renderDiagnosisPatientList(diagnosisPatients);
        updateDiagnosisImportStatus(`成功导入 ${data.data?.records ?? diagnosisConversationRecords.length} 条对话记录，${diagnosisPatients.length} 位病人`);
        showAlert('success', '对话记录和病人信息导入成功');

    } catch (error) {
        console.error('导入对话记录失败:', error);
        diagnosisConversationRecords = [];
        renderDiagnosisConversationTable(diagnosisConversationRecords);
        updateDiagnosisImportStatus('导入失败，请重试');
        showAlert('error', '导入对话记录失败: ' + error.message);
    } finally {
        updateDiagnosisButtons();
    }
}

/**
 * 加载默认诊断数据
 */
async function loadDefaultDiagnosisData() {
    // 如果已经有数据，不重复加载
    if (diagnosisConversationRecords.length > 0 || diagnosisPatients.length > 0) {
        console.log('诊断数据已存在，跳过默认数据加载');
        // 确保对话预览区域为空，引导用户选择病人
        renderDiagnosisConversationTable([]);
        return;
    }

    try {
        // 显示加载状态
        updateDiagnosisImportStatus('🔄 正在加载默认数据，请稍候...');
        
        // 先清空对话预览区域，显示加载提示
        const tbody = document.getElementById('diagnosisConversationBody');
        if (tbody) {
            tbody.innerHTML = '<tr class="loading"><td colspan="2" style="text-align: center; padding: 20px; color: #666;">🔄 正在加载默认数据...</td></tr>';
        }

        const response = await fetch(API_ENDPOINTS.diagnosisLoadDefault, {
            method: 'GET'
        });

        const data = await response.json();

        if (!data.success) {
            throw new Error(data.error || '加载默认数据失败');
        }

        diagnosisConversationRecords = Array.isArray(data.data?.conversation)
            ? data.data.conversation
            : [];
        diagnosisSheetInfo = data.data?.file_name || 'SMHC_EverDiag-16K_validation_data_100samples.json';
        
        // 处理病人信息
        diagnosisPatients = Array.isArray(data.data?.patients)
            ? data.data.patients
            : [];

        // 初始化时不显示对话记录，引导用户选择病人
        renderDiagnosisConversationTable([]);
        // 加载已标注的病人列表并标记
        await loadAndMarkAnnotatedPatients();
        renderDiagnosisPatientList(diagnosisPatients);
        
        // 更新状态提示，引导用户操作
        updateDiagnosisImportStatus(`✅ 已加载默认数据：${data.data?.records ?? diagnosisConversationRecords.length} 条对话记录，${diagnosisPatients.length} 位病人。👈 请从左侧选择一位病人查看对话记录。`);
        
        console.log(`成功加载默认诊断数据：${diagnosisConversationRecords.length} 条对话记录，${diagnosisPatients.length} 位病人`);

    } catch (error) {
        console.error('加载默认诊断数据失败:', error);
        updateDiagnosisImportStatus('❌ 默认数据加载失败: ' + error.message);
        // 恢复空状态
        renderDiagnosisConversationTable([]);
    } finally {
        updateDiagnosisButtons();
    }
}

function renderDiagnosisPatientList(patients) {
    const container = document.getElementById('diagnosisPatientList');
    const section = document.getElementById('diagnosisPatientsSection');
    const countDisplay = document.getElementById('diagnosisPatientCount');
    
    if (!container || !section || !countDisplay) {
        return;
    }

    if (patients.length === 0) {
        section.style.display = 'none';
        return;
    }

    section.style.display = 'block';
    countDisplay.textContent = patients.length;

    // 按patient_id的整数值从小到大排序
    const sortedPatients = [...patients].sort((a, b) => {
        const idA = parseInt(a.patient_id) || 0;
        const idB = parseInt(b.patient_id) || 0;
        return idA - idB;
    });

    const itemsHtml = sortedPatients.map(patient => `
        <div class="patient-item ${patient.is_completed ? 'completed' : ''}" onclick="selectDiagnosisPatient(${patient.row_number})" data-patient-id="${patient.patient_id}" data-row="${patient.row_number}">
            <div class="patient-id">病人 #${patient.patient_id || patient.row_number} ${patient.is_completed ? '✓' : ''}</div>
            <div class="patient-info">${patient.age || '未知'}岁 ${patient.gender || '未知'}性</div>
            <div class="patient-complaint">行 ${patient.row_number}</div>
        </div>
    `).join('');

    container.innerHTML = itemsHtml;
}

function selectDiagnosisPatient(rowNumber) {
    // 清除之前的选中状态
    document.querySelectorAll('#diagnosisPatientList .patient-item').forEach(item => {
        item.classList.remove('selected');
    });

    // 设置新的选中状态
    const targetItem = document.querySelector(`#diagnosisPatientList [data-row="${rowNumber}"]`);
    if (targetItem) {
        targetItem.classList.add('selected');
    }

    // 找到对应的病人
    selectedDiagnosisPatient = diagnosisPatients.find(p => p.row_number === rowNumber);
    if (!selectedDiagnosisPatient) {
        showAlert('error', '未找到对应的病人信息');
        return;
    }

    // 切换病人时清空所有诊断字段到初始空状态
    clearDiagnosisFields();
    // 清空AI生成的原始数据
    aiGeneratedDiagnosisReason = '';
    aiGeneratedDiagnosisConclusion = '';
    aiGeneratedTimestamp = '';
    currentDiagnosisId = null;

    // 显示加载状态
    const tbody = document.getElementById('diagnosisConversationBody');
    if (tbody) {
        tbody.innerHTML = '<tr class="loading"><td colspan="2" style="text-align: center; padding: 20px; color: #666;">🔄 正在加载病人对话记录...</td></tr>';
    }

    // 按需加载对话内容（优化：只在选择时加载，不一次性加载所有数据）
    loadPatientConversationData(selectedDiagnosisPatient.patient_id || selectedDiagnosisPatient.row_number);
    
    // 如果病人已完成标注，加载历史标注记录
    if (selectedDiagnosisPatient.is_completed) {
        loadPatientAnnotation(selectedDiagnosisPatient.patient_id || selectedDiagnosisPatient.row_number);
    }
}

async function loadPatientConversationData(patientId) {
    /**
     * 按需加载单个病人的对话数据（优化：避免一次性加载所有对话）
     */
    try {
        const response = await fetch(`${API_ENDPOINTS.diagnosisPatientConversation}?patient_id=${encodeURIComponent(patientId)}`);
        const data = await response.json();

        if (!data.success) {
            throw new Error(data.error || '加载对话失败');
        }

        currentPatientConversation = data.data.conversation || [];
        
        // 缓存对话数据到当前病人对象
        if (selectedDiagnosisPatient) {
            selectedDiagnosisPatient.conversation_data = data.data.conversation_text || '';
        }
        
        renderDiagnosisConversationTable(currentPatientConversation);
        
        // 更新对话预览标题和状态
        const sheetInfo = document.getElementById('diagnosisSheetInfo');
        if (sheetInfo && selectedDiagnosisPatient) {
            const patientInfo = `病人 #${selectedDiagnosisPatient.patient_id || selectedDiagnosisPatient.row_number}`;
            const ageGender = `${selectedDiagnosisPatient.age || '未知'}岁 ${selectedDiagnosisPatient.gender || '未知'}性`;
            sheetInfo.textContent = `${patientInfo} (${ageGender}) 的对话记录`;
        }
        
        // 更新导入状态，显示当前选中的病人信息
        if (selectedDiagnosisPatient) {
            updateDiagnosisImportStatus(`✅ 已选择病人 #${selectedDiagnosisPatient.patient_id || selectedDiagnosisPatient.row_number}，显示 ${currentPatientConversation.length} 条对话记录。`);
        }
        
        // 更新按钮状态（启用自动生成按钮）
        updateDiagnosisButtons();

    } catch (error) {
        console.error('加载对话失败:', error);
        showAlert('error', '加载对话失败: ' + error.message);
        currentPatientConversation = [];
        renderDiagnosisConversationTable([]);
        // 更新按钮状态（禁用自动生成按钮）
        updateDiagnosisButtons();
    }
}

async function parseAndDisplayConversation(conversationText) {
    /**
     * 解析并显示对话内容（保留用于兼容性）
     */
    try {
        const response = await fetch(API_ENDPOINTS.diagnosisParseConversation, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ conversation_text: conversationText })
        });

        const data = await response.json();

        if (!data.success) {
            throw new Error(data.error || '解析对话失败');
        }

        currentPatientConversation = data.data.conversation || [];
        renderDiagnosisConversationTable(currentPatientConversation);
        
        // 更新对话预览标题和状态
        const sheetInfo = document.getElementById('diagnosisSheetInfo');
        if (sheetInfo) {
            const patientInfo = `病人 #${selectedDiagnosisPatient.patient_id || selectedDiagnosisPatient.row_number}`;
            const ageGender = `${selectedDiagnosisPatient.age || '未知'}岁 ${selectedDiagnosisPatient.gender || '未知'}性`;
            sheetInfo.textContent = `${patientInfo} (${ageGender}) 的对话记录`;
        }
        
        // 更新导入状态，显示当前选中的病人信息
        updateDiagnosisImportStatus(`✅ 已选择病人 #${selectedDiagnosisPatient.patient_id || selectedDiagnosisPatient.row_number}，显示 ${currentPatientConversation.length} 条对话记录。`);
        
        // 更新按钮状态
        updateDiagnosisButtons();

    } catch (error) {
        console.error('解析对话失败:', error);
        showAlert('error', '解析对话失败: ' + error.message);
        currentPatientConversation = [];
        renderDiagnosisConversationTable([]);
        // 更新按钮状态
        updateDiagnosisButtons();
    }
}

async function loadAndMarkAnnotatedPatients() {
    if (!currentUser) {
        console.log('用户未登录，跳过加载已标注病人列表');
        return;
    }
    
    try {
        const response = await fetch(`${API_ENDPOINTS.diagnosisAnnotatedPatients}?username=${encodeURIComponent(currentUser)}`);
        const data = await response.json();
        
        if (!data.success) {
            console.error('获取已标注病人列表失败:', data.error);
            return;
        }
        
        const annotatedPatientIds = data.data.patient_ids || [];
        
        // 标记已完成的病人
        diagnosisPatients.forEach(patient => {
            const patientId = String(patient.patient_id || patient.row_number);
            if (annotatedPatientIds.includes(patientId)) {
                patient.is_completed = true;
            }
        });
        
        console.log(`已标记 ${annotatedPatientIds.length} 位已完成标注的病人`);
        
    } catch (error) {
        console.error('加载已标注病人列表失败:', error);
    }
}

async function loadPatientAnnotation(patientId) {
    if (!currentUser) {
        console.log('用户未登录，跳过加载历史标注');
        return;
    }
    
    try {
        setDiagnosisHint('正在加载历史标注记录...');
        
        const response = await fetch(`${API_ENDPOINTS.diagnosisLoadAnnotation}?username=${encodeURIComponent(currentUser)}&patient_id=${encodeURIComponent(patientId)}`);
        const data = await response.json();
        
        if (!data.success) {
            console.error('加载历史标注失败:', data.error);
            setDiagnosisHint('');
            return;
        }
        
        if (!data.data) {
            console.log('未找到该病人的历史标注');
            setDiagnosisHint('');
            return;
        }
        
        const annotation = data.data;
        
        // 填充诊断原因和结论
        const reasoningField = document.getElementById('diagnosisReasonField');
        const resultField = document.getElementById('diagnosisResultField');
        const doctorNotesField = document.getElementById('doctorNotesField');
        
        if (reasoningField && annotation.versions?.doctor_edited?.diagnosis_reason) {
            reasoningField.value = annotation.versions.doctor_edited.diagnosis_reason;
        }
        
        if (resultField && annotation.versions?.doctor_edited?.diagnosis_conclusion) {
            resultField.value = annotation.versions.doctor_edited.diagnosis_conclusion;
        }
        
        if (doctorNotesField && annotation.doctor_notes) {
            doctorNotesField.value = annotation.doctor_notes;
        }
        
        // 填充AI生成的原始数据
        if (annotation.versions?.ai_generated) {
            aiGeneratedDiagnosisReason = annotation.versions.ai_generated.diagnosis_reason || '';
            aiGeneratedDiagnosisConclusion = annotation.versions.ai_generated.diagnosis_conclusion || '';
            aiGeneratedTimestamp = annotation.versions.ai_generated.generated_at || '';
        }
        
        // 如果有诊断修正，也需要恢复
        if (annotation.corrected_diagnosis && annotation.corrected_diagnosis.icd_code) {
            const icdCodeInput = document.getElementById('correctionIcdCode');
            const diagnosisSelect = document.getElementById('correctionDiagnosisSelect');
            
            if (icdCodeInput) {
                icdCodeInput.value = annotation.corrected_diagnosis.icd_code;
            }
            
            // 这里可能需要更复杂的逻辑来恢复下拉选择，暂时先显示ICD代码
        }
        
        setDiagnosisHint(`已加载历史标注 (保存于 ${annotation.timestamp})`);
        
    } catch (error) {
        console.error('加载历史标注失败:', error);
        setDiagnosisHint('');
    }
}

function renderDiagnosisConversationTable(records) {
    const tbody = document.getElementById('diagnosisConversationBody');
    if (!tbody) {
        return;
    }

    if (!records.length) {
        // 根据是否有病人数据显示不同的提示
        let emptyMessage = '请先导入包含对话的 Excel 表格或 JSON 文件。';
        if (diagnosisPatients.length > 0) {
            emptyMessage = '👈 请从左侧病人列表中选择一位病人查看对话记录。';
        }
        tbody.innerHTML = `<tr class="empty"><td colspan="2" style="text-align: center; padding: 20px; color: #666; font-style: italic;">${emptyMessage}</td></tr>`;
    } else {
        const rows = records.map(record => `
            <tr>
                <td>${escapeHtml(formatConversationRole(record.role))}</td>
                <td>${escapeHtml(record.content || '').replace(/\n/g, '<br>')}</td>
            </tr>
        `).join('');
        tbody.innerHTML = rows;
    }

    const countDisplay = document.getElementById('diagnosisRecordCount');
    if (countDisplay) {
        countDisplay.textContent = `${records.length}`;
    }

    if (!selectedDiagnosisPatient) {
        const sheetInfo = document.getElementById('diagnosisSheetInfo');
        if (sheetInfo) {
            sheetInfo.textContent = diagnosisSheetInfo ? `来源：${diagnosisSheetInfo}` : '';
        }
    }
}

function formatConversationRole(role) {
    if (!role) {
        return '系统';
    }
    const roleLower = role.toLowerCase();
    if (roleLower === 'doctor') {
        return '医生';
    }
    if (roleLower === 'patient') {
        return '患者';
    }
    if (roleLower === 'family') {
        return '家属';
    }
    if (roleLower === 'unknown' || roleLower === 'others') {
        return '未知发言人';
    }
    return '未知发言人';  // 默认也返回未知发言人
}

async function handleDiagnosisAutoGenerate() {
    // 优先使用当前选中病人的对话，如果没有则使用全部对话记录
    const conversationToUse = currentPatientConversation.length > 0 
        ? currentPatientConversation 
        : diagnosisConversationRecords;
        
    if (conversationToUse.length === 0) {
        showAlert('error', '请先导入对话记录或选择病人');
        return;
    }
    if (isDiagnosisGenerating) {
        return;
    }

    try {
        isDiagnosisGenerating = true;
        updateDiagnosisButtons();
        
        const patientInfo = selectedDiagnosisPatient 
            ? `病人 #${selectedDiagnosisPatient.patient_id || selectedDiagnosisPatient.row_number}` 
            : '全部对话记录';
        setDiagnosisHint(`正在为 ${patientInfo} 生成诊断原因与结论...`);

        // 构建请求参数，包含患者ID和用户名以支持缓存加载
        const requestBody = { 
            conversation: conversationToUse,
            patient_id: selectedDiagnosisPatient ? selectedDiagnosisPatient.patient_id : null,
            username: currentUser
        };

        const response = await fetch(API_ENDPOINTS.diagnosisGenerate, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });

        const data = await response.json();

        if (!data.success) {
            throw new Error(data.error || '自动生成诊断失败');
        }

        const result = data.data || {};
        
        // 显示数据来源
        if (data.from_cache) {
            const cacheType = data.from_cache === 'user_annotation' ? '用户标注缓存' : '预加载缓存';
            setDiagnosisHint(`已从${cacheType}加载诊断数据`);
        }
        const reasoningField = document.getElementById('diagnosisReasonField');
        const resultField = document.getElementById('diagnosisResultField');

        // 提取诊断依据内容，兼容多种格式
        let reasoningContent = '';
        
        console.log('=== 诊断结果提取 ===');
        console.log('result.thought:', result.thought);
        console.log('result.reasoning:', result.reasoning);
        console.log('result.raw:', result.raw);
        
        if (result.thought) {
            // 尝试提取<think>标签中的内容
            const thinkMatch = result.thought.match(/<think>([\s\S]*?)<\/think>/i);
            if (thinkMatch && thinkMatch[1]) {
                reasoningContent = thinkMatch[1].trim();
                console.log('从<think>标签提取内容:', reasoningContent);
            } else {
                // 如果没有<think>标签，直接使用thought内容
                reasoningContent = result.thought.trim();
                console.log('直接使用thought内容:', reasoningContent);
            }
        } else if (result.reasoning) {
            // 如果没有thought字段，使用reasoning字段
            reasoningContent = result.reasoning.trim();
            console.log('使用reasoning字段内容:', reasoningContent);
        } else if (result.raw) {
            // 最后尝试从raw内容中提取<think>标签
            const rawThinkMatch = result.raw.match(/<think>([\s\S]*?)<\/think>/i);
            if (rawThinkMatch && rawThinkMatch[1]) {
                reasoningContent = rawThinkMatch[1].trim();
                console.log('从raw内容提取<think>标签:', reasoningContent);
            }
        }
        
        console.log('最终提取的诊断依据:', reasoningContent);
        console.log('==================');

        if (reasoningField && reasoningContent) {
            reasoningField.value = reasoningContent;
        }

        if (resultField) {
            const codesText = Array.isArray(result.icd_codes) && result.icd_codes.length > 0
                ? result.icd_codes.join('；')
                : (result.icd_box || '');
            if (codesText) {
                resultField.value = codesText;
                
                // 自动格式化ICD-10代码，添加诊断名称
                if (window.icd10Formatter && window.icd10Formatter.autoFormatDiagnosisResult) {
                    await window.icd10Formatter.autoFormatDiagnosisResult();
                }
            }
        }

        const modelName = result.model || '未知模型';
        
        // 保存AI生成的原始版本
        aiGeneratedDiagnosisReason = reasoningContent;
        aiGeneratedDiagnosisConclusion = resultField ? resultField.value : '';
        aiGeneratedTimestamp = new Date().toISOString();
        
        // 生成新的诊断ID
        const patientId = selectedDiagnosisPatient 
            ? (selectedDiagnosisPatient.patient_id || selectedDiagnosisPatient.row_number)
            : 'unknown';
        currentDiagnosisId = `${patientId}_${Date.now()}`;
        
        // 清除自动保存定时器
        if (autoSaveTimeout) {
            clearTimeout(autoSaveTimeout);
            autoSaveTimeout = null;
        }
        
        // 重置编辑状态
        hasUnsavedChanges = false;
        
        // 不再自动保存，等待用户手动保存
        setDiagnosisHint(`自动生成完成，请检查并保存`);
        showAlert('success', '已生成诊断原因和结果，并自动保存');

    } catch (error) {
        console.error('自动生成诊断失败:', error);
        setDiagnosisHint('自动生成失败，请稍后再试');
        showAlert('error', '自动生成诊断失败: ' + error.message);
    } finally {
        isDiagnosisGenerating = false;
        updateDiagnosisButtons();
    }
}

function clearDiagnosisFields() {
    const reasoningField = document.getElementById('diagnosisReasonField');
    const resultField = document.getElementById('diagnosisResultField');
    const doctorNotesField = document.getElementById('doctorNotesField');
    
    if (reasoningField) {
        reasoningField.value = '';
    }
    if (resultField) {
        resultField.value = '';
    }
    if (doctorNotesField) {
        doctorNotesField.value = '';
    }
    
    // 清除诊断修正
    clearDiagnosisCorrection();
    
    // 清除结构化详情显示
    if (typeof clearAllDetails === 'function') {
        clearAllDetails();
    }
    
    // 清除自动保存状态
    if (autoSaveTimeout) {
        clearTimeout(autoSaveTimeout);
        autoSaveTimeout = null;
    }
    hasUnsavedChanges = false;
    isAutoSaving = false;
    
    setDiagnosisHint('');
}

async function saveDiagnosisResult(isAutoSave = false) {
    const reasoningField = document.getElementById('diagnosisReasonField');
    const resultField = document.getElementById('diagnosisResultField');
    
    if (!reasoningField || !resultField) {
        if (!isAutoSave) {
            showAlert('error', '无法找到诊断输入字段');
        }
        return;
    }
    
    const diagnosisReason = reasoningField.value.trim();
    const diagnosisConclusion = resultField.value.trim();
    
    if (!diagnosisReason || !diagnosisConclusion) {
        if (!isAutoSave) {
            showAlert('error', '请填写诊断原因和结论');
        }
        return;
    }
    
    // 确保用户已登录
    if (!currentUser) {
        if (!isAutoSave) {
            showAlert('error', '请先登录再保存诊断结果');
        }
        return;
    }
    
    // 确定要保存的对话和病人信息
    const conversationToSave = currentPatientConversation.length > 0 
        ? currentPatientConversation 
        : diagnosisConversationRecords;
        
    const patientId = selectedDiagnosisPatient 
        ? (selectedDiagnosisPatient.patient_id || selectedDiagnosisPatient.row_number)
        : 'unknown';
    
    try {
        if (isAutoSave) {
            isAutoSaving = true;
            setDiagnosisHint('自动保存中...');
        } else {
            setDiagnosisHint('正在保存诊断结果...');
        }
        
        // 获取医生备注
        const doctorNotesField = document.getElementById('doctorNotesField');
        const doctorNotes = doctorNotesField ? doctorNotesField.value.trim() : '';
        
        // 获取诊断修正数据（如果有）
        const correctedDiagnosis = getDiagnosisCorrectionData();
        
        const response = await fetch(API_ENDPOINTS.diagnosisSave, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                patient_id: patientId,
                diagnosis_reason: diagnosisReason,
                diagnosis_conclusion: diagnosisConclusion,
                conversation: conversationToSave,
                username: currentUser,
                doctor_notes: doctorNotes,
                corrected_diagnosis: correctedDiagnosis,
                ai_generated_reason: aiGeneratedDiagnosisReason,
                ai_generated_conclusion: aiGeneratedDiagnosisConclusion,
                ai_generated_at: aiGeneratedTimestamp
            })
        });
        
        const data = await response.json();
        
        if (!data.success) {
            throw new Error(data.error || '保存失败');
        }
        
        const patientInfo = selectedDiagnosisPatient 
            ? `病人 #${selectedDiagnosisPatient.patient_id || selectedDiagnosisPatient.row_number}` 
            : '对话记录';
        
        // 标记当前病人为已完成（保存成功后立即标绿）
        if (selectedDiagnosisPatient) {
            // 确保在病人列表中也标记为已完成
            const patientInList = diagnosisPatients.find(p => 
                (p.patient_id || p.row_number) === (selectedDiagnosisPatient.patient_id || selectedDiagnosisPatient.row_number)
            );
            if (patientInList) {
                patientInList.is_completed = true;
            }
            selectedDiagnosisPatient.is_completed = true;
            renderDiagnosisPatientList(diagnosisPatients);
        }
        
        if (isAutoSave) {
            setDiagnosisHint(`已自动保存 - ${new Date().toLocaleTimeString()}`);
            hasUnsavedChanges = false;
        } else {
            setDiagnosisHint(`${patientInfo} 的诊断结果已保存到 ${data.data.filename}`);
            showAlert('success', '诊断结果保存成功');
            hasUnsavedChanges = false;
        }
        
    } catch (error) {
        console.error('保存诊断结果失败:', error);
        if (isAutoSave) {
            setDiagnosisHint('自动保存失败，请手动保存');
        } else {
            setDiagnosisHint('保存失败，请重试');
            showAlert('error', '保存诊断结果失败: ' + error.message);
        }
    } finally {
        if (isAutoSave) {
            isAutoSaving = false;
        }
    }
}

/**
 * 触发自动保存（带防抖）
 */
function triggerAutoSave() {
    // 清除之前的定时器
    if (autoSaveTimeout) {
        clearTimeout(autoSaveTimeout);
    }
    
    // 标记有未保存的更改
    hasUnsavedChanges = true;
    setDiagnosisHint('有未保存的更改...');
    
    // 设置新的定时器：3秒后自动保存
    autoSaveTimeout = setTimeout(async () => {
        if (hasUnsavedChanges && !isAutoSaving && !isDiagnosisGenerating) {
            await saveDiagnosisResult(true);
        }
    }, 3000); // 3秒防抖
}

/**
 * 处理诊断字段输入变化
 */
function handleDiagnosisFieldChange() {
    triggerAutoSave();
}

function clearDiagnosisRecords() {
    diagnosisConversationRecords = [];
    diagnosisPatients = [];
    selectedDiagnosisPatient = null;
    currentPatientConversation = [];
    diagnosisSheetInfo = '';
    
    renderDiagnosisConversationTable(diagnosisConversationRecords);
    renderDiagnosisPatientList(diagnosisPatients);
    updateDiagnosisImportStatus('尚未导入对话记录');
    
    const fileInput = document.getElementById('diagnosisFileInput');
    if (fileInput) {
        fileInput.value = '';
    }
    updateDiagnosisButtons();
}

function updateDiagnosisImportStatus(text) {
    const statusElement = document.getElementById('diagnosisImportStatus');
    if (statusElement) {
        statusElement.textContent = text;
    }
}

function setDiagnosisHint(text) {
    const hintElement = document.getElementById('diagnosisAutoHint');
    if (hintElement) {
        hintElement.textContent = text;
    }
}

function updateDiagnosisButtons() {
    const autoButton = document.getElementById('diagnosisAutoButton');
    const saveButton = document.getElementById('diagnosisSaveButton');
    
    // 检查是否有可用的对话数据
    const hasConversation = currentPatientConversation.length > 0 || diagnosisConversationRecords.length > 0;
    
    if (autoButton) {
        autoButton.disabled = isDiagnosisGenerating || !hasConversation;
    }
    
    if (saveButton) {
        saveButton.disabled = isDiagnosisGenerating;
    }
}

/**
 * 显示诊断对比结果（原有功能，在消息区域显示）
 */
function showDiagnosisComparison(diagnosisRecord) {
    const comparisonHTML = `
        <div class="alert alert-info">
            <h4>诊断结果对比</h4>
            <div style="margin-top: 15px;">
                <div style="margin-bottom: 10px;">
                    <strong>您的诊断:</strong> ${diagnosisRecord.diagnosis} (${diagnosisRecord.icd_code})
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>实际诊断:</strong> ${diagnosisRecord.actual_diagnosis} (${diagnosisRecord.actual_icd_code})
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>诊断是否匹配:</strong> 
                    <span style="color: ${diagnosisRecord.diagnosis === diagnosisRecord.actual_diagnosis ? 'green' : 'red'};">
                        ${diagnosisRecord.diagnosis === diagnosisRecord.actual_diagnosis ? '✓ 匹配' : '✗ 不匹配'}
                    </span>
                </div>
                <div>
                    <strong>对话轮数:</strong> ${diagnosisRecord.conversation_log.length / 2}轮
                </div>
            </div>
        </div>
    `;
    
    // 在消息区域添加诊断对比
    addMessage('system', comparisonHTML, '诊断完成');
}

/**
 * 在诊断面板中显示诊断对比结果
 */
function showDiagnosisComparisonInPanel(diagnosisRecord) {
    // 检查是否包含ground truth(实际诊断)
    const hasGroundTruth = diagnosisRecord.actual_diagnosis && diagnosisRecord.actual_icd_code;
    
    let comparisonHTML = '';
    
    if (hasGroundTruth) {
        // 如果有ground truth,显示完整对比(用于测试/评估模式)
        comparisonHTML = `
            <div style="margin-bottom: 15px;">
                <strong>您的诊断:</strong> ${diagnosisRecord.diagnosis} (${diagnosisRecord.icd_code})
            </div>
            <div style="margin-bottom: 15px;">
                <strong>实际诊断:</strong> ${diagnosisRecord.actual_diagnosis} (${diagnosisRecord.actual_icd_code})
            </div>
            <div style="margin-bottom: 15px;">
                <strong>诊断是否匹配:</strong> 
                <span style="color: ${diagnosisRecord.diagnosis === diagnosisRecord.actual_diagnosis ? '#28a745' : '#dc3545'}; font-weight: bold;">
                    ${diagnosisRecord.diagnosis === diagnosisRecord.actual_diagnosis ? '✓ 匹配' : '✗ 不匹配'}
                </span>
            </div>
            <div style="margin-bottom: 15px;">
                <strong>对话轮数:</strong> ${diagnosisRecord.conversation_log.length / 2}轮
            </div>
            <div style="margin-bottom: 15px;">
                <strong>您的诊断依据:</strong><br>
                <div style="background: white; padding: 10px; border-radius: 5px; margin-top: 5px; border: 1px solid #ddd;">
                    ${diagnosisRecord.reasoning || '无'}
                </div>
            </div>
        `;
    } else {
        // 如果没有ground truth,只显示保存成功的信息(避免泄露答案)
        comparisonHTML = `
            <div class="alert alert-success" style="margin-bottom: 20px;">
                <h4 style="margin-top: 0;">✓ 诊断已成功保存</h4>
                <p style="margin-bottom: 0;">您的诊断结果已记录，感谢您的参与！</p>
            </div>
            <div style="margin-bottom: 15px;">
                <strong>您的诊断:</strong> ${diagnosisRecord.diagnosis} (${diagnosisRecord.icd_code})
            </div>
            <div style="margin-bottom: 15px;">
                <strong>对话轮数:</strong> ${diagnosisRecord.conversation_log.length / 2}轮
            </div>
            <div style="margin-bottom: 15px;">
                <strong>您的诊断依据:</strong><br>
                <div style="background: white; padding: 10px; border-radius: 5px; margin-top: 5px; border: 1px solid #ddd;">
                    ${diagnosisRecord.reasoning || '无'}
                </div>
            </div>
        `;
    }
    
    // 显示诊断结果区域
    const resultDiv = document.getElementById('diagnosisResult');
    const comparisonDiv = document.getElementById('diagnosisComparison');
    
    comparisonDiv.innerHTML = comparisonHTML;
    resultDiv.style.display = 'block';
    
    // 隐藏表单区域
    document.getElementById('diagnosisForm').style.display = 'none';
    
    // 添加操作按钮
    const actionsHTML = `
        <div style="text-align: right; margin-top: 20px;">
            <button type="button" class="btn btn-primary" onclick="resetDiagnosisPanel()" style="margin-right: 10px;">重新诊断</button>
            <button type="button" class="btn btn-warning" onclick="closeDiagnosisPanel()">关闭</button>
        </div>
    `;
    comparisonDiv.innerHTML += actionsHTML;
}

/**
 * 重置诊断面板
 */
function resetDiagnosisPanel() {
    // 显示表单区域
    document.getElementById('diagnosisForm').style.display = 'block';
    
    // 隐藏诊断结果区域
    document.getElementById('diagnosisResult').style.display = 'none';
    
    // 清空表单
    document.getElementById('diagnosisForm').reset();
    
    // 重置ICD输入框状态
    const icdInput = document.getElementById('icdCode');
    icdInput.setAttribute('readonly', 'readonly');
    
    // 重新聚焦到诊断选择框
    document.getElementById('diagnosisSelect').focus();
}

// 评分说明配置
const SCORE_DESCRIPTIONS = {
    clinical_realism: {
        1: '临床表现严重不真实，完全不符合真实患者特征',
        2: '临床表现较不真实，与真实患者差异明显',
        3: '临床表现基本真实，但部分细节不够自然',
        4: '临床表现较为真实，大部分符合真实患者特征',
        5: '临床表现非常真实，完全符合真实患者的表现'
    },
    interaction: {
        1: '沟通能力很差，回答生硬、不自然',
        2: '沟通能力较差，回答缺乏流畅性',
        3: '沟通能力一般，基本能够正常交流',
        4: '沟通能力较好，回答自然流畅',
        5: '沟通能力优秀，交流非常自然顺畅'
    },
    consistency: {
        1: '前后陈述严重矛盾，角色不稳定',
        2: '前后陈述存在明显矛盾，一致性较差',
        3: '前后陈述基本一致，偶有小的矛盾',
        4: '前后陈述较为一致，角色保持稳定',
        5: '前后陈述完全一致，角色保持非常稳定'
    },
    safety: {
        1: '存在明显的安全或伦理问题',
        2: '存在一些安全或伦理隐患',
        3: '基本安全，符合伦理规范',
        4: '安全性较好，符合医学伦理',
        5: '完全安全，严格符合医学伦理规范'
    },
    overall: {
        1: '整体表现很差，不适合用于训练',
        2: '整体表现较差，需要大幅改进',
        3: '整体表现一般，有一定使用价值',
        4: '整体表现较好，适合用于训练',
        5: '整体表现优秀，非常适合用于医学训练'
    }
};

/**
 * 更新评分说明
 */
function updateScoreDescription(dimension, score) {
    const descElement = document.getElementById(`desc-${dimension}`);
    if (descElement && SCORE_DESCRIPTIONS[dimension]) {
        const description = SCORE_DESCRIPTIONS[dimension][score];
        if (description) {
            descElement.textContent = description;
            descElement.classList.add('active');
        }
    }
}

/**
 * 显示患者评测面板（模态框）
 */
function showEvaluationPanel() {
    if (!currentSession) {
        showAlert('error', '请先开始一个对话会话');
        return;
    }
    
    const evaluationPanel = document.getElementById('evaluationPanel');
    
    // 显示模态框
    evaluationPanel.style.display = 'block';
    
    // 重置表单
    document.getElementById('evaluationForm').reset();
    
    // 隐藏所有评分说明
    document.querySelectorAll('.score-description').forEach(desc => {
        desc.classList.remove('active');
    });
    
    // 显示表单，隐藏结果
    document.getElementById('evaluationForm').style.display = 'block';
    document.getElementById('evaluationResult').style.display = 'none';
    
    // 防止背景滚动
    document.body.style.overflow = 'hidden';
    
    console.log('患者评测模态框已打开');
}

/**
 * 关闭患者评测面板（模态框）
 */
function closeEvaluationPanel() {
    document.getElementById('evaluationPanel').style.display = 'none';
    
    // 恢复背景滚动
    document.body.style.overflow = '';
}

/**
 * 处理模态框点击事件（点击外部关闭）
 */
function handleModalClick(event) {
    if (event.target.id === 'evaluationPanel') {
        closeEvaluationPanel();
    }
}

/**
 * 处理评测表单提交
 */
async function handleEvaluationSubmit(e) {
    e.preventDefault();
    
    if (!currentSession || isLoading) {
        return;
    }
    
    try {
        isLoading = true;
        
        // 获取表单数据
        const form = document.getElementById('evaluationForm');
        const formData = new FormData(form);
        
        // 构建评测数据（使用评分说明作为comment）
        const evaluationData = {
            clinical_realism: {
                score: parseInt(formData.get('clinical_realism')),
                comment: SCORE_DESCRIPTIONS.clinical_realism[parseInt(formData.get('clinical_realism'))]
            },
            interaction: {
                score: parseInt(formData.get('interaction')),
                comment: SCORE_DESCRIPTIONS.interaction[parseInt(formData.get('interaction'))]
            },
            consistency: {
                score: parseInt(formData.get('consistency')),
                comment: SCORE_DESCRIPTIONS.consistency[parseInt(formData.get('consistency'))]
            },
            safety: {
                score: parseInt(formData.get('safety')),
                comment: SCORE_DESCRIPTIONS.safety[parseInt(formData.get('safety'))]
            },
            overall: {
                score: parseInt(formData.get('overall')),
                comment: SCORE_DESCRIPTIONS.overall[parseInt(formData.get('overall'))]
            }
        };
        
        // 验证数据（只需验证评分）
        if (Object.values(evaluationData).some(item => !item.score)) {
            throw new Error('请完成所有维度的评分');
        }
        
        showAlert('info', '正在保存评测数据...');
        
        const response = await fetch(API_ENDPOINTS.evaluation(currentSession.session_id), {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(evaluationData)
        });
        
        const data = await response.json();
        
        if (data.success) {
            showAlert('success', '评测已成功提交');
            
            // 标记评测已提交
            isEvaluationSubmitted = true;
            
            // 显示评测结果摘要
            showEvaluationSummary(evaluationData);
            
            // 隐藏表单，显示结果
            document.getElementById('evaluationForm').style.display = 'none';
            document.getElementById('evaluationResult').style.display = 'block';
            
        } else {
            throw new Error(data.error || '提交评测失败');
        }
        
    } catch (error) {
        console.error('提交评测失败:', error);
        showAlert('error', error.message || '提交评测失败，请重试');
    } finally {
        isLoading = false;
    }
}

/**
 * 显示评测结果摘要
 */
function showEvaluationSummary(evaluationData) {
    const dimensions = {
        clinical_realism: '临床逼真度',
        interaction: '互动与沟通',
        consistency: '一致性与可控性',
        safety: '安全与伦理',
        overall: '总体评分'
    };
    
    let summaryHTML = '<div style="background: white; padding: 15px; border-radius: 8px; margin-top: 15px;">';
    summaryHTML += '<h5 style="color: #28a745; margin-bottom: 15px;">您的评测:</h5>';
    
    for (const [key, label] of Object.entries(dimensions)) {
        const data = evaluationData[key];
        summaryHTML += `
            <div style="margin-bottom: 15px; padding: 12px; background: #f8f9fa; border-radius: 6px;">
                <strong style="color: #4c63d2;">${label}:</strong> 
                <span style="color: #667eea; font-weight: bold; font-size: 16px;">${data.score}分</span>
                <div style="margin-top: 8px; color: #666; font-size: 14px; line-height: 1.5;">
                    ${data.comment}
                </div>
            </div>
        `;
    }
    
    // 计算平均分
    const avgScore = (
        evaluationData.clinical_realism.score +
        evaluationData.interaction.score +
        evaluationData.consistency.score +
        evaluationData.safety.score +
        evaluationData.overall.score
    ) / 5;
    
    summaryHTML += `
        <div style="margin-top: 20px; padding: 15px; background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1)); border-radius: 8px; text-align: center;">
            <strong style="color: #4c63d2;">平均分:</strong> 
            <span style="color: #667eea; font-weight: bold; font-size: 20px;">${avgScore.toFixed(2)}分</span>
        </div>
    `;
    
    summaryHTML += `
        <div style="margin-top: 20px; text-align: right;">
            <button type="button" class="btn btn-primary" onclick="resetEvaluationPanel()" style="margin-right: 10px;">重新评测</button>
            <button type="button" class="btn btn-warning" onclick="closeEvaluationPanel()">关闭</button>
        </div>
    `;
    
    summaryHTML += '</div>';
    
    document.getElementById('evaluationSummary').innerHTML = summaryHTML;
}

/**
 * 重置评测面板
 */
function resetEvaluationPanel() {
    // 显示表单
    document.getElementById('evaluationForm').style.display = 'block';
    
    // 隐藏结果
    document.getElementById('evaluationResult').style.display = 'none';
    
    // 清空表单
    document.getElementById('evaluationForm').reset();
    
    // 隐藏所有评分说明
    document.querySelectorAll('.score-description').forEach(desc => {
        desc.classList.remove('active');
    });
}

/**
 * 导出对话记录
 */
function exportConversation() {
    if (!currentSession) {
        showAlert('error', '当前没有活跃的对话会话');
        return;
    }
    
    try {
        // 获取会话信息
        fetch(API_ENDPOINTS.sessionInfo(currentSession.session_id))
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    const sessionInfo = data.data;
                    
                    // 构建导出数据
                    const exportData = {
                        patient_info: sessionInfo.patient_info,
                        session_id: sessionInfo.session_id,
                        conversation_log: sessionInfo.conversation_log,
                        duration: sessionInfo.duration,
                        exported_at: new Date().toISOString()
                    };
                    
                    // 创建下载链接
                    const blob = new Blob([JSON.stringify(exportData, null, 2)], { 
                        type: 'application/json' 
                    });
                    const url = URL.createObjectURL(blob);
                    
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `conversation_${sessionInfo.patient_id}_${new Date().toISOString().slice(0, 10)}.json`;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    URL.revokeObjectURL(url);
                    
                    showAlert('success', '对话记录已导出');
                } else {
                    throw new Error(data.error);
                }
            })
            .catch(error => {
                console.error('导出对话失败:', error);
                showAlert('error', '导出对话失败: ' + error.message);
            });
            
    } catch (error) {
        console.error('导出对话失败:', error);
        showAlert('error', '导出对话失败: ' + error.message);
    }
}

/**
 * 结束会话
 */
async function endSession() {
    if (!currentSession) {
        showAlert('error', '当前没有活跃的对话会话');
        return;
    }
    
    // 使用统一的检查函数
    const shouldProceed = await checkAndPromptCompletion('结束会话');
    if (!shouldProceed) {
        return; // 用户选择留下完成任务
    }
    
    if (!confirm('确定要结束当前会话吗？未保存的对话将丢失。')) {
        return;
    }
    
    try {
        // 删除服务器端会话
        const response = await fetch(API_ENDPOINTS.sessionInfo(currentSession.session_id), {
            method: 'DELETE'
        });
        
        const data = await response.json();
        
        if (data.success) {
            showAlert('success', '会话已结束');
        }
        
    } catch (error) {
        console.error('结束会话失败:', error);
        showAlert('error', '结束会话失败: ' + error.message);
    } finally {
        // 重置状态
        currentSession = null;
        selectedPatient = null;
        lastAutoDiagnosis = null;
        isAutoDiagnosing = false;
        isEvaluationSubmitted = false;
        isDiagnosisSaved = false;
        clearSuggestedQuestions();
        updateAutoDiagnosisAvailability();
        updateRecommendButtonState();
        
        // 隐藏聊天界面
        document.getElementById('chatInterface').style.display = 'none';
        document.getElementById('welcomeScreen').style.display = 'block';

        // 清除选中状态
        document.querySelectorAll('#module-psychiatrist .patient-item').forEach(item => {
            item.classList.remove('selected');
        });
    }
}

/**
 * 显示提示信息
 */
function showAlert(type, message) {
    // 移除现有的提示
    const existingAlerts = document.querySelectorAll('.alert');
    existingAlerts.forEach(alert => {
        if (alert.classList.contains('alert-info') || 
            alert.classList.contains('alert-error') || 
            alert.classList.contains('alert-success')) {
            alert.remove();
        }
    });
    
    // 创建新提示
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type}`;
    alertDiv.textContent = message;
    
    // 添加到页面顶部
    const container = document.querySelector('.container');
    container.insertBefore(alertDiv, container.firstChild);
    
    // 3秒后自动移除
    setTimeout(() => {
        alertDiv.remove();
    }, 3000);
}

// 工具函数
function escapeHtml(text) {
    if (text === null || text === undefined) {
        return '';
    }
    return text
        .toString()
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function formatTime(timestamp) {
    return new Date(timestamp * 1000).toLocaleTimeString();
}

// 费用格式化功能已移除

// 错误处理
window.addEventListener('error', function(e) {
    console.error('JavaScript错误:', e.error);
    showAlert('error', '发生未知错误，请刷新页面重试');
});

window.addEventListener('unhandledrejection', function(e) {
    console.error('未处理的Promise拒绝:', e.reason);
    showAlert('error', '请求失败，请检查网络连接');
});

// ==================== 滚动功能 ====================

/**
 * 滚动到聊天区域顶部
 */
function scrollToTop() {
    const messagesContainer = document.getElementById('messages');
    if (messagesContainer) {
        messagesContainer.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    }
}

/**
 * 滚动到聊天区域底部
 */
function scrollToBottom() {
    const messagesContainer = document.getElementById('messages');
    if (messagesContainer) {
        messagesContainer.scrollTo({
            top: messagesContainer.scrollHeight,
            behavior: 'smooth'
        });
    }
}

/**
 * 检查滚动位置并显示/隐藏快速滚动按钮
 */
function updateScrollButtons() {
    const messagesContainer = document.getElementById('messages');
    const quickScrollButton = document.getElementById('quickScrollToBottom');
    const scrollToTopBtn = document.getElementById('scrollToTop');
    const scrollToBottomBtn = document.getElementById('scrollToBottom');
    
    if (!messagesContainer) return;
    
    const { scrollTop, scrollHeight, clientHeight } = messagesContainer;
    const isAtBottom = scrollTop + clientHeight >= scrollHeight - 10; // 10px 容错
    const isAtTop = scrollTop <= 10; // 10px 容错
    
    // 控制快速滚动到底部按钮的显示
    if (quickScrollButton) {
        if (isAtBottom) {
            quickScrollButton.classList.remove('show');
        } else {
            quickScrollButton.classList.add('show');
        }
    }
    
    // 控制滚动控制按钮的启用状态
    if (scrollToTopBtn) {
        scrollToTopBtn.disabled = isAtTop;
    }
    
    if (scrollToBottomBtn) {
        scrollToBottomBtn.disabled = isAtBottom;
    }
}

/**
 * 处理键盘快捷键
 */
function handleScrollKeyboard(e) {
    const messagesContainer = document.getElementById('messages');
    if (!messagesContainer) return;
    
    // 检查是否在输入框中
    if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') {
        return;
    }
    
    switch(e.key) {
        case 'Home':
            e.preventDefault();
            scrollToTop();
            break;
        case 'End':
            e.preventDefault();
            scrollToBottom();
            break;
        case 'PageUp':
            e.preventDefault();
            messagesContainer.scrollBy({
                top: -messagesContainer.clientHeight * 0.8,
                behavior: 'smooth'
            });
            updateScrollButtons();
            break;
        case 'PageDown':
            e.preventDefault();
            messagesContainer.scrollBy({
                top: messagesContainer.clientHeight * 0.8,
                behavior: 'smooth'
            });
            updateScrollButtons();
            break;
    }
}

/**
 * 初始化滚动监听器
 */
function initScrollListeners() {
    const messagesContainer = document.getElementById('messages');
    if (messagesContainer) {
        // 添加滚动事件监听器
        messagesContainer.addEventListener('scroll', updateScrollButtons);
        
        // 初始化按钮状态
        updateScrollButtons();
        
        // 监听窗口大小变化，重新计算按钮状态
        window.addEventListener('resize', updateScrollButtons);
        
        // 添加键盘快捷键支持
        document.addEventListener('keydown', handleScrollKeyboard);
        
        // 使用 MutationObserver 监听消息区域的变化
        const observer = new MutationObserver(function(mutations) {
            mutations.forEach(function(mutation) {
                if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                    // 有新消息添加时，稍微延迟更新按钮状态
                    setTimeout(updateScrollButtons, 100);
                }
            });
        });
        
        observer.observe(messagesContainer, {
            childList: true,
            subtree: true
        });
    }
}

/**
 * 智能滚动到底部（仅在用户接近底部时自动滚动）
 */
function smartScrollToBottom() {
    const messagesContainer = document.getElementById('messages');
    if (!messagesContainer) return;
    
    const { scrollTop, scrollHeight, clientHeight } = messagesContainer;
    const distanceFromBottom = scrollHeight - scrollTop - clientHeight;
    
    // 如果用户在底部附近（距离底部少于100px），则自动滚动到底部
    if (distanceFromBottom < 100) {
        scrollToBottom();
    }
    
    // 更新按钮状态
    updateScrollButtons();
}


// 对话记录状态更新功能已隐藏
/*
async function updateConversationLogStatus() {
    try {
        if (!psychosisSession || !psychosisSession.session_id) {
            return;
        }

        const response = await fetch(API_ENDPOINTS.conversationLog(psychosisSession.session_id));
        const data = await response.json();

        const statusElement = document.getElementById("conversationLogStatus");
        if (statusElement) {
            if (data.success) {
                const eventCount = data.data.total_events || 0;
                const logFile = data.data.log_file || "未知";
                statusElement.innerHTML = `已保存 ${eventCount} 条记录 <small>(${logFile})</small>`;
                statusElement.style.color = "#28a745";
            } else {
                statusElement.textContent = "保存失败";
                statusElement.style.color = "#dc3545";
            }
        }
    } catch (error) {
        console.error("更新对话记录状态失败:", error);
        const statusElement = document.getElementById("conversationLogStatus");
        if (statusElement) {
            statusElement.textContent = "状态未知";
            statusElement.style.color = "#ffc107";
        }
    }
}
*/

// ==================== Agent版本设置功能 ====================

/**
 * 获取Agent设置（从localStorage）
 */
function getAgentSettings() {
    const defaultSettings = {
        patientVersion: 'random',  // 默认随机
        doctorVersion: 'base'      // 默认base
    };
    
    try {
        const saved = localStorage.getItem('agentSettings');
        if (saved) {
            return { ...defaultSettings, ...JSON.parse(saved) };
        }
    } catch (error) {
        console.error('读取设置失败:', error);
    }
    
    return defaultSettings;
}

/**
 * 保存Agent设置（到localStorage）
 */
function saveAgentSettings(settings) {
    try {
        localStorage.setItem('agentSettings', JSON.stringify(settings));
        return true;
    } catch (error) {
        console.error('保存设置失败:', error);
        return false;
    }
}

/**
 * 随机选择Patient版本（cot或v1）
 */
function randomPatientVersion() {
    const versions = ['cot', 'v1'];
    return versions[Math.floor(Math.random() * versions.length)];
}

/**
 * 获取实际使用的Patient版本（处理随机情况）
 */
function getActualPatientVersion() {
    const settings = getAgentSettings();
    if (settings.patientVersion === 'random') {
        return randomPatientVersion();
    }
    return settings.patientVersion;
}

/**
 * 显示设置面板
 */
function showSettingsPanel() {
    const panel = document.getElementById('settingsPanel');
    const patientSelect = document.getElementById('globalPatientVersionSelect');
    const doctorSelect = document.getElementById('globalDoctorVersionSelect');
    
    // 加载当前设置
    const settings = getAgentSettings();
    if (patientSelect) patientSelect.value = settings.patientVersion;
    if (doctorSelect) doctorSelect.value = settings.doctorVersion;
    
    // 显示面板
    if (panel) {
        panel.style.display = 'flex';
    }
}

/**
 * 关闭设置面板
 */
function closeSettingsPanel() {
    const panel = document.getElementById('settingsPanel');
    if (panel) {
        panel.style.display = 'none';
    }
}

/**
 * 处理设置面板点击背景关闭
 */
function handleSettingsModalClick(event) {
    if (event.target.id === 'settingsPanel') {
        closeSettingsPanel();
    }
}

/**
 * 保存设置
 */
function saveSettings() {
    const patientSelect = document.getElementById('globalPatientVersionSelect');
    const doctorSelect = document.getElementById('globalDoctorVersionSelect');
    
    const settings = {
        patientVersion: patientSelect ? patientSelect.value : 'random',
        doctorVersion: doctorSelect ? doctorSelect.value : 'base'
    };
    
    if (saveAgentSettings(settings)) {
        showAlert('success', '设置已保存');
        closeSettingsPanel();
        
        // 更新侧边栏的提示（如果需要）
        updateVersionBadges();
    } else {
        showAlert('error', '保存设置失败');
    }
}

/**
 * 更新版本标记显示
 */
function updateVersionBadges() {
    const settings = getAgentSettings();
    
    // 在设置按钮上显示当前配置
    const settingsButton = document.querySelector('.settings-button');
    if (settingsButton) {
        const patientVersionText = settings.patientVersion === 'random' ? '随机' : settings.patientVersion.toUpperCase();
        const doctorVersionText = settings.doctorVersion.toUpperCase();
        
        // 更新按钮的title提示
        settingsButton.title = `Agent版本设置 (Patient: ${patientVersionText}, Doctor: ${doctorVersionText})`;
    }
    
    console.log('当前设置:', settings);
}

// ========== 诊断修正功能 ==========

/**
 * 切换诊断修正面板的显示/隐藏
 */
function toggleDiagnosisCorrection() {
    const selectorsDiv = document.getElementById('correctionSelectors');
    const toggleBtn = document.getElementById('toggleCorrectionBtn');
    
    if (selectorsDiv.style.display === 'none') {
        selectorsDiv.style.display = 'block';
        toggleBtn.textContent = '隐藏修正选项';
        
        // 初始化修正选择器
        populateCorrectionCategorySelect();
    } else {
        selectorsDiv.style.display = 'none';
        toggleBtn.textContent = '显示修正选项';
    }
}

/**
 * 填充诊断大类选择器
 */
function populateCorrectionCategorySelect() {
    const categorySelect = document.getElementById('correctionCategorySelect');
    
    if (!categorySelect) {
        console.error('修正大类选择器未找到');
        return;
    }
    
    // 清空现有选项
    categorySelect.innerHTML = '<option value="">请选择诊断大类...</option>';
    
    // 检查数据是否已加载
    if (!diagnosisGroupedData || diagnosisGroupedData.length === 0) {
        console.warn('诊断分组数据未加载');
        return;
    }
    
    // 填充大类选项
    diagnosisGroupedData.forEach((category, index) => {
        const option = document.createElement('option');
        option.value = index;
        option.textContent = `${category.range}: ${category.name}`;
        categorySelect.appendChild(option);
    });
}

/**
 * 更新修正子类选择器
 */
function updateCorrectionSubcategory() {
    const categorySelect = document.getElementById('correctionCategorySelect');
    const subcategorySelect = document.getElementById('correctionSubcategorySelect');
    const diagnosisSelect = document.getElementById('correctionDiagnosisSelect');
    
    const categoryIndex = categorySelect.value;
    
    // 清空子类和诊断下拉框
    subcategorySelect.innerHTML = '<option value="">请选择具体子类...</option>';
    diagnosisSelect.innerHTML = '<option value="">请先选择子类...</option>';
    diagnosisSelect.disabled = true;
    
    // 清空ICD代码
    document.getElementById('correctionIcdCode').value = '';
    
    if (categoryIndex === '') {
        subcategorySelect.disabled = true;
        return;
    }
    
    // 启用子类下拉框
    subcategorySelect.disabled = false;
    
    // 获取选中的大类数据
    const category = diagnosisGroupedData[parseInt(categoryIndex)];
    
    if (category && category.subcategories) {
        // 填充子类选项
        category.subcategories.forEach((subcat, index) => {
            const option = document.createElement('option');
            option.value = `${categoryIndex}-${index}`;
            option.textContent = `${subcat.code}: ${subcat.name}`;
            subcategorySelect.appendChild(option);
        });
    }
}

/**
 * 更新修正诊断项
 */
function updateCorrectionDiagnosis() {
    const subcategorySelect = document.getElementById('correctionSubcategorySelect');
    const diagnosisSelect = document.getElementById('correctionDiagnosisSelect');
    
    const subcategoryValue = subcategorySelect.value;
    
    // 清空诊断下拉框
    diagnosisSelect.innerHTML = '<option value="">请选择具体诊断...</option>';
    
    // 清空ICD代码
    document.getElementById('correctionIcdCode').value = '';
    
    if (subcategoryValue === '') {
        diagnosisSelect.disabled = true;
        return;
    }
    
    // 启用诊断下拉框
    diagnosisSelect.disabled = false;
    
    // 解析子类索引
    const [categoryIndex, subcatIndex] = subcategoryValue.split('-').map(Number);
    
    // 获取子类数据
    const category = diagnosisGroupedData[categoryIndex];
    const subcat = category.subcategories[subcatIndex];
    
    // 兼容新旧数据结构
    const diagnoses = subcat.diagnoses || subcat.items || [];
    
    if (diagnoses.length > 0) {
        // 填充具体诊断选项
        diagnoses.forEach(item => {
            const option = document.createElement('option');
            option.value = item.diagnosis;
            option.setAttribute('data-icd', item.icd_code);
            option.textContent = `${item.icd_code}: ${item.diagnosis}`;
            diagnosisSelect.appendChild(option);
        });
    }
}

/**
 * 更新修正ICD代码
 */
function updateCorrectionIcdCode() {
    const diagnosisSelect = document.getElementById('correctionDiagnosisSelect');
    const icdCodeInput = document.getElementById('correctionIcdCode');
    
    const selectedOption = diagnosisSelect.options[diagnosisSelect.selectedIndex];
    
    if (selectedOption && selectedOption.value) {
        const icdCode = selectedOption.getAttribute('data-icd') || '';
        const diagnosisName = selectedOption.value;
        
        icdCodeInput.value = `${icdCode} - ${diagnosisName}`;
    } else {
        icdCodeInput.value = '';
    }
}

/**
 * 清除诊断修正
 */
function clearDiagnosisCorrection() {
    const categorySelect = document.getElementById('correctionCategorySelect');
    const subcategorySelect = document.getElementById('correctionSubcategorySelect');
    const diagnosisSelect = document.getElementById('correctionDiagnosisSelect');
    const icdCodeInput = document.getElementById('correctionIcdCode');
    
    if (categorySelect) {
        categorySelect.value = '';
    }
    if (subcategorySelect) {
        subcategorySelect.innerHTML = '<option value="">请先选择诊断大类...</option>';
        subcategorySelect.disabled = true;
    }
    if (diagnosisSelect) {
        diagnosisSelect.innerHTML = '<option value="">请先选择子类...</option>';
        diagnosisSelect.disabled = true;
    }
    if (icdCodeInput) {
        icdCodeInput.value = '';
    }
}

/**
 * 获取诊断修正数据
 */
function getDiagnosisCorrectionData() {
    const diagnosisSelect = document.getElementById('correctionDiagnosisSelect');
    const icdCodeInput = document.getElementById('correctionIcdCode');
    
    if (!diagnosisSelect || !diagnosisSelect.value || !icdCodeInput || !icdCodeInput.value) {
        return null;
    }
    
    const selectedOption = diagnosisSelect.options[diagnosisSelect.selectedIndex];
    const icdCode = selectedOption.getAttribute('data-icd') || '';
    const diagnosisName = selectedOption.value;
    
    return {
        icd_code: icdCode,
        diagnosis_name: diagnosisName
    };
}

