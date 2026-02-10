/**
 * ICD-10 代码格式化和名称查询
 */

// 全局ICD-10数据缓存
let icd10DataCache = null;

/**
 * 加载ICD-10数据
 */
async function loadICD10Data() {
    if (icd10DataCache) {
        return icd10DataCache;
    }
    
    try {
        const response = await fetch('/data/icd10_mental_disorders_hierarchical.json');
        const data = await response.json();
        icd10DataCache = buildICD10Index(data);
        console.log('ICD-10数据加载成功，共', Object.keys(icd10DataCache).length, '条记录');
        return icd10DataCache;
    } catch (error) {
        console.error('加载ICD-10数据失败:', error);
        return {};
    }
}

/**
 * 构建ICD-10代码索引（快速查询）
 */
function buildICD10Index(data) {
    const index = {};
    
    if (data.categories && Array.isArray(data.categories)) {
        data.categories.forEach(category => {
            if (category.subcategories && Array.isArray(category.subcategories)) {
                category.subcategories.forEach(subcategory => {
                    if (subcategory.diagnoses && Array.isArray(subcategory.diagnoses)) {
                        subcategory.diagnoses.forEach(diagnosis => {
                            if (diagnosis.icd_code && diagnosis.diagnosis) {
                                index[diagnosis.icd_code.toUpperCase()] = diagnosis.diagnosis;
                            }
                        });
                    }
                });
            }
        });
    }
    
    return index;
}

/**
 * 从文本中提取ICD-10代码
 * 支持格式：F32.0, F32, F32.0;F33.1, F32.0；F33.1 等
 */
function extractICD10Codes(text) {
    if (!text) return [];
    
    // 匹配F开头，后跟数字，可能有小数点和更多数字的模式
    const pattern = /F\d{1,2}(?:\.\d{1,2})?/gi;
    const matches = text.match(pattern);
    
    if (!matches) return [];
    
    // 去重并转为大写
    const uniqueCodes = [...new Set(matches.map(code => code.toUpperCase()))];
    return uniqueCodes;
}

/**
 * 查询ICD-10代码对应的诊断名称
 */
function lookupICD10Name(code, icd10Data) {
    const upperCode = code.toUpperCase();
    return icd10Data[upperCode] || null;
}

/**
 * 格式化诊断结论文本，将ICD代码替换为"代码：名称"格式
 */
async function formatDiagnosisWithICD10Names(text) {
    if (!text || typeof text !== 'string') return text;
    
    // 加载ICD-10数据
    const icd10Data = await loadICD10Data();
    
    // 提取所有ICD-10代码
    const codes = extractICD10Codes(text);
    
    if (codes.length === 0) return text;
    
    // 对每个代码进行替换
    let formattedText = text;
    
    codes.forEach(code => {
        const name = lookupICD10Name(code, icd10Data);
        if (name) {
            // 创建正则表达式，匹配独立的代码（不在"代码："格式中）
            // 避免重复替换已经格式化的内容
            const regex = new RegExp(`(?<!：)\\b${code}\\b(?!：)`, 'gi');
            formattedText = formattedText.replace(regex, `${code}：${name}`);
        }
    });
    
    return formattedText;
}

/**
 * 自动格式化诊断结论文本框
 */
async function autoFormatDiagnosisResult() {
    const resultField = document.getElementById('diagnosisResultField');
    if (!resultField) return;
    
    const originalText = resultField.value.trim();
    if (!originalText) return;
    
    const formattedText = await formatDiagnosisWithICD10Names(originalText);
    
    // 只有在文本确实改变时才更新
    if (formattedText !== originalText) {
        resultField.value = formattedText;
        console.log('诊断结论已自动格式化，添加ICD-10诊断名称');
    }
}

/**
 * 提取纯代码列表（用于显示）
 */
async function getFormattedICD10List(text) {
    if (!text) return [];
    
    const icd10Data = await loadICD10Data();
    const codes = extractICD10Codes(text);
    
    return codes.map(code => {
        const name = lookupICD10Name(code, icd10Data);
        return {
            code: code,
            name: name || '未找到对应诊断',
            found: !!name
        };
    });
}

// 页面加载时预加载ICD-10数据
document.addEventListener('DOMContentLoaded', function() {
    loadICD10Data();
    console.log('ICD-10格式化模块已加载');
});

// 导出函数供其他模块使用
window.icd10Formatter = {
    formatDiagnosisWithICD10Names,
    autoFormatDiagnosisResult,
    extractICD10Codes,
    getFormattedICD10List,
    loadICD10Data
};





