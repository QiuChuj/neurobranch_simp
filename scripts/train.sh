#!/bin/bash

# NeuroBranch 增量训练脚本
# 功能：分层抽样20%的问题，并进行多轮增量训练

# ========================
# 配置参数区（请根据实际情况修改）
# ========================

# 数据源路径
DATA_SOURCE="/home/richard/project/neurobranch_train_data/neurobranch_simp"
# 训练数据目标路径
TRAIN_DATA_DIR="/home/richard/project/neurobranch_simp/dimacs/data"
TRAIN_LABEL_DIR="/home/richard/project/neurobranch_simp/dimacs/label"
# 配置文件路径
CONFIG_FILE="/home/richard/project/neurobranch_simp/configs/params.json"
# 训练脚本路径
TRAIN_SCRIPT="/home/richard/project/neurobranch_simp/python/train.py"
# 抽样比例 (20%)
SAMPLE_RATIO=0.2
# 日志文件
LOG_FILE="/home/richard/project/neurobranch_simp/logs/training.log"

# ========================
# 函数定义区
# ========================

# 日志记录函数
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# 错误处理函数
error_exit() {
    log_message "错误: $1"
    exit 1
}

# 检查路径是否存在
check_paths() {
    log_message "检查路径和依赖..."
    
    [ -d "$DATA_SOURCE" ] || error_exit "数据源路径不存在: $DATA_SOURCE"
    [ -f "$CONFIG_FILE" ] || error_exit "配置文件不存在: $CONFIG_FILE"
    [ -f "$TRAIN_SCRIPT" ] || error_exit "训练脚本不存在: $TRAIN_SCRIPT"
    
    # 创建训练目录（如果不存在）
    mkdir -p "$TRAIN_DATA_DIR" "$TRAIN_LABEL_DIR"
    
    log_message "路径检查完成"
}

stratified_sampling() {
    log_message "开始分层抽样，抽样比例: $SAMPLE_RATIO"
    
    # 1. 获取所有基础问题类型（关键修正：去掉末尾数字编号后去重）
    local base_types=($(find "$DATA_SOURCE/data" -maxdepth 1 -type d -name "*_*" | sed 's|.*/||' | sed 's/_[0-9][0-9]*$//' | sort | uniq))
    # 注：sed 's/_[0-9][0-9]*$//' 会移除目录名末尾的 _数字 部分（如 BMS_k3_n100_m429_0 -> BMS_k3_n100_m429）
    
    if [ ${#base_types[@]} -eq 0 ]; then
        error_exit "在 $DATA_SOURCE/data 中未找到任何问题类型"
    fi
    
    log_message "发现 ${#base_types[@]} 种基础问题类型: ${base_types[*]}"
    
    # 清空之前的抽样结果
    > "/home/richard/project/neurobranch_simp/logs/selected_problems.txt"
    
    # 2. 对每种基础问题类型进行抽样
    for base_type in "${base_types[@]}"; do
        # 获取该基础类型下的所有具体问题实例（完整目录名）
        local instances=($(find "$DATA_SOURCE/data" -name "${base_type}_*" -type d | sed 's|.*/||' | sort))
        
        if [ ${#instances[@]} -eq 0 ]; then
            log_message "警告: 基础类型 $base_type 下未找到实例，跳过"
            continue
        fi
        
        # 计算该类型需要抽样的数量（至少1个）
        local type_count=${#instances[@]}
        local sample_count=$(echo "$type_count * $SAMPLE_RATIO" | bc | awk '{printf "%.0f\n", $1}')
        [ $sample_count -eq 0 ] && sample_count=1
        
        log_message "基础类型 $base_type: 共 $type_count 个实例，抽样 $sample_count 个"
        
        # 随机抽样：使用 shuf 打乱实例顺序并抽取所需数量[7,8](@ref)
        printf "%s\n" "${instances[@]}" | shuf | head -n "$sample_count" >> "/home/richard/project/neurobranch_simp/logs/selected_problems.txt"
    done
    
    # 打乱最终选中的问题顺序（避免训练顺序偏差）
    shuf "/home/richard/project/neurobranch_simp/logs/selected_problems.txt" > "/home/richard/project/neurobranch_simp/logs/selected_problems.tmp"
    mv "/home/richard/project/neurobranch_simp/logs/selected_problems.tmp" "/home/richard/project/neurobranch_simp/logs/selected_problems.txt"
    
    local total_selected=$(wc -l < "/home/richard/project/neurobranch_simp/logs/selected_problems.txt")
    log_message "分层抽样完成，共选择 $total_selected 个问题实例"
}

# 清理训练目录函数
clean_training_dirs() {
    log_message "清理训练目录..."
    rm -rf "$TRAIN_DATA_DIR"/* "$TRAIN_LABEL_DIR"/*
}

# 复制数据到训练目录（修正版）
copy_problem_data() {
    local problem_name="$1"
    log_message "复制问题数据: $problem_name"
    
    # 复制data数据：将源子文件夹内的所有文件复制到目标data目录根下
    local source_data_dir="$DATA_SOURCE/data/$problem_name"
    if [ -d "$source_data_dir" ]; then
        # 检查源目录是否有文件
        if ls "$source_data_dir"/* >/dev/null 2>&1; then
            # 使用cp命令复制所有文件到目标目录根下 [1,5](@ref)
            cp "$source_data_dir"/* "$TRAIN_DATA_DIR/" 2>/dev/null || {
                # 如果通配符复制失败，尝试使用find命令更安全地复制 [4,5](@ref)
                find "$source_data_dir" -maxdepth 1 -type f -exec cp {} "$TRAIN_DATA_DIR/" \; || error_exit "复制data数据失败: $problem_name"
            }
            log_message "成功复制 $(ls "$source_data_dir" | wc -l) 个data文件"
        else
            log_message "警告: data目录为空 $source_data_dir"
            return 1
        fi
    else
        log_message "警告: data目录不存在 $source_data_dir"
        return 1
    fi
    
    # 复制label数据：将源子文件夹内的所有文件复制到目标label目录根下
    local source_label_dir="$DATA_SOURCE/label/$problem_name"
    if [ -d "$source_label_dir" ]; then
        if ls "$source_label_dir"/* >/dev/null 2>&1; then
            cp "$source_label_dir"/* "$TRAIN_LABEL_DIR/" 2>/dev/null || {
                find "$source_label_dir" -maxdepth 1 -type f -exec cp {} "$TRAIN_LABEL_DIR/" \; || error_exit "复制label数据失败: $problem_name"
            }
            log_message "成功复制 $(ls "$source_label_dir" | wc -l) 个label文件"
        else
            log_message "警告: label目录为空 $source_label_dir"
            return 1
        fi
    else
        log_message "警告: label目录不存在 $source_label_dir"
        return 1
    fi
    
    return 0
}

# 修改配置文件函数
update_training_config() {
    local is_new_model="$1"
    log_message "更新训练配置: new_model = $is_new_model"
    
    # 使用sed修改JSON配置[1](@ref)
    if command -v jq >/dev/null 2>&1; then
        # 如果有jq工具，使用jq更安全地修改JSON
        jq --argjson new_model "$is_new_model" '.new_model = $new_model' "$CONFIG_FILE" > "${CONFIG_FILE}.tmp" && mv "${CONFIG_FILE}.tmp" "$CONFIG_FILE"
    else
        # 使用sed作为备选方案
        sed -i "s/\"new_model\":.*$/\"new_model\": $is_new_model/g" "$CONFIG_FILE"
    fi
}

# 执行训练函数
run_training() {
    local round="$1"
    local problem_name="$2"
    
    log_message "开始第 $round 轮训练，问题: $problem_name"
    
    # 执行Python训练脚本
    cd "$(dirname "$TRAIN_SCRIPT")" || error_exit "无法切换到训练脚本目录"
    
    if python3 "$(basename "$TRAIN_SCRIPT")" 2>&1 | tee -a "$LOG_FILE"; then
        log_message "第 $round 轮训练完成"
    else
        error_exit "第 $round 轮训练失败"
    fi
}

# ========================
# 主执行流程
# ========================

main() {
    log_message "=== NeuroBranch 增量训练开始 ==="
    
    # 检查环境
    check_paths
    
    # 执行分层抽样
    stratified_sampling
    
    # 读取抽样结果
    mapfile -t selected_problems < "/home/richard/project/neurobranch_simp/logs/selected_problems.txt"
    
    if [ ${#selected_problems[@]} -eq 0 ]; then
        error_exit "没有抽到任何问题实例，无法开始训练"
    fi
    
    log_message "将按以下顺序进行增量训练:"
    printf '%s\n' "${selected_problems[@]}" | tee -a "$LOG_FILE"
    
    # 第一轮训练（创建新模型）
    log_message "=== 开始第一轮训练（创建新模型）==="
    clean_training_dirs
    copy_problem_data "${selected_problems[0]}" || error_exit "第一轮数据复制失败"
    update_training_config "true,"
    run_training 1 "${selected_problems[0]}"
    
    # 后续轮次训练（增量训练）
    update_training_config "false,"
    
    for ((i=1; i<${#selected_problems[@]}; i++)); do
        round=$((i+1))
        log_message "=== 开始第 $round 轮训练（增量训练）==="
        clean_training_dirs
        copy_problem_data "${selected_problems[i]}" || { log_message "跳过问题 ${selected_problems[i]}"; continue; }
        run_training "$round" "${selected_problems[i]}"
    done
    
    # 清理临时文件
    rm -f "/home/richard/project/neurobranch_simp/logs/selected_problems.txt"
    
    log_message "=== 所有训练轮次完成！共完成了 ${#selected_problems[@]} 轮训练 ==="
    log_message "训练日志已保存至: $LOG_FILE"
}

# 执行主函数（带有错误处理）
trap 'error_exit "脚本被用户中断"' INT TERM
main "$@"