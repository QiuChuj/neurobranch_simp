#!/bin/bash

# NeuroBranch 增量训练脚本
# 功能：分层抽样部分问题，并进行多轮增量训练
# 更新：对所有 benchmark 目录做分组，
#      仅当某基础问题类型实例数 > SAMPLE_THRESHOLD 时才按 SAMPLE_RATIO 抽样，
#      其他类型的所有实例全部加入训练。

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
# 抽样比例（仅在某基础问题类型实例数 > SAMPLE_THRESHOLD 时生效）
SAMPLE_RATIO=0.2
# 只有当某基础问题类型的实例数量 > 该阈值时，才对其做分层抽样
SAMPLE_THRESHOLD=100
# 抽样结果文件
SELECTED_FILE="/home/richard/project/neurobranch_simp/logs/selected_problems.txt"
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

# 分层抽样 / 全量选择
stratified_sampling() {
    log_message "开始分层抽样: 抽样比例 = $SAMPLE_RATIO，仅当某基础类型实例数 > $SAMPLE_THRESHOLD 时启用抽样"
    
    local data_root="$DATA_SOURCE/data"

    # 收集所有一级子目录（问题实例目录名）
    mapfile -t all_dirs < <(find "$data_root" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)

    if [ ${#all_dirs[@]} -eq 0 ]; then
        error_exit "在 $data_root 中未找到任何问题实例目录"
    fi

    # 使用关联数组将实例按 base_type 分组
    # base_type 的定义：将目录名末尾的 _数字 去掉后的部分
    # 例如 BMS_k3_n100_m429_0 -> BMS_k3_n100_m429
    declare -A groups

    for d in "${all_dirs[@]}"; do
        # 提取基础类型
        local base_type
        base_type=$(echo "$d" | sed 's/_[0-9][0-9]*$//')
        # 防御性：如果意外为空，则退回原名
        [ -z "$base_type" ] && base_type="$d"
        # 将该实例加入对应基础类型的列表（用空格分隔）
        groups["$base_type"]+="$d "
    done

    log_message "发现 ${#groups[@]} 种基础问题类型"

    # 清空之前的抽样结果文件
    : > "$SELECTED_FILE"

    # 遍历每一种基础类型，按规则决定抽样或全选
    for base_type in "${!groups[@]}"; do
        # 取出该基础类型下的所有实例名
        IFS=' ' read -r -a instances <<< "${groups[$base_type]}"
        local type_count=${#instances[@]}

        if (( type_count > SAMPLE_THRESHOLD )); then
            # 该基础类型实例数大于阈值，执行分层抽样
            local sample_count
            sample_count=$(echo "$type_count * $SAMPLE_RATIO" | bc | awk '{printf "%.0f\n", $1}')
            # 理论上 type_count > 100 且 SAMPLE_RATIO>0 时 sample_count >= 1
            [ "$sample_count" -lt 1 ] && sample_count=1

            log_message "基础类型 $base_type: 共 $type_count 个实例，抽样 $sample_count 个"

            # 随机打乱实例并抽取所需数量
            printf "%s\n" "${instances[@]}" | shuf | head -n "$sample_count" >> "$SELECTED_FILE"
        else
            # 实例数不超过阈值，全部选入
            log_message "基础类型 $base_type: 共 $type_count 个实例（≤ $SAMPLE_THRESHOLD），全部选入"
            printf "%s\n" "${instances[@]}" >> "$SELECTED_FILE"
        fi
    done

    # 打乱最终选中的问题顺序（避免训练顺序偏差）
    shuf "$SELECTED_FILE" > "${SELECTED_FILE}.tmp"
    mv "${SELECTED_FILE}.tmp" "$SELECTED_FILE"
    
    local total_selected
    total_selected=$(wc -l < "$SELECTED_FILE")
    log_message "分层抽样完成，共选择 $total_selected 个问题实例"
}

# 清理训练目录函数
clean_training_dirs() {
    log_message "清理训练目录..."
    rm -rf "$TRAIN_DATA_DIR"/* "$TRAIN_LABEL_DIR"/*
}

# 复制数据到训练目录
copy_problem_data() {
    local problem_name="$1"
    log_message "复制问题数据: $problem_name"
    
    # 复制data数据：将源子文件夹内的所有文件复制到目标data目录根下
    local source_data_dir="$DATA_SOURCE/data/$problem_name"
    if [ -d "$source_data_dir" ]; then
        # 检查源目录是否有文件
        if ls "$source_data_dir"/* >/dev/null 2>&1; then
            cp "$source_data_dir"/* "$TRAIN_DATA_DIR/" 2>/dev/null || {
                find "$source_data_dir" -maxdepth 1 -type f -exec cp {} "$TRAIN_DATA_DIR/" \; \
                    || error_exit "复制data数据失败: $problem_name"
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
                find "$source_label_dir" -maxdepth 1 -type f -exec cp {} "$TRAIN_LABEL_DIR/" \; \
                    || error_exit "复制label数据失败: $problem_name"
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
    
    if command -v jq >/dev/null 2>&1; then
        jq --argjson new_model "$is_new_model" '.new_model = $new_model' "$CONFIG_FILE" \
            > "${CONFIG_FILE}.tmp" && mv "${CONFIG_FILE}.tmp" "$CONFIG_FILE"
    else
        sed -i "s/\"new_model\":.*$/\"new_model\": $is_new_model/g" "$CONFIG_FILE"
    fi
}

# 执行训练函数
run_training() {
    local round="$1"
    local problem_name="$2"
    
    log_message "开始第 $round 轮训练，问题: $problem_name"
    
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
    
    # 执行分层抽样 / 全量选择
    stratified_sampling
    
    # 读取抽样结果
    mapfile -t selected_problems < "$SELECTED_FILE"
    
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
        copy_problem_data "${selected_problems[i]}" || { 
            log_message "跳过问题 ${selected_problems[i]}"
            continue
        }
        run_training "$round" "${selected_problems[i]}"
    done
    
    # 清理临时文件
    rm -f "$SELECTED_FILE"
    
    log_message "=== 所有训练轮次完成！共完成了 ${#selected_problems[@]} 轮训练 ==="
    log_message "训练日志已保存至: $LOG_FILE"
}

trap 'error_exit "脚本被用户中断"' INT TERM
main "$@"