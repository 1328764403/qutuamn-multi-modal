"""
可视化量子混合模型的结构
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def visualize_quantum_circuit(n_qubits=4, n_layers=2, save_path='quantum_circuit.png'):
    """可视化量子电路结构"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 绘制量子比特线
    qubit_spacing = 1.0
    qubit_positions = [i * qubit_spacing for i in range(n_qubits)]
    
    for i, pos in enumerate(qubit_positions):
        ax.plot([0, 10], [pos, pos], 'k-', linewidth=2, alpha=0.3)
        ax.text(-0.5, pos, f'q{i}', fontsize=12, ha='right', va='center')
    
    # 绘制数据编码层
    x_encode = 1
    for i, pos in enumerate(qubit_positions):
        circle = plt.Circle((x_encode, pos), 0.15, color='blue', fill=True)
        ax.add_patch(circle)
        ax.text(x_encode, pos, 'RY', fontsize=8, ha='center', va='center', color='white', weight='bold')
    
    ax.text(x_encode, -0.5, '数据编码', fontsize=10, ha='center', weight='bold')
    
    # 绘制变分层和纠缠层
    layer_width = 1.5
    for layer in range(n_layers):
        x_start = 2.5 + layer * layer_width * 2
        
        # 变分层
        for i, pos in enumerate(qubit_positions):
            rect = FancyBboxPatch((x_start - 0.2, pos - 0.2), 0.4, 0.4,
                                  boxstyle="round,pad=0.05", 
                                  facecolor='green', edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            ax.text(x_start, pos, 'Rot', fontsize=7, ha='center', va='center', color='white', weight='bold')
        
        ax.text(x_start, -0.5, f'变分层{layer+1}', fontsize=9, ha='center')
        
        # 纠缠层
        x_entangle = x_start + layer_width
        for i in range(n_qubits - 1):
            y1 = qubit_positions[i]
            y2 = qubit_positions[i + 1]
            
            # CNOT门
            circle = plt.Circle((x_entangle, y1), 0.15, color='red', fill=True)
            ax.add_patch(circle)
            ax.text(x_entangle, y1, '•', fontsize=20, ha='center', va='center', color='white')
            
            # 控制线
            ax.plot([x_entangle, x_entangle], [y1, y2], 'r-', linewidth=2)
            
            # 目标门
            plus = mpatches.RegularPolygon((x_entangle, y2), 4, radius=0.15, 
                                          orientation=np.pi/4, facecolor='red', edgecolor='black')
            ax.add_patch(plus)
            ax.text(x_entangle, y2, '+', fontsize=12, ha='center', va='center', color='white', weight='bold')
        
        ax.text(x_entangle, -0.5, f'纠缠层{layer+1}', fontsize=9, ha='center')
    
    # 绘制测量层
    x_measure = 2.5 + n_layers * layer_width * 2
    for i, pos in enumerate(qubit_positions):
        # 测量符号
        ax.plot([x_measure, x_measure + 0.3], [pos, pos], 'k-', linewidth=2)
        ax.plot([x_measure + 0.3, x_measure + 0.5], [pos - 0.1, pos], 'k-', linewidth=2)
        ax.plot([x_measure + 0.3, x_measure + 0.5], [pos + 0.1, pos], 'k-', linewidth=2)
        ax.text(x_measure + 0.7, pos, f'⟨Z{i}⟩', fontsize=10, ha='left', va='center')
    
    ax.text(x_measure, -0.5, '测量', fontsize=10, ha='center', weight='bold')
    
    ax.set_xlim(-1, 10)
    ax.set_ylim(-1, n_qubits)
    ax.axis('off')
    ax.set_title('量子融合层电路结构', fontsize=14, weight='bold', pad=20)
    
    # 添加图例
    legend_elements = [
        mpatches.Patch(color='blue', label='数据编码 (RY)'),
        mpatches.Patch(color='green', label='变分旋转 (Rot)'),
        mpatches.Patch(color='red', label='纠缠门 (CNOT)'),
        mpatches.Patch(color='black', label='测量 (Pauli-Z)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"量子电路图已保存到: {save_path}")
    plt.close()


def visualize_model_architecture(save_path='quantum_model_architecture.png'):
    """可视化整个量子混合模型架构"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 定义位置
    y_modalities = [8, 6, 4]
    y_encoded = 7
    y_quantum = 5.5
    y_cross = 3.5
    y_output = 2
    
    # 输入模态
    mod_colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
    for i, (y, color) in enumerate(zip(y_modalities, mod_colors)):
        rect = FancyBboxPatch((i*3 - 0.8, y - 0.3), 1.6, 0.6,
                              boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(i*3, y, f'模态{i+1}', fontsize=11, ha='center', va='center', weight='bold')
    
    # 经典编码器
    for i in range(3):
        rect = FancyBboxPatch((i*3 - 0.6, y_encoded - 0.25), 1.2, 0.5,
                              boxstyle="round,pad=0.05",
                              facecolor='#FFE66D', edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(i*3, y_encoded, '经典编码', fontsize=9, ha='center', va='center')
    
    # 量子融合层
    for i in range(3):
        rect = FancyBboxPatch((i*3 - 0.6, y_quantum - 0.25), 1.2, 0.5,
                              boxstyle="round,pad=0.05",
                              facecolor='#A8E6CF', edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(i*3, y_quantum, '量子融合', fontsize=9, ha='center', va='center')
    
    # 连接线：输入到编码器
    for i in range(3):
        ax.arrow(i*3, y_modalities[i] - 0.3, 0, -0.45,
                head_width=0.15, head_length=0.1, fc='black', ec='black')
    
    # 连接线：编码器到量子融合
    for i in range(3):
        ax.arrow(i*3, y_encoded - 0.25, 0, -0.25,
                head_width=0.15, head_length=0.1, fc='black', ec='black')
    
    # 跨模态量子纠缠
    rect = FancyBboxPatch((1 - 1.5, y_cross - 0.3), 3.0, 0.6,
                          boxstyle="round,pad=0.1",
                          facecolor='#FF8B94', edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(1, y_cross, '跨模态量子纠缠', fontsize=11, ha='center', va='center', weight='bold')
    
    # 连接线：量子融合到跨模态
    for i in range(3):
        ax.arrow(i*3, y_quantum - 0.25, 1 - i*3, y_cross - y_quantum + 0.25,
                head_width=0.15, head_length=0.1, fc='black', ec='black')
    
    # 输出层
    rect = FancyBboxPatch((1 - 1.0, y_output - 0.25), 2.0, 0.5,
                         boxstyle="round,pad=0.05",
                         facecolor='#C7CEEA', edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(1, y_output, '经典输出层', fontsize=11, ha='center', va='center', weight='bold')
    
    # 连接线：跨模态到输出
    ax.arrow(1, y_cross - 0.3, 0, -0.45,
            head_width=0.2, head_length=0.15, fc='black', ec='black', linewidth=2)
    
    # 输出
    ax.text(1, y_output - 0.6, '预测结果', fontsize=10, ha='center', va='top', weight='bold')
    
    # 添加说明
    ax.text(-2, 8.5, '输入层', fontsize=10, weight='bold')
    ax.text(-2, 7, '经典处理', fontsize=10, weight='bold')
    ax.text(-2, 5.5, '量子处理', fontsize=10, weight='bold')
    ax.text(-2, 3.5, '跨模态融合', fontsize=10, weight='bold')
    ax.text(-2, 2, '输出层', fontsize=10, weight='bold')
    
    ax.set_xlim(-3, 5)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('量子混合模型架构', fontsize=16, weight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"模型架构图已保存到: {save_path}")
    plt.close()


def visualize_data_flow(save_path='quantum_data_flow.png'):
    """可视化数据流"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    stages = ['经典编码', '量子融合', '跨模态纠缠']
    colors = ['#FFE66D', '#A8E6CF', '#FF8B94']
    
    for idx, (ax, stage, color) in enumerate(zip(axes, stages, colors)):
        # 输入
        input_rect = FancyBboxPatch((0.1, 0.3), 0.3, 0.4,
                                   boxstyle="round,pad=0.05",
                                   facecolor='lightgray', edgecolor='black')
        ax.add_patch(input_rect)
        ax.text(0.25, 0.5, '输入', fontsize=9, ha='center', va='center')
        
        # 处理
        process_rect = FancyBboxPatch((0.5, 0.2), 0.3, 0.6,
                                     boxstyle="round,pad=0.05",
                                     facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(process_rect)
        ax.text(0.65, 0.5, stage, fontsize=10, ha='center', va='center', weight='bold')
        
        # 输出
        output_rect = FancyBboxPatch((0.9, 0.3), 0.3, 0.4,
                                    boxstyle="round,pad=0.05",
                                    facecolor='lightblue', edgecolor='black')
        ax.add_patch(output_rect)
        ax.text(1.05, 0.5, '输出', fontsize=9, ha='center', va='center')
        
        # 箭头
        ax.arrow(0.4, 0.5, 0.1, 0, head_width=0.05, head_length=0.03, fc='black', ec='black')
        ax.arrow(0.8, 0.5, 0.1, 0, head_width=0.05, head_length=0.03, fc='black', ec='black')
        
        # 维度标注
        if idx == 0:
            ax.text(0.25, 0.15, '(B, d)', fontsize=8, ha='center')
            ax.text(0.65, 0.1, '(B, h)', fontsize=8, ha='center')
            ax.text(1.05, 0.15, '(B, h)', fontsize=8, ha='center')
        elif idx == 1:
            ax.text(0.25, 0.15, '(B, h)', fontsize=8, ha='center')
            ax.text(0.65, 0.1, '量子态', fontsize=8, ha='center')
            ax.text(1.05, 0.15, '(B, h)', fontsize=8, ha='center')
        else:
            ax.text(0.25, 0.15, '(B, 3h)', fontsize=8, ha='center')
            ax.text(0.65, 0.1, '纠缠态', fontsize=8, ha='center')
            ax.text(1.05, 0.15, '(B, h)', fontsize=8, ha='center')
        
        ax.set_xlim(0, 1.4)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title(f'阶段 {idx+1}: {stage}', fontsize=11, weight='bold', pad=10)
    
    plt.suptitle('量子混合模型数据流', fontsize=14, weight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"数据流图已保存到: {save_path}")
    plt.close()


if __name__ == '__main__':
    print("生成量子混合模型可视化...")
    visualize_quantum_circuit(n_qubits=4, n_layers=2, save_path='quantum_circuit.png')
    visualize_model_architecture(save_path='quantum_model_architecture.png')
    visualize_data_flow(save_path='quantum_data_flow.png')
    print("\n所有可视化图表已生成！")







