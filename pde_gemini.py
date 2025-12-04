import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize # 用于热力图
import time
import requests
from openai import OpenAI
from openai import APIError


# --- 页面配置 ---
st.set_page_config(
    page_title="PDE 交互式学习实验室",
    page_icon="∫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 全局默认 API 配置 (用于初始化)
# ==========================================
# 模式 1: 免费 ChatGPT 兼容 API 配置 (隐藏)
DEFAULT_CHATGPT_KEY = "sk-G9x9qxNXuMJe05q92586F5751e3c43C09154B60e7414EaB1"
DEFAULT_CHATGPT_BASE_URL = "https://free.v36.cm/v1/"
DEFAULT_CHATGPT_MODEL = "gpt-4o-mini"

# 模式 2: DeepSeek API 配置 (需要用户 Key)
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"

# ==========================================
# 0. 习题数据字典 (用于习题板块) - 完整版
# ==========================================

EXERCISES = {
    "基础知识 (第 1 套)": [
        {
            "id": 1,
            "question": "热传导方程在数学上属于哪一类偏微分方程？",
            "options": ["椭圆型", "双曲型", "抛物线型", "混合型"],
            "answer": "抛物线型",
            "explanation": "热传导方程包含对时间的奇数阶导数（一阶），描述扩散过程，属于**抛物线型**。相关知识链接：[知识点链接：PDE分类]"
        },
        {
            "id": 2,
            "question": "描述波传播的波动方程，在数学上属于哪一类偏微分方程？",
            "options": ["椭圆型", "双曲型", "抛物线型", "混合型"],
            "answer": "双曲型",
            "explanation": "波动方程包含对时间的偶数阶导数（二阶），描述波动过程，属于**双曲型**。相关知识链接：[知识点链接：波动方程]"
        },
        {
            "id": 3,
            "question": "稳态（不含时间项）的热传导方程通常被称为？",
            "options": ["薛定谔方程", "纳维-斯托克斯方程", "泊松方程", "拉普拉斯方程"],
            "answer": "拉普拉斯方程",
            "explanation": "当 $\\frac{\\partial u}{\\partial t}=0$ 时，方程简化为拉普拉斯方程 $\\nabla^2 u = 0$。相关知识链接：[知识点链接：拉普拉斯方程]"
        },
        {
            "id": 4,
            "question": "$\\nabla^2 u$ 在方程中代表的物理意义是？",
            "options": ["梯度", "时间导数", "空间曲率/散度", "对流项"],
            "answer": "空间曲率/散度",
            "explanation": "它是拉普拉斯算子，在物理上描述了场的空间变化趋势（曲率或散度）。相关知识链接：[知识点链接：算子]"
        },
        {
            "id": 5,
            "question": "在有限差分法中，空间二阶导数通常至少需要几个相邻的网格点进行离散？",
            "options": ["2 个", "3 个", "4 个", "5 个"],
            "answer": "3 个",
            "explanation": "中心差分格式需要 $u_{i-1}, u_{i}, u_{i+1}$ 三个点来近似二阶导数。相关知识链接：[知识点链接：FDM]"
        },
    ],
    
    "进阶 FDM 基础应用 (第 2 套)": [
        {
            "id": 6,
            "question": "在一维热传导 FDM 显式格式中，迭代公式 $u_i^{n+1}$ 仅依赖于哪一时间步的数据？",
            "options": ["$u^{n+1}$ 步", "$u^{n}$ 步", "所有历史时间步", "边界条件"],
            "answer": "$u^{n}$ 步",
            "explanation": "显式格式的特点是当前时间步（$n+1$）的解可以直接从前一时间步（$n$）的数据计算得到，无需解方程组。相关知识链接：[知识点链接：显式FDM]"
        },
        {
            "id": 7,
            "question": "对于时间导数 $\\frac{\\partial u}{\\partial t}$，若采用**中心差分**格式进行离散，则该格式的精度是多少阶？",
            "options": ["一阶 $O(\\Delta t)$", "二阶 $O(\\Delta t^2)$", "三阶 $O(\\Delta t^3)$", "零阶"],
            "answer": "二阶 $O(\\Delta t^2)$",
            "explanation": "时间中心差分 $\\frac{u^{n+1}-u^{n-1}}{2\\Delta t}$ 具有二阶精度。但在显式 FDM 中，通常使用前向差分（一阶精度）。相关知识链接：[知识点链接：差分精度]"
        },
        {
            "id": 8,
            "question": "有限元法（FEM）的核心思想是将微分方程首先转化为哪种形式进行求解？",
            "options": ["解析解", "强形式", "特征值形式", "弱形式（积分形式）"],
            "answer": "弱形式（积分形式）",
            "explanation": "FEM 通过将微分方程乘以权函数并在求解域上积分，将其转化为积分形式（弱形式），降低了对解的光滑度要求。相关知识链接：[知识点链接：FEM弱形式]"
        },
        {
            "id": 9,
            "question": "在 FDM 显式格式中，如果时间步长 $\\Delta t$ **过大**，可能导致的结果是？",
            "options": ["收敛速度变慢", "数值解精度提高", "数值解发散（不稳定）", "计算效率提高"],
            "answer": "数值解发散（不稳定）",
            "explanation": "显式格式受 Courant-Friedrichs-Lewy (CFL) 条件限制，$\\Delta t$ 过大将破坏数值稳定性，导致解发散。相关知识链接：[知识点链接：CFL条件]"
        },
        {
            "id": 10,
            "question": "相比于 FDM，有限体积法（FVM）在流体力学（CFD）中更受欢迎的主要原因是？",
            "options": ["精度更高", "编程更简单", "更容易保证物理量的守恒性", "速度更快"],
            "answer": "更容易保证物理量的守恒性",
            "explanation": "FVM 是基于积分形式的守恒律推导的，天生具备在局部和全局上严格满足质量、动量和能量守恒的特性。相关知识链接：[知识点链接：FVM]"
        },
    ],
    
    "综合边界条件与稳定性判断 (第 3 套)": [
        {
            "id": 11,
            "question": "在热传导问题中，将边界处的温度**固定为已知常数**（例如 $u(L, t)=100$）属于哪种类型的边界条件？",
            "options": ["诺伊曼条件 (Neumann)", "柯西条件 (Cauchy)", "迪里赫利条件 (Dirichlet)", "周期性条件"],
            "answer": "迪里赫利条件 (Dirichlet)",
            "explanation": "迪里赫利条件指定边界上的**函数值**（即温度值）为已知。诺伊曼条件指定导数（即热通量）。相关知识链接：[知识点链接：边界条件]"
        },
        {
            "id": 12,
            "question": "在绝热边界上（无热量交换），应施加哪种边界条件？",
            "options": ["$u=0$", "$\\frac{\\partial u}{\\partial t}=0$", "$\\frac{\\partial u}{\\partial n}=0$", "$u=f(t)$"],
            "answer": "$\\frac{\\partial u}{\\partial n}=0$",
            "explanation": "绝热意味着边界上的热通量为零，热通量与温度的法向导数（$\\frac{\\partial u}{\\partial n}$）成正比，因此施加诺伊曼条件（零法向导数）。相关知识链接：[知识点链接：诺伊曼条件]"
        },
        {
            "id": 13,
            "question": "对于瞬态 PDE，若采用**隐式**有限差分格式求解，其在时间步长 $\\Delta t$ 方面的稳定性特点是？",
            "options": ["无条件稳定", "需满足CFL条件", "稳定性取决于空间步长", "无条件不稳定"],
            "answer": "无条件稳定",
            "explanation": "隐式格式（如 Crank-Nicolson）在理论上对任何 $\\Delta t$ 都是稳定的，尽管大的 $\\Delta t$ 会降低精度。相关知识链接：[知识点链接：隐式FDM]"
        },
        {
            "id": 14,
            "question": "泊松方程 $\\nabla^2 u = f$ 属于哪一类 PDE？它通常描述的是什么状态？",
            "options": ["抛物型；扩散状态", "椭圆型；稳态平衡", "双曲型；波动状态", "混合型；非线性状态"],
            "answer": "椭圆型；稳态平衡",
            "explanation": "泊松方程和拉普拉斯方程一样，不含时间项，描述系统达到稳定平衡时的状态，属于**椭圆型**。相关知识链接：[知识点链接：PDE分类]"
        },
        {
            "id": 15,
            "question": "物理信息神经网络（PINNs）中，$Loss_{physics}$ 项的计算需要利用深度学习框架的哪一项关键技术？",
            "options": ["蒙特卡洛采样", "稀疏矩阵求解器", "自动微分 (Auto-differentiation)", "L2正则化"],
            "answer": "自动微分 (Auto-differentiation)",
            "explanation": "PINNs 通过自动微分计算网络输出（$u$）对输入变量（$x, t$）的偏导数，从而计算 PDE 残差。相关知识链接：[知识点链接：PINNs原理]"
        },
    ],
}

# --- 侧边栏导航 ---
st.sidebar.title("🏠 导航")

menu = [
    "1. 基础知识 (Foundations) 📚",      # 添加 📚
    "2. 方程博物馆 (Equation Zoo) 🏛️",    # 添加 🏛️
    "3. 经典数值模拟 (FDM Demo) 💻",    # 添加 💻
    "4. 习题与测验 (Quizzes) 📝",       # 添加 📝
    "5. AI 求解 (PINNs & More) 🤖",     # 添加 🤖
    "6. 主观问题答疑 (Q&A Corner) 💬"    # 保持不变
]

choice = st.sidebar.selectbox("选择模块", menu)

st.sidebar.markdown("---")
st.sidebar.info("偏微分方程 (PDE) 教学原型")

# ==========================================
# 辅助函数: 模块 2 绘图与模拟
# ==========================================

def simulate_laplace():
    """使用有限差分法 (FDM) 模拟二维拉普拉斯方程 (稳态温度/电势)"""
    N = 50
    T = np.zeros((N, N))
    
    # 边界条件 (Dirichlet)
    T[:, 0] = 0        # 左边界
    T[:, -1] = 0       # 右边界
    T[0, :] = 100      # 上边界
    T[-1, :] = 0       # 下边界
    
    # 迭代求解 (Jacobi 迭代)
    for _ in range(500):
        T_new = T.copy()
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                T_new[i, j] = 0.25 * (T[i + 1, j] + T[i - 1, j] + T[i, j + 1] + T[i, j - 1])
        T = T_new

    # 绘图
    fig, ax = plt.subplots(figsize=(6, 5))
    c = ax.contourf(T, cmap='hot', levels=20)
    fig.colorbar(c, ax=ax, label='Potential / Temperature')
    ax.set_title('Laplace Equation (Steady State)')
    ax.set_xlabel('X Grid')
    ax.set_ylabel('Y Grid')
    return fig

def simulate_heat_transfer():
    """使用显式 FDM 模拟一维热传导方程 (动态扩散)"""
    L = 1.0  # 长度
    T = 1.0  # 总时间
    N = 50   # 空间网格点
    M = 1000 # 时间步数
    dx = L / (N - 1)
    dt = T / M
    alpha = 0.01  # 扩散系数
    
    # CFL 条件 (稳定性要求)
    if alpha * dt / dx**2 > 0.5:
        alpha = 0.5 * dx**2 / dt * 0.9  # 自动调整alpha确保稳定
        
    u = np.zeros(N)
    u[20:30] = 100  # 初始条件：中心加热
    
    # 时间迭代
    history = []
    for _ in range(M):
        un = u.copy()
        for i in range(1, N - 1):
            u[i] = un[i] + alpha * dt / dx**2 * (un[i + 1] - 2 * un[i] + un[i - 1])
        if _ % (M // 4) == 0 or _ == M - 1:
            history.append(u.copy())

    # 绘图
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, profile in enumerate(history):
        time_step = int(i * M / 4) if i < len(history) - 1 else M
        ax.plot(np.linspace(0, L, N), profile, label=f'Time Step {time_step}')
    
    ax.set_title('Heat Equation (1D Diffusion)')
    ax.set_xlabel('Position (x)')
    ax.set_ylabel('Temperature (u)')
    ax.legend()
    return fig

def simulate_wave_equation():
    """使用 FDM 模拟一维波动方程 (弦振动快照)"""
    L = 1.0; c = 1.0; T = 2.0; N = 100; M = 2000
    dx = L / (N - 1); dt = T / M
    
    r = c * dt / dx
    if r > 1.0: # CFL 稳定性检查
        dt = dx / c * 0.9
        M = int(T / dt) + 1
        r = c * dt / dx

    u = np.zeros(N)   # 当前时间层 u(i, j)
    u_prev = np.zeros(N) # 上一时间层 u(i, j-1)
    
    # 初始条件: 三角形波
    x = np.linspace(0, L, N)
    u[45:55] = np.linspace(0, 10, 10)
    u[50:] = u[50:][::-1] # 峰值在中间

    u_prev = u.copy() # 初始速度为零
    
    # 时间迭代 (使用蛙跳格式)
    history = []
    for m in range(M):
        u_next = np.zeros(N) # 下一时间层 u(i, j+1)
        for i in range(1, N - 1):
            u_next[i] = 2 * u[i] - u_prev[i] + r**2 * (u[i + 1] - 2 * u[i] + u[i - 1])
        u_prev = u.copy()
        u = u_next
        if m % (M // 5) == 0:
            history.append(u.copy())

    # 绘图
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, profile in enumerate(history):
        ax.plot(x, profile, label=f'Time {i * dt * (M // 5):.2f}s', alpha=0.7)

    ax.set_title('Wave Equation (1D String Vibration)')
    ax.set_xlabel('Position (x)')
    ax.set_ylabel('Displacement (u)')
    ax.set_ylim(-10, 10)
    ax.legend(loc='upper right')
    return fig

# ==========================================
# 辅助函数: 一维热传导模拟
# ==========================================

def run_1d_simulation(alpha, steps, initial_cond):
    """一维热传导方程模拟代码"""
    
    # --- 模拟设置 ---
    nx = 100  # 空间网格数
    dx = 1.0 / (nx - 1)
    
    # 自动计算满足稳定性条件的 dt
    # 稳定性条件: gamma = alpha * dt / dx**2 <= 0.5
    dt = 0.5 * dx**2 / alpha * 0.9 # 乘以0.9确保安全稳定
    
    x = np.linspace(0, 1, nx)
    u = np.zeros(nx)
    
    # 初始化
    if initial_cond == "高斯脉冲 (Gaussian)":
        u = np.exp(-100 * (x - 0.5)**2)
    elif initial_cond == "方波 (Square)":
        u[int(0.4*nx):int(0.6*nx)] = 1.0
    elif initial_cond == "随机 (Random)":
        u = np.random.rand(nx) * 0.5
        
    # 边界条件 (Dirichlet: 两端为0)
    u[0] = 0
    u[-1] = 0

    st.subheader("一维热传导模拟结果 (温度曲线)")
    chart_placeholder = st.empty()
    progress_bar = st.progress(0)
    
    for n in range(steps):
        # FDM 核心迭代 (一维显式格式)
        # u[1:-1] 是当前时间步的内部点
        # u[2:] - 2*u[1:-1] + u[:-2] 是空间二阶导数的差分近似
        gamma = alpha * dt / dx**2
        u[1:-1] = u[1:-1] + gamma * (u[2:] - 2*u[1:-1] + u[:-2])
        
        # 每隔几步更新一次图表，避免卡顿
        if n % 10 == 0:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(x, u, color='red', label=f'Time Step: {n}')
            ax.set_ylim(0, 1.1)
            ax.set_xlabel('Space (x)')
            ax.set_ylabel('Temperature (u)')
            ax.set_title(f'1D Heat Diffusion (Alpha={alpha}, $\\gamma={gamma:.4f}$)')
            ax.grid(True)
            ax.legend()
            
            # 在 Streamlit 中渲染 Matplotlib 图
            chart_placeholder.pyplot(fig)
            plt.close(fig) # 释放内存
            
            progress_bar.progress((n + 1) / steps)
            time.sleep(0.01) # 稍微暂停，产生动画效果
    
    st.success("一维模拟完成！")

# ==========================================
# 辅助函数: 二维热传导模拟 (骨架)
# ==========================================

def run_2d_simulation(N, M, alpha, initial_temp_type, boundary_type, steps):
    """二维热传导方程模拟的骨架代码"""
    st.subheader("二维热传导模拟结果 (Heatmap)")
    
    # 初始化网格
    dx, dy = 1.0/(N-1), 1.0/(M-1)
    # 为满足稳定性，dt通常需要很小
    dt = 0.9 * (dx**2 * dy**2) / (2 * alpha * (dx**2 + dy**2)) 
    u = np.zeros((N, M))
    
    # 设置初始条件 (Initial Temp.)
    if initial_temp_type == "中心热源":
        u[N//2 - 5:N//2 + 5, M//2 - 5:M//2 + 5] = 100.0
    elif initial_temp_type == "随机":
        u[1:-1, 1:-1] = np.random.rand(N-2, M-2) * 50.0
    # 其他初始条件...

    # 设置边界条件 (Boundary Cond.) (仅在循环外初始化一次)
    if boundary_type == "固定温度":
        u[0, :], u[-1, :] = 0, 0
        u[:, 0], u[:, -1] = 0, 0
    # 绝热或周期性边界条件需要在循环内处理

    # 绘图设置
    fig, ax = plt.subplots(figsize=(8, 6))
    norm = Normalize(vmin=0, vmax=100) # 假设最大温度为100
    
    heatmap_placeholder = st.empty()
    
    for n in range(steps):
        u_new = u.copy()
        
        # FDM 核心迭代 (二维显式格式)
        u_new[1:-1, 1:-1] = u[1:-1, 1:-1] + alpha * dt * (
            (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx**2 +
            (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) / dy**2
        )
        u = u_new
        
        # 边界条件 (需要重新应用)
        if boundary_type == "固定温度":
            u[0, :], u[-1, :] = 0, 0
            u[:, 0], u[:, -1] = 0, 0
        
        if n % 20 == 0: # 减少绘图频率以加速
            ax.clear()
            im = ax.imshow(u.T, origin='lower', cmap='hot', norm=norm)
            ax.set_title(f'Time Step: {n}')
            if n == 0: # 首次绘制时添加颜色条
                fig.colorbar(im, ax=ax, label='Temperature')
            
            heatmap_placeholder.pyplot(fig)
            plt.close(fig)
            time.sleep(0.01) # 模拟动画效果

    st.success(f"二维模拟完成，总步数: {steps}")

def simulate_poisson():
    """使用有限差分法 (FDM) 模拟二维泊松方程 (有源电势/温度)"""
    N = 50
    T = np.zeros((N, N))
    f = np.zeros((N, N))  # 源项 f(x)
    
    # 放置两个源/汇点
    f[N//3, N//3] = 100    # 正源 (热源/正电荷)
    f[2*N//3, 2*N//3] = -100 # 负源 (热汇/负电荷)

    # 边界条件 (Dirichlet): 边界保持为 0
    
    # 迭代求解 (Jacobi 迭代)
    for _ in range(1000):
        T_new = T.copy()
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                # 泊松方程的 FDM 离散化: T_new[i, j] = 0.25 * (T[i+1, j] + ... + f[i, j] * dx^2)
                T_new[i, j] = 0.25 * (T[i + 1, j] + T[i - 1, j] + T[i, j + 1] + T[i, j - 1] + f[i, j] * 1**2) 
        T = T_new

    # 绘图
    fig, ax = plt.subplots(figsize=(6, 5))
    c = ax.contourf(T, cmap='seismic', levels=20) # 使用seismic cmap来区分正负
    fig.colorbar(c, ax=ax, label='Potential / Temperature')
    ax.set_title('Poisson Equation (With Sources)')
    ax.set_xlabel('X Grid')
    ax.set_ylabel('Y Grid')
    return fig

def simulate_helmholtz():
    """使用有限差分法 (FDM) 模拟二维亥姆霍兹方程 (稳态波场)"""
    N = 50
    k = 5.0  # 波数 (Wave Number)
    
    # 矩阵 A (离散化的亥姆霍兹算子)
    A = np.zeros((N*N, N*N))
    b = np.zeros(N*N) # 源项 (设置为零，求解特征波)
    
    # 构建矩阵 A (五点差分)
    for i in range(N):
        for j in range(N):
            idx = i * N + j
            
            # 内部节点
            if 0 < i < N - 1 and 0 < j < N - 1:
                A[idx, idx] = 4 + k**2  # 中心点项 (2*Dxx + 2*Dyy + k^2)
                
                # 邻居点
                A[idx, (i + 1) * N + j] = -1 # T[i+1, j]
                A[idx, (i - 1) * N + j] = -1 # T[i-1, j]
                A[idx, i * N + (j + 1)] = -1 # T[i, j+1]
                A[idx, i * N + (j - 1)] = -1 # T[i, j-1]
            
            # 边界节点 (Dirichlet u=0)
            else:
                A[idx, idx] = 1.0 
    
    # 求解 (用于演示，我们简单设置一个初始激励并求解)
    b[N*N // 2] = 1.0 # 在中心点设置一个点源激励
    
    try:
        u_flat = np.linalg.solve(A, b)
        u = u_flat.reshape(N, N)
    except np.linalg.LinAlgError:
        u = np.zeros((N, N))
        
    # 绘图 (展示波场振幅)
    fig, ax = plt.subplots(figsize=(6, 5))
    c = ax.contourf(u, cmap='plasma', levels=20)
    fig.colorbar(c, ax=ax, label='Wave Amplitude')
    ax.set_title(f'Helmholtz Equation (k={k:.1f})')
    ax.set_xlabel('X Grid')
    ax.set_ylabel('Y Grid')
    return fig

def simulate_navier_stokes_cavity():
    """使用简化方法（方腔流）模拟纳维-斯托克斯方程 (稳态流场)"""
    N = 41 # 网格点
    max_iter = 500 # 迭代次数
    rho = 1.0 # 密度
    nu = 0.1 # 运动粘度 (用于雷诺数 Re=1/nu)
    
    # 初始化涡度 (omega) 和流函数 (psi)
    omega = np.zeros((N, N))
    psi = np.zeros((N, N))

    # 迭代求解 (简化方法)
    for _ in range(max_iter):
        omega_new = omega.copy()
        
        # 1. 求解涡度输运方程 (简化的时间步)
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                # 简化离散化，演示涡度扩散
                omega_new[i, j] = 0.25 * (omega[i+1, j] + omega[i-1, j] + omega[i, j+1] + omega[i, j-1])
        omega = omega_new

        # 2. 求解泊松方程 (用于流函数 psi)
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                psi[i, j] = 0.25 * (psi[i+1, j] + psi[i-1, j] + psi[i, j+1] + psi[i, j-1] + omega[i, j])

        # 3. 施加边界条件 (顶部移动的盖子)
        psi[:, 0] = 0; psi[:, N-1] = 0; psi[0, :] = 0; psi[N-1, :] = 0
        omega[N-1, :] = (psi[N-2, :] - psi[N-1, :]) * 2 / 1**2 + 10 # 顶部移动
        
    # 计算速度场 (u, v) 用于绘图
    u = np.zeros((N, N))
    v = np.zeros((N, N))
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            u[i, j] = (psi[i, j+1] - psi[i, j-1]) / 2 # d(psi)/dy
            v[i, j] = -(psi[i+1, j] - psi[i-1, j]) / 2 # -d(psi)/dx

    # 绘图 (流线图)
    Y, X = np.mgrid[0:N, 0:N]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.streamplot(X, Y, u, v, density=1.5, linewidth=None, color=psi, cmap='coolwarm')
    ax.set_title(f'Navier-Stokes (Lid-Driven Cavity Flow, Re≈{1/nu})')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    return fig


def simulate_schrodinger():
    """使用显式 FDM 模拟一维薛定谔方程 (粒子在势阱中的演化)"""
    N = 100 # 空间点
    T = 0.5 # 总时间
    dx = 1.0; dt = 0.001
    
    # 定义势能 V(x) (方势阱)
    x = np.linspace(-N/2, N/2, N)
    V = np.zeros(N)
    V[:N//4] = 1000 # 左边界墙
    V[3*N//4:] = 1000 # 右边界墙

    # 初始波包 (高斯波包)
    sigma = 5.0
    k0 = 1.0
    psi_real = np.exp(-(x / sigma)**2) * np.cos(k0 * x)
    psi_imag = np.exp(-(x / sigma)**2) * np.sin(k0 * x)
    
    # 时间迭代 (使用显式差分，需要非常小的 dt)
    for _ in range(int(T / dt)):
        # 计算下一时间步的实部和虚部
        psi_real_next = psi_real.copy()
        psi_imag_next = psi_imag.copy()
        
        for i in range(1, N - 1):
            Laplace_real = (psi_real[i+1] - 2*psi_real[i] + psi_real[i-1]) / dx**2
            Laplace_imag = (psi_imag[i+1] - 2*psi_imag[i] + psi_imag[i-1]) / dx**2
            
            # 离散化 (简化的 Crank-Nicolson 或 Euler-Forward 形式)
            # d(psi_real)/dt = -1 * (Laplace_imag + V * psi_imag)
            # d(psi_imag)/dt = 1 * (Laplace_real - V * psi_real)
            
            # 使用 Euler-Forward (显式，不稳定但简单演示)
            psi_real_next[i] = psi_real[i] - dt * (Laplace_imag - V[i] * psi_imag[i])
            psi_imag_next[i] = psi_imag[i] + dt * (Laplace_real - V[i] * psi_real[i])

        psi_real = psi_real_next
        psi_imag = psi_imag_next
        
        # 边界条件
        psi_real[0] = 0; psi_real[-1] = 0
        psi_imag[0] = 0; psi_imag[-1] = 0
        
    # 计算最终概率密度
    Prob_Density = psi_real**2 + psi_imag**2
    
    # 绘图
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, Prob_Density, label='Probability Density $|\Psi|^2$')
    ax.plot(x, V * 0.05, label='Potential V(x) (Scaled)', linestyle='--') # 缩放势能 V 以便绘图
    
    ax.set_title('Schrödinger Equation (Particle in Potential Well)')
    ax.set_xlabel('Position (x)')
    ax.set_ylabel('Probability Density')
    ax.legend()
    return fig

# ==========================================
# 辅助函数: 模拟 AI 回答 (需替换为真实 LLM API 调用)
# ==========================================
def simulate_ai_response(prompt):
    """根据用户输入，模拟一个关于 PDE 的回答"""
    # 这是一个占位符，用于演示聊天交互
    
    if "FDM" in prompt or "有限差分" in prompt:
        return "有限差分法（FDM）是一种通过将微分方程中的导数用代数差分近似来求解 PDE 的方法。它适用于规则网格，但处理复杂几何边界较为困难。您具体想了解 FDM 的哪种格式（如显式、隐式）？"
    elif "PINNs" in prompt or "物理信息" in prompt:
        return "PINNs（物理信息神经网络）是一种无需网格和大量标签数据的求解方法。它将 PDE 残差加入损失函数中，让神经网络在训练过程中遵守物理定律。它非常擅长解决反问题。您希望我提供一个 PINNs 解决反问题的例子吗？"
    elif "Navier-Stokes" in prompt or "纳维-斯托克斯" in prompt:
        return "纳维-斯托克斯方程是描述粘性流体动量守恒的核心方程。它是一个复杂的非线性 PDE 组，求解难度极大，传统上多采用有限体积法（FVM）进行离散化求解。"
    else:
        return "欢迎提出您关于偏微分方程、数值方法或 AI 求解的任何问题！请尽量具体地描述您想了解的概念，我会尽力为您解答。"

# ==========================================
# 辅助函数: 实际 API 调用 (使用 OpenAI SDK) (保持不变)
# ==========================================
def call_llm_api(prompt, api_key, base_url, model_name):
    """使用 OpenAI SDK 执行外部 LLM API 请求"""
    
    try:
        # DeepSeek 的 system message
        system_message = {"role": "system", "content": "你是一位精通偏微分方程（PDE）、数值分析和科学计算的专业助教。你的回答应准确、简洁、专业。"}
        
        # 1. 实例化 OpenAI 客户端
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=30.0
        )

        # 构造消息列表：只有 DeepSeek 默认需要 system 消息
        messages = [
            {"role": "user", "content": prompt}
        ]
        if model_name == DEFAULT_DEEPSEEK_MODEL:
            messages.insert(0, system_message)
            
        # 2. 调用 Chat Completion API
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.7,
            stream=False
        )
        
        # 3. 提取结果
        if completion.choices:
            return completion.choices[0].message.content
        else:
            return "API 响应无内容 (choices 列表为空)。"

    except APIError as e:
        return f"API 请求失败（{e.status_code} {e.code}）。请检查 Base URL, Key 或模型。\n错误详情：{e.message}"
    except Exception as e:
        return f"处理时发生未知错误：{e}"


# ==========================================
# 模块 1: 基础知识 (Foundations)
# ==========================================
if choice == "1. 基础知识 (Foundations) 📚":
    st.title("❓ 什么是偏微分方程 (PDE)?")
    st.markdown("""
    偏微分方程 (Partial Differential Equation, PDE) 是包含未知函数及其对多个自变量的偏导数的方程。
    它是描述自然界物理法则（如热、流体、波、量子力学）的通用语言。
    """)
    st.markdown("---")

    ## 1.1 通用形式与组成
    st.subheader("1.1 通用形式与组成")
    st.markdown("### 📝 通用形式")
    st.latex(r"""
    F(x_1, \dots, x_n, u, \frac{\partial u}{\partial x_1}, \dots, \frac{\partial^2 u}{\partial x_1^2}, \dots) = 0
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(r"**未知函数 $u$** $\rightarrow$ 通常代表物理量，如温度、压力、位移或波函数。")
    with col2:
        st.info(r"**自变量 $x, t$** $\rightarrow$ 通常代表空间坐标 $(x, y, z)$ 和时间 $t$。")
    with col3:
        st.info(r"💡 **偏导数** $\rightarrow$ 描述物理量随空间或时间的变化率 (如速度、加速度、梯度)。")
    
    st.markdown("---")

    ## 1.2 核心分类 (类型决定性质)
    st.subheader("1.2 核心分类 (类型决定性质)")
    st.markdown("PDE 通常根据其最高阶导数的系数，分为三类，这决定了信息传播的方式和求解的难度:")
    
    # 使用表格展示，增强对比和可读性
    table_data = {
        "类型": ["**椭圆型** (Elliptic)", "**抛物型** (Parabolic)", "**双曲型** (Hyperbolic)"],
        "信息传播": ["描述**平衡状态** (信息瞬间传播到全局)", "描述**扩散过程** (信息随时间逐渐平滑)", "描述**波动过程** (信息以有限速度传播)"],
        "物理例子": ["拉普拉斯方程 ($ \\nabla^2 u = 0 $)", "热传导方程 ($\\frac{\partial u}{\\partial t} = \\alpha \\nabla^2 u$)", "波动方程 ($\\frac{\partial^2 u}{\partial t^2} = c^2 \\nabla^2 u$ )"],
        "数学特征": ["只有空间导数，无时间项", "含时间一阶导数和空间二阶导数", "含时间二阶导数和空间二阶导数"]
    }
    st.table(table_data)

    st.markdown("---")
    
    ## 1.3 求解条件 (定解条件)
    st.subheader("1.3 求解条件 (定解条件)")
    st.markdown("求解 PDE 必须同时给定**定解条件**，以确定唯一的解。")
    
    col_ic, col_bc = st.columns(2)
    with col_ic:
        st.success("#### 初始条件 (Initial Conditions, IC)")
        st.markdown("* **适用:** 涉及时间 $t$ 的**动态方程** (抛物型、双曲型)。")
        st.markdown("* **作用:** 规定系统在 $t=0$ 时刻的初始状态。")
        st.latex(r"""
        u(x, t=0) = f(x)
        """)
        
    with col_bc:
        st.success("#### 边界条件 (Boundary Conditions, BC)")
        st.markdown("* **适用:** 涉及空间 $x$ 的所有方程。")
        st.markdown("* **作用:** 规定解在给定空间区域边界上的行为。")
        st.markdown("主要类型:")
        st.markdown("* **第一类 (Dirichlet):** 规定边界上的函数值 $u$。")
        st.markdown("* **第二类 (Neumann):** 规定边界上的法向导数 $\\frac{\\partial u}{\partial n}$ (通量)。")
        st.markdown("* **第三类 (Robin):** 规定函数值和导数的线性组合。")

# ==========================================
# 模块 2: 方程博物馆 (Equation Zoo) 🏛️
# ==========================================
elif choice == "2. 方程博物馆 (Equation Zoo) 🏛️":
    st.title("🏛️ 方程博物馆")
    st.write("这里展示了数学物理中最著名的方程。点击 **[查看模拟]** 按钮，可以观察这些方程的数值解行为。")
    st.markdown("---")

    # 使用 st.tabs 将方程分为静态和动态两类
    tab1, tab2 = st.tabs(["静态方程 (Time-Independent)", "动态方程 (Time-Dependent)"])

    # ------------------------------------------
    # Tab 1: 静态方程 (时间无关) (保持不变)
    # ------------------------------------------
    with tab1:
        # 1. 拉普拉斯方程 (Laplace Equation)
        st.subheader("1. 拉普拉斯方程 (Laplace Equation)")
        st.latex(r"\nabla^2 u = 0") 
        st.caption("描述: 在无源区域内的**稳态**分布，例如静电势、稳态温度分布。")
        
        if st.button("查看模拟 (拉普拉斯)"):
            with st.spinner("正在计算二维稳态解..."):
                fig_laplace = simulate_laplace()
                st.pyplot(fig_laplace)
        
        st.markdown("---")

        # 2. 泊松方程 (Poisson Equation)
        st.subheader("2. 泊松方程 (Poisson Equation)")
        st.latex(r"\nabla^2 u = f(\mathbf{x})")
        st.caption(r"描述: 在有源区域内的分布，例如由电荷密度 $f(\mathbf{x})$ 产生的静电势。")
        
        if st.button("查看模拟 (泊松方程)"):
            with st.spinner("正在计算二维有源稳态解..."):
                fig_poisson = simulate_poisson()
                st.pyplot(fig_poisson)
        
        st.markdown("---")
        
        # 3. 亥姆霍兹方程 (Helmholtz Equation)
        st.subheader("3. 亥姆霍兹方程 (Helmholtz Equation)")
        st.latex(r"(\nabla^2 + k^2) u = 0")
        st.caption("描述: 波动方程在**频域**上的形式，用于研究声波和电磁波的稳态传播和散射问题。")
        
        if st.button("查看模拟 (亥姆霍兹方程)"):
            with st.spinner("正在计算二维稳态波场..."):
                fig_helmholtz = simulate_helmholtz()
                st.pyplot(fig_helmholtz)

    # ------------------------------------------
    # Tab 2: 动态方程 (时间相关) (新增 NS 和薛定谔)
    # ------------------------------------------
    with tab2:
        # 1. 热传导方程 (Heat Equation)
        st.subheader("1. 热传导方程 (Heat Equation)")
        st.latex(r"\frac{\partial u}{\partial t} = \alpha \nabla^2 u") 
        st.caption(r"描述: 热量或物质如何在介质中扩散。$\alpha$ 是热扩散率，方程属于抛物型。")
        
        if st.button("查看模拟 (热传导)"):
            with st.spinner("正在计算一维热扩散过程..."):
                fig_heat = simulate_heat_transfer()
                st.pyplot(fig_heat)

        st.markdown("---")

        # 2. 波动方程 (Wave Equation)
        st.subheader("2. 波动方程 (Wave Equation)")
        st.latex(r"\frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u") 
        st.caption(r"描述: 声波、光波或弦的振动。信息以有限速度 $c$ 传播，方程属于双曲型。")
        
        if st.button("查看模拟 (波动方程)"):
            with st.spinner("正在计算一维弦振动过程..."):
                fig_wave = simulate_wave_equation()
                st.pyplot(fig_wave)

        st.markdown("---")

        # 3. 纳维-斯托克斯方程 (Navier-Stokes) <-- 新增模拟
        st.subheader("3. 纳维-斯托克斯方程 (Navier-Stokes)")
        st.latex(r"""
        \rho \left( \frac{\partial \mathbf{u}}{\partial t} + \mathbf{u} \cdot \nabla \mathbf{u} \right) = - \nabla p + \mu \nabla^2 \mathbf{u} + \mathbf{f}
        """)
        st.caption("描述: 粘性流体的动量守恒。这是流体力学 (CFD) 的核心，求解难度极大。")
        
        if st.button("查看模拟 (Navier-Stokes)"):
            with st.spinner("正在计算方腔流（简易 Navier-Stokes）..."):
                fig_ns = simulate_navier_stokes_cavity()
                st.pyplot(fig_ns)
        
        st.markdown("---")
        
        # 4. 薛定谔方程 (Schrödinger Equation) <-- 新增模拟
        st.subheader("4. 薛定谔方程 (Schrödinger Equation)")
        st.latex(r"i\hbar \frac{\partial \Psi}{\partial t} = \hat{H} \Psi")
        st.caption(r"描述: 量子力学中，波函数 $\Psi$ 随时间演化的基本方程。")
        
        if st.button("查看模拟 (薛定谔方程)"):
            with st.spinner("正在计算粒子概率密度演化..."):
                fig_schrodinger = simulate_schrodinger()
                st.pyplot(fig_schrodinger)

# ==========================================
# 模块 3: 经典数值模拟 (整合 1D 和 2D)
# ==========================================

elif choice == "3. 经典数值模拟 (FDM Demo) 💻":
    st.title("💻 经典数值模拟：FDM 演示")
    
    sim_type = st.radio("选择模拟类型", ["1D 热传导 (Heat Equation)", "2D 热传导 (Heatmap) "])
    
    if sim_type == "1D 热传导 (Heat Equation)":
        st.header("🔥 一维热传导方程模拟")
        st.latex(r"\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}")
        st.markdown("本模拟通过**显式有限差分法 (FDM)** 求解。注意 $\\alpha$ 较大或步数较多时可能导致数值不稳定。")
        
        # 1D 模拟的用户控件
        col_1d_c1, col_1d_c2, col_1d_c3 = st.columns(3)
        with col_1d_c1:
            alpha_1d = st.slider("热扩散率 $\\alpha$", 0.1, 1.0, 0.5)
        with col_1d_c2:
            steps_1d = st.slider("时间步数", 100, 1000, 500)
        with col_1d_c3:
            init_cond_1d = st.selectbox("初始条件", ["高斯脉冲 (Gaussian)", "方波 (Square)", "随机 (Random)"])
            
        st.markdown("---")
        
        if st.button("启动 1D 模拟 ▶️"):
            run_1d_simulation(alpha_1d, steps_1d, init_cond_1d)
        
    elif sim_type == "2D 热传导 (Heatmap) ":
        st.header("🔥🔥 二维热传导方程模拟")
        st.latex(r"\frac{\partial u}{\partial t} = \alpha (\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2})")
        st.markdown("本模拟通过**有限差分法 (FDM)** 求解二维瞬态热传导过程。拖动下方参数，观察温度场随时间的变化。")

        # 2D 模拟的用户控件 (与您提供的结构一致)
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            N = st.slider("网格尺寸 N (N x N)", 40, 100, 60)
            M = N # 简化为方格
        with col_c2:
            alpha_2d = st.slider("热扩散率 $\\alpha$", 0.05, 1.0, 0.2)
        with col_c3:
            steps_2d = st.slider("模拟步数", 100, 1000, 300, step=50)

        col_c4, col_c5 = st.columns(2)
        with col_c4:
            init_cond_2d = st.selectbox("初始温度分布", ["中心热源", "随机", "均匀"])
        with col_c5:
            bnd_cond_2d = st.selectbox("边界条件", ["固定温度", "绝热", "周期性"])
            
        st.markdown("---")
        
        # run_2d_simulation 函数骨架在整个文件中，此处为调用
        if st.button("启动 2D 模拟 ▶️"):
            run_2d_simulation(N, M, alpha_2d, init_cond_2d, bnd_cond_2d, steps_2d) 

# ==========================================
# 模块 4: 习题与测验 (新增)
# ==========================================
elif choice == "4. 习题与测验 (Quizzes) 📝":
    st.title("📝 习题与测验：巩固知识点")
    st.markdown("选择一套习题开始测验。测验包含 **3 套** 由浅入深的题目，每套 **5 题**。")

    quiz_set = st.selectbox("选择测验套数", list(EXERCISES.keys()))

    if quiz_set in EXERCISES:
        st.markdown("---")
        questions = EXERCISES[quiz_set]
        
        user_answers = {}
        
        # 渲染习题表单
        with st.form(key='quiz_form'):
            for q_data in questions:
                st.subheader(f"题号 {q_data['id']}. {q_data['question']}")
                user_answers[q_data['id']] = st.radio(
                    "选择你的答案:",
                    q_data['options'],
                    key=f"q_{q_data['id']}"
                )
                
            submitted = st.form_submit_button("提交答案并批改")
            
        # 提交后的反馈逻辑
        if submitted:
            st.markdown("## 批改结果")
            correct_count = 0
            
            for q_data in questions:
                user_ans = user_answers[q_data['id']]
                
                if user_ans == q_data['answer']:
                    correct_count += 1
                    st.success(f"✅ 题号 {q_data['id']}：恭喜！回答正确。")
                else:
                    st.error(f"❌ 题号 {q_data['id']}：很遗憾，答案不正确。")
                    st.markdown(f"**正确答案：** {q_data['answer']}")
                    st.markdown(f"**详细解答：** {q_data['explanation']}")
            
            st.markdown("---")
            if correct_count == len(questions):
                st.balloons()
                st.header(f"🎉 完美！您全部答对了 {correct_count}/{len(questions)} 题！")
            else:
                st.header(f"总分：您答对了 {correct_count}/{len(questions)} 题。")

# ==========================================
# 模块 5: AI 求解 (升级 - 增强代码和链接)
# ==========================================
elif choice == "5. AI 求解 (PINNs & More) 🤖":
    st.title("🤖 AI 求解器：前沿方法")
    st.markdown("传统的数值方法在处理高维或反问题时效率低下。AI/ML 方法提供了新的思路，尤其在科学计算 (SciML) 领域展现巨大潜力。")
    
    tab_pinn, tab_dgm, tab_fno, tab_surrogate = st.tabs([
        "1. PINNs (物理信息网络)", 
        "2. DGM (深度伽辽金)", 
        "3. 算子学习 (FNO/DeepONet)", 
        "4. 代理模型 (Surrogate)"
    ])
    
    with tab_pinn:
        st.subheader("1. 物理信息神经网络 (PINNs)")
        st.info("核心思想：将 PDE、初始条件和边界条件嵌入到神经网络的**损失函数**中，通过**自动微分**实现无网格求解。")
        st.markdown("""
        * **优势：** 无需大量标签数据；可用于反问题求解（参数识别）。
        * **应用：** 复杂流体、材料科学。
        """)
        
        st.markdown("### 📝 代码指南 (PyTorch 骨架)")
        st.code("""
# 核心 Loss 函数的构建 (以 1D Heat Equation 为例)
def physics_loss(model, x, t):
    # 启用自动微分追踪
    u = model(x, t)
    
    # 计算 du/dt
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    # 计算 d2u/dx2
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    
    # PDE 残差: F = u_t - alpha * u_xx
    residual = u_t - alpha * u_xx
    
    # 物理损失: 强制 F ≈ 0
    return torch.mean(residual ** 2)

# 总损失 = Loss_BC + Loss_IC + Loss_Physics
# 常用库: DeepXDE, NVIDIA Modulus
        """, language="python")
        st.markdown("---")
        st.markdown("### 🔗 参考文献与工具")
        st.markdown("* **经典论文：** [Physics-informed neural networks: A deep learning framework for solving forward and inverse PDE problems (M. Raissi et al., 2019)](https://arxiv.org/abs/1711.10561)")
        st.markdown("* **开源工具：** [DeepXDE (GitHub)](https://github.com/lululxvi/deepxde)")


    with tab_dgm:
        st.subheader("2. 深度伽辽金方法 (Deep Galerkin Method, DGM)")
        st.info("核心思想：利用深度网络逼近 PDE 的解，将 PDE 转化为等价的积分形式，并使用蒙特卡洛（Monte Carlo）积分计算梯度。")
        st.markdown("""
        * **优势：** 能有效处理**高维 PDE 问题**，避免“维度灾难”。
        * **应用：** 量子化学、金融衍生品定价等高维问题。
        """)
        
        st.markdown("### 📝 代码指南 (概念)")
        st.code("""
# DGM 损失函数基于 L^2 范数在随机采样点上的近似
# Loss = E_x [ (PDE_Operator(NN(x)) - f(x))^2 ]  # 期望通过 Monte Carlo 采样近似
# 步骤: 
# 1. 在求解域内随机采样大量点 (Monte Carlo)。
# 2. 计算每个点上的 PDE 残差。
# 3. 损失函数即为这些残差的均方误差。
        """, language="python")
        st.markdown("---")
        st.markdown("### 🔗 参考文献与工具")
        st.markdown("* **经典论文：** [Deep Galerkin Method for Solving Partial Differential Equations (J. Sirignano and K. Spiliopoulos, 2018)](https://arxiv.org/abs/1708.07469)")
        st.markdown("* **代码示例：** [DGM实现 (GitHub)](https://github.com/alialaradi/DeepGalerkinMethod?utm_source=catalyzex.com)")


    with tab_fno:
        st.subheader("3. 算子学习 (Operator Learning, FNO/DeepONet)")
        st.info("核心思想：目标是学习从**输入函数空间到输出函数空间**的映射（即求解算子），而不是学习特定的解。")
        st.markdown("""
        * **优势：** 一旦训练完成，可以**瞬间预测**新参数或新初始条件下的整个解场（超快求解）。
        * **应用：** 实时仿真、数字孪生、快速设计迭代。
        """)
        
        st.markdown("### 📝 代码指南 (FNO 骨架)")
        st.code("""
# FNO (Fourier Neural Operator) 骨架
class FNO(nn.Module):
    def __init__(self, modes, width):
        # 相比传统NN，FNO在频域（傅里叶变换后）进行参数化操作
        # 主要包含: Lift -> Fourier Layers -> Inverse Fourier -> Project
        # Fourier Layers: 在频域中，通过卷积操作捕获全局信息
        ...

# 训练目标: F(a) ≈ u
# 损失函数: MSE( FNO(a_i), u_i )  其中 a_i是初始函数/源项，u_i是对应解函数
        """, language="python")

        st.markdown("### 📝 代码指南 (DeepONet 骨架)")
        st.code("""
import torch.nn as nn

class DeepONet(nn.Module):
    def __init__(self, input_dim, output_dim, branch_layers, trunk_layers):
        super(DeepONet, self).__init__()
        
        # 1. Branch Net (分支网络): 处理输入函数 a(y) 的测量值 (例如，网格上的 N 个点)
        # 输入维度: N (测量点数量)
        self.branch = self._make_net(input_dim, branch_layers)
        
        # 2. Trunk Net (主干网络): 处理输出的位置坐标 x (例如，(x, t))
        # 输入维度: 坐标维度 (例如 2 for (x, t))
        self.trunk = self._make_net(output_dim, trunk_layers)
        
        # Branch Net 和 Trunk Net 的最终输出维度必须一致 (P)
        self.P = trunk_layers[-1] 

    def forward(self, u_in, x_loc):
        # u_in: 输入函数 a(y) 的测量向量
        # x_loc: 输出位置坐标向量 x
        
        v = self.branch(u_in)  # Shape: (Batch, P)
        w = self.trunk(x_loc)  # Shape: (Batch, P)
        
        # 3. 核心操作: 逐元素相乘并求和 (近似积分)
        # 最终输出 u(x) = sum_{k=1}^{P} v_k * w_k
        return torch.sum(v * w, dim=1, keepdim=True)

# 训练目标: F(a) ≈ u
# 损失函数: MSE( DeepONet(a_i), u_i ) 
        """, language="python")

        st.markdown("---")
        st.markdown("### 🔗 参考文献与工具")
        st.markdown("* **经典论文 (FNO)：** [Fourier Neural Operator for Parametric Partial Differential Equations (Zongyi Li et al., 2020)](https://arxiv.org/abs/2010.08895)")
        st.markdown("* **经典论文 (DeepONet)：** [DeepONet: Learning nonlinear operators for identifying differential equations (Lu et al., 2021)](https://arxiv.org/abs/1910.03193)")

    with tab_surrogate:
        st.subheader("4. 深度学习代理模型 (Surrogate Models)")
        st.info("核心思想：使用大量传统数值模拟结果（数据）训练神经网络，建立**输入参数到输出解**的映射关系。")
        st.markdown("""
        * **优势：** 训练后预测速度极快，用于替代计算量大的传统模拟。
        * **应用：** 工程优化、参数敏感性分析、加速黑箱系统。
        """)
        
        st.markdown("### 📝 代码指南 (数据集构建)")
        st.code("""
# 步骤:
# 1. 生成数据集 (Offline Phase): 
#    - 循环 N 次:
#        - 随机选择输入参数 P_i (如扩散率, 边界值)。
#        - 使用传统求解器 (FDM/FEM) 得到解 U_i。
#        - 数据集 D = { (P_i, U_i) }
# 2. 训练NN (Online Phase):
#    - 训练一个全连接或卷积网络: P_i -> U_i 
#    - Loss: MSE( NN(P_i), U_i )
# 这种方法在 CFD 和高维问题中非常高效。
        """, language="python")
        st.markdown("---")
        st.markdown("### 🔗 参考文献与工具")
        st.markdown("* **综述论文：** [Rapid CFD Prediction Based on Machine Learning Surrogate Model in Built Environment: A Review (MDPI, 2023)](https://www.mdpi.com/2311-5521/10/8/193)")
        st.markdown("* **研究案例：** [Deep learning-based surrogate models outperform simulators and could hasten scientific discoveries (LLNL, 2020)](https://www.llnl.gov/article/46491/deep-learning-based-surrogate-models-outperform-simulators-could-hasten-scientific-discoveries)")

# ==========================================
# 模块 6: 主观问题答疑
# ==========================================
elif choice == "6. 主观问题答疑 (Q&A Corner) 💬":
    st.title("💬 主观问题答疑：AI 助教")
    st.markdown("在这里，您可以提出任何关于偏微分方程理论、数值方法应用或新兴 AI 求解算法的**开放性问题**。")
    
    # ---------------------------------
    # 1. 模型选择和 API 配置界面
    # ---------------------------------
    with st.expander("🔑 LLM 配置与选择", expanded=True):
        
        # --- 模型选择 ---
        model_choice = st.selectbox(
            "选择 AI 模型",
            ["Python 模拟 (离线测试)", "ChatGPT (免费默认 API)", "DeepSeek (需 Key)"],
            key="model_select"
        )

        # 初始化配置变量
        current_api_key = ""
        current_base_url = ""
        current_model_name = ""
        use_llm_api = False
        
        if model_choice == "Python 模拟 (离线测试)":
            st.info("ℹ️ 当前使用内置 Python 函数模拟问答。无需网络连接。\n 仅供如下问题测试：\n 1、什么是有限差分算法？\n 2、什么是PINNs？\n 3、什么是Navier-Stokes方程？")
        
        elif model_choice == "ChatGPT (免费默认 API)":
            # 模式 1: ChatGPT (使用隐藏的默认配置)
            current_api_key = DEFAULT_CHATGPT_KEY
            current_base_url = DEFAULT_CHATGPT_BASE_URL
            current_model_name = DEFAULT_CHATGPT_MODEL
            
            st.success("✅ 正在使用免费通用 API。已自动配置 Key 和 Base URL。")
            use_llm_api = True
            
            # 隐藏输入框，显示配置信息
            st.caption(f"模型: `{current_model_name}`, Base URL: `{current_base_url}`")
        
        elif model_choice == "DeepSeek (需 Key)":
            # 模式 2: DeepSeek (需要用户输入 Key)
            current_base_url = DEFAULT_DEEPSEEK_BASE_URL
            current_model_name = DEFAULT_DEEPSEEK_MODEL

            # --- API Key 输入 ---
            api_key_input = st.text_input(
                "请输入您的 DeepSeek API Key:",
                type="password",
                placeholder="在此输入 Key",
                key="deepseek_api_key_input"
            )
            
            if api_key_input:
                current_api_key = api_key_input
                st.success(f"✅ DeepSeek API 已配置。模型: `{current_model_name}`, Base URL: `{current_base_url}`")
                use_llm_api = True
            else:
                st.warning("⚠️ 请输入 Key 以启用 DeepSeek 模型。")
                
    st.markdown("---")
    
    # --- 聊天记录初始化和显示 ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ---------------------------------
    # 3. 捕获用户输入和响应
    # ---------------------------------
    if prompt := st.chat_input(f"输入你的问题 ({model_choice} 模式)"):
        
        # 1. 记录用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 2. 生成 AI 响应
        with st.spinner(f"AI 助教 正在思考中..."):
            
            if use_llm_api:
                # 调用真实的 OpenAI SDK API
                ai_response = call_llm_api(
                    prompt, 
                    current_api_key, 
                    current_base_url,
                    current_model_name
                )
            else:
                # 离线模拟模式
                ai_response = simulate_ai_response(prompt)
                
        # 3. 记录并显示 AI 消息
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
        with st.chat_message("assistant"):
            st.markdown(ai_response)
