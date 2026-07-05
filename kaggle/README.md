# Kaggle GPU 加速

## 使用步骤

1. 打开 https://www.kaggle.com/ 并登录
2. **Create New Notebook**
3. 上传 `run.ipynb`：File → Import Notebook → 选择 `kaggle/run.ipynb`
4. **添加数据**：右侧 +Add Data → Upload → 选择你的 `data01/` 文件夹（含 `knowledge_*.txt` + `queries_*.txt`）
5. **开启 GPU**：Notebook Settings → Accelerator → **GPU T4 x2**
6. 按顺序运行各 Cell

## 文件说明

| 文件 | 用途 |
|------|------|
| `run.ipynb` | Kaggle Notebook，直接导入使用 |
| `install.sh` | 安装脚本（Notebook 已内置，备用） |
