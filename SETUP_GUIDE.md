# 🚀 Complete Setup Guide - Local RAG with Multi-Model Support

## 📋 Prerequisites
- Python 3.11+ (Note: Python 3.14 may have compatibility issues)
- Ollama installed and running
- Bright Data API key (optional but recommended)

---

## 🛠️ Step-by-Step Setup Commands

### 1. **Navigate to Project Directory**
```bash
cd Local-RAG-with-Ollama
```

### 2. **Create and Activate Virtual Environment**
```bash
python -m venv venv
venv\Scripts\Activate
```

### 3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 4. **Verify Ollama Installation**
```bash
ollama --version
ollama list
```

### 5. **Pull Required Ollama Models**
```bash
ollama pull llama3:latest
ollama pull mxbai-embed-large
ollama pull gpt-oss:120b-cloud
ollama pull gemma2:2b
ollama pull qwen3:4b
```

### 6. **Configure Environment Variables**
```bash
# Edit .env file with your settings
# Make sure BRIGHTDATA_API_KEY is set if using Bright Data
# Optional for live resume market signals:
# GOOGLE_API_KEY, GOOGLE_CSE_ID, NAUKRI_API_URL, NAUKRI_API_KEY
```

Example optional keys:
```env
GOOGLE_API_KEY="..."
GOOGLE_CSE_ID="..."
NAUKRI_API_URL="https://<your-naukri-endpoint>"
NAUKRI_API_KEY="..."
NAUKRI_APP_ID="..."
```

---

## 📚 Data Processing Commands

### Option A: **Basic Wikipedia Processing**
```bash
python data_processor.py
# Choose option 1 (Process Wikipedia data)
```

### Option B: **Bright Data Optimization (Recommended)**
```bash
python data_processor.py
# Choose option 2 (Bright Data scraping)
```

### Option C: **Complete Pipeline (Best Results)**
```bash
python data_processor.py
# Choose option 4 (Complete pipeline)
```

---

## 🤖 Launch Chatbot

### **Start Multi-Model RAG Chatbot**
```bash
streamlit run rag_chatbot.py
```

### **Access the Application**
- Open browser to: `http://localhost:8501` (or port shown in terminal)

---

## 🔄 Additional Commands

### **Run Examples (Optional)**
```bash
python examples.py
```

### **Check Available Models**
```bash
ollama list
```

### **Pull Additional Models**
```bash
ollama pull llama2:latest
ollama pull minimax-m2.5:cloud
```

---

## 🛠️ Troubleshooting Commands

### **If Python 3.14 Issues Occur**
```bash
# Install Microsoft C++ Build Tools (Windows)
# Or use Python 3.11/3.12 instead
```

### **If Dependencies Fail**
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### **If Ollama Connection Issues**
```bash
# Restart Ollama service
ollama serve
```

### **If Bright Data Fails**
```bash
# Check API key in .env file
# Verify Bright Data account status
```

---

## 📊 Quick Test Commands

### **Test Data Processing**
```bash
python -c "from data_processor import DataProcessor; dp = DataProcessor(); print('Data processor initialized successfully')"
```

### **Test Chatbot Import**
```bash
python -c "from rag_chatbot import RAGChatbot; print('Chatbot import successful')"
```

### **Test Ollama Connection**
```bash
python -c "from langchain_ollama import ChatOllama; llm = ChatOllama(model='llama3:latest'); print('Ollama connection successful')"
```

---

## 🎯 Recommended Workflow

### **First Time Setup:**
```bash
# 1. Setup environment
python -m venv venv
venv\Scripts\Activate
pip install -r requirements.txt

# 2. Setup models
ollama pull llama3:latest
ollama pull mxbai-embed-large

# 3. Process data
python data_processor.py
# Choose option 4 for complete pipeline

# 4. Launch chatbot
streamlit run rag_chatbot.py
```

### **Regular Usage:**
```bash
# Activate environment
venv\Scripts\Activate

# Launch chatbot
streamlit run rag_chatbot.py
```

### **Data Updates:**
```bash
# Re-process data with new sources
python data_processor.py
# Choose option 3 (Create unified dataset)
```

---

## 🔧 Advanced Configuration

### **Custom Keywords for Bright Data**
Edit `data_processor.py` and modify the `get_optimal_keywords()` method

### **Add New Models**
Add model names to the `available_models` list in `rag_chatbot.py`

### **Adjust Chunk Size**
Modify `chunk_size` and `chunk_overlap` in `data_processor.py`

---

## 📈 Performance Optimization

### **For Better Performance:**
```bash
# Use gemma2:2b for faster responses
# Use gpt-oss:120b-cloud for better quality
# Increase chunk size for more context
```

### **For Memory Efficiency:**
```bash
# Use smaller models (gemma2:2b)
# Reduce chunk overlap
# Limit number of retrieved chunks
```

---

## 🎉 Success Indicators

✅ **Setup Complete When:**
- Virtual environment activated
- All dependencies installed
- Ollama models pulled successfully
- Data processed without errors
- Chatbot launches and responds to queries
- Model switching works in sidebar

✅ **Expected Results:**
- 500+ document chunks (with Bright Data)
- Multi-model switching capability
- Source citations in responses
- Real-time Wikipedia fallback
- Bright Data integration working

---

## 🆘 Common Issues & Solutions

### **Issue: "Module not found"**
```bash
# Solution: Activate virtual environment
venv\Scripts\Activate
pip install -r requirements.txt
```

### **Issue: "Ollama connection failed"**
```bash
# Solution: Restart Ollama
ollama serve
# Wait 30 seconds, then retry
```

### **Issue: "No chunks found"**
```bash
# Solution: Process data first
python data_processor.py
# Choose option 4
```

### **Issue: "Model switching not working"**
```bash
# Solution: Clear Streamlit cache
streamlit cache clear
# Restart chatbot
```

---

## 🎯 End-to-End Test

### **Complete Test Sequence:**
```bash
# 1. Setup
cd Local-RAG-with-Ollama
python -m venv venv
venv\Scripts\Activate
pip install -r requirements.txt

# 2. Models
ollama pull llama3:latest
ollama pull mxbai-embed-large

# 3. Data
python data_processor.py
# Select option 4

# 4. Launch
streamlit run rag_chatbot.py

# 5. Test in browser:
# - Ask: "What is Python programming?"
# - Switch models in sidebar
# - Try different questions
# - Check sources appear
```

---

## 🎊 You're Ready!

Once you complete these steps, you'll have:
- ✅ Multi-model RAG chatbot
- ✅ Bright Data integration
- ✅ Wikipedia + real-time search
- ✅ Source citations
- ✅ Dynamic model switching

**🚀 Your advanced Local RAG system is ready to use!**
