"""
Complete RAG Chatbot with Multi-Source Support
Wikipedia + Bright Data + Real-time Search
"""

import os
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
from dotenv import load_dotenv
import streamlit as st
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
import wikipedia
from pypdf import PdfReader
from pydantic import BaseModel, Field, ValidationError

load_dotenv()


def inject_custom_css():
    """Improve app look-and-feel with lightweight custom styling"""
    st.markdown(
        """
        <style>
            .main .block-container {
                padding-top: 1.4rem;
                padding-bottom: 1.2rem;
                max-width: 1180px;
            }
            .hero-card {
                border: 1px solid rgba(120,120,120,0.20);
                border-radius: 14px;
                padding: 14px 18px;
                background: linear-gradient(135deg, rgba(99,102,241,0.12), rgba(16,185,129,0.10));
                margin-bottom: 12px;
            }
            .mini-card {
                border: 1px solid rgba(120,120,120,0.18);
                border-radius: 12px;
                padding: 10px 12px;
                background: rgba(255,255,255,0.02);
            }
            div[data-testid="stMetric"] {
                border: 1px solid rgba(120,120,120,0.22);
                border-radius: 12px;
                padding: 8px;
                background: rgba(255,255,255,0.02);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


class ATSCompatibilityModel(BaseModel):
    status: str = "Pass"
    issues: List[str] = Field(default_factory=list)


class FeedbackModel(BaseModel):
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    grammar_spelling: List[str] = Field(default_factory=list)
    ats_compatibility: ATSCompatibilityModel = Field(default_factory=ATSCompatibilityModel)


class RewriteBulletsModel(BaseModel):
    original: str = ""
    rewritten: str = ""


class SuggestionsModel(BaseModel):
    skills_to_add: List[str] = Field(default_factory=list)
    keywords_to_include: List[str] = Field(default_factory=list)
    rewrite_bullets: RewriteBulletsModel = Field(default_factory=RewriteBulletsModel)
    professional_summary: str = ""


class ResourceModel(BaseModel):
    name: str
    link: str


class ResumeAnalysisOutputModel(BaseModel):
    resume_score: int = Field(ge=0, le=100)
    summary: str
    feedback: FeedbackModel
    suggestions: SuggestionsModel
    resources: List[ResourceModel] = Field(default_factory=list)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    evidence: List[str] = Field(default_factory=list)


class ResumeRevisionOutputModel(BaseModel):
    revised_resume: str
    improvement_plan: List[str] = Field(default_factory=list, min_length=6)
    high_impact_changes: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    evidence: List[str] = Field(default_factory=list)

class RAGChatbot:
    """Complete RAG Chatbot with all features"""
    
    def __init__(self, chat_model=None, fast_mode=True):
        self.embeddings = OllamaEmbeddings(model=os.getenv("EMBEDDING_MODEL"))
        self.chat_model = chat_model or os.getenv("CHAT_MODEL")
        self.fast_mode = fast_mode
        self.llm = self._create_llm(self.chat_model)
        self.data = self.load_data()
        self.embedding_matrix = None
        self.embedding_norms = None
        self.valid_chunks = []
        self._prepare_embeddings_index()

    def _create_llm(self, model_name):
        """Create tuned LLM instance for faster responses"""
        return ChatOllama(
            model=model_name,
            temperature=0,
            num_ctx=2048,
            num_predict=420,
        )

    def _get_backup_model_name(self):
        """Resolve backup model for fallback strategy"""
        backup = os.getenv("BACKUP_CHAT_MODEL", "gemma2:2b")
        if backup == self.chat_model:
            return "llama2:latest"
        return backup

    def _log_evaluation_event(self, event_type: str, payload: Dict[str, Any]):
        """Write lightweight JSONL telemetry for evaluation and quality monitoring"""
        try:
            log_path = Path(os.getenv("EVAL_LOG_FILE", "logs/evaluation_runs.jsonl"))
            log_path.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": event_type,
                **payload,
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _validate_analysis_payload(self, payload: Dict[str, Any]):
        """Validate analysis payload against strict schema"""
        model = ResumeAnalysisOutputModel.model_validate(payload)
        return model.model_dump()

    def _is_low_quality_analysis(self, payload: Dict[str, Any]) -> bool:
        """Detect placeholder-like analysis that is formally valid but not useful"""
        if not payload:
            return True

        summary = (payload.get("summary") or "").strip()
        feedback = payload.get("feedback", {}) or {}
        suggestions = payload.get("suggestions", {}) or {}

        strengths = [s.strip() for s in feedback.get("strengths", []) if isinstance(s, str)]
        weaknesses = [s.strip() for s in feedback.get("weaknesses", []) if isinstance(s, str)]
        grammar = [s.strip() for s in feedback.get("grammar_spelling", []) if isinstance(s, str)]
        skills = [s.strip() for s in suggestions.get("skills_to_add", []) if isinstance(s, str)]
        keywords = [s.strip() for s in suggestions.get("keywords_to_include", []) if isinstance(s, str)]
        rewritten = (suggestions.get("rewrite_bullets", {}) or {}).get("rewritten", "").strip()
        evidence = [e.strip() for e in payload.get("evidence", []) if isinstance(e, str)]

        usable_strengths = [s for s in strengths if s and s != ""]
        usable_weaknesses = [s for s in weaknesses if s and s != ""]
        usable_grammar = [s for s in grammar if s and s != ""]
        usable_skills = [s for s in skills if s and s != ""]
        usable_keywords = [k for k in keywords if k and k != ""]
        usable_evidence = [e for e in evidence if e and e != ""]

        score = int(payload.get("resume_score", 0) or 0)

        if score <= 5:
            return True
        if len(summary) < 20:
            return True
        if not usable_strengths and not usable_weaknesses and not usable_grammar:
            return True
        if len(usable_keywords) < 3:
            return True
        if len(usable_skills) < 2:
            return True
        if len(rewritten) < 20:
            return True
        if len(usable_evidence) < 2:
            return True

        return False

    def _validate_revision_payload(self, payload: Dict[str, Any]):
        """Validate revision payload against strict schema"""
        model = ResumeRevisionOutputModel.model_validate(payload)
        return model.model_dump()

    def _invoke_json_task(
        self,
        task_name: str,
        prompt: str,
        schema_hint: str,
        validator,
        max_attempts: int = 2,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Invoke model with retry/repair and model fallback; returns validated JSON + metadata"""
        model_sequence = [self.chat_model, self._get_backup_model_name()]
        repair_prompt = prompt
        last_error = ""

        for attempt in range(1, max_attempts + 1):
            for idx, model_name in enumerate(model_sequence):
                model_client = self.llm if model_name == self.chat_model else self._create_llm(model_name)
                try:
                    response = model_client.invoke(repair_prompt)
                    raw_output = getattr(response, "content", "")
                    parsed = self._safe_json_parse(raw_output)
                    if not parsed:
                        raise ValueError("Model output is not valid JSON")

                    validated = validator(parsed)
                    return validated, {
                        "task": task_name,
                        "model_used": model_name,
                        "attempt": attempt,
                        "used_fallback_model": idx > 0,
                        "used_repair_prompt": attempt > 1,
                        "validation_error": "",
                    }
                except (ValidationError, ValueError, TypeError, KeyError) as err:
                    last_error = str(err)
                except Exception as err:
                    last_error = f"Model invocation error: {err}"

            repair_prompt = f"""Your previous output failed validation.
Return ONLY valid JSON matching this schema and constraints:
{schema_hint}

Validation error:
{last_error}

Re-generate strictly valid JSON now."""

        return {}, {
            "task": task_name,
            "model_used": "none",
            "attempt": max_attempts,
            "used_fallback_model": False,
            "used_repair_prompt": True,
            "validation_error": last_error,
        }

    def _prepare_embeddings_index(self):
        """Precompute embedding matrix for fast vectorized similarity search"""
        chunks = self.data.get("chunks", []) if self.data else []
        valid_chunks = []
        vectors = []

        for chunk in chunks:
            embedding = chunk.get("embedding")
            if isinstance(embedding, list) and embedding:
                valid_chunks.append(chunk)
                vectors.append(embedding)

        if not vectors:
            self.embedding_matrix = None
            self.embedding_norms = None
            self.valid_chunks = []
            return

        matrix = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1)
        norms[norms == 0] = 1e-9

        self.embedding_matrix = matrix
        self.embedding_norms = norms
        self.valid_chunks = valid_chunks
    
    def switch_model(self, new_model):
        """Switch to a different chat model"""
        self.chat_model = new_model
        self.llm = self._create_llm(new_model)
        print(f"🔄 Switched to model: {new_model}")
    
    def load_data(self):
        """Load best available data source"""
        
        # Try unified data first
        unified_file = os.path.join(os.getenv("DATASET_STORAGE_FOLDER"), "unified_chunks.json")
        if os.path.exists(unified_file):
            with open(unified_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # Try Bright Data
        bd_file = os.path.join(os.getenv("DATASET_STORAGE_FOLDER"), "brightdata_chunks.json")
        if os.path.exists(bd_file):
            with open(bd_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # Fallback to Wikipedia
        wiki_file = os.path.join(os.getenv("DATASET_STORAGE_FOLDER"), "processed_chunks.json")
        if os.path.exists(wiki_file):
            with open(wiki_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        return {"chunks": []}
    
    def similarity_search(self, query, chunks_data, k=5):
        """Enhanced similarity search"""
        if self.embedding_matrix is None or not self.valid_chunks:
            return []
        
        query_embedding = np.array(self.embeddings.embed_query(query), dtype=np.float32)
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return []

        scores = np.dot(self.embedding_matrix, query_embedding) / (self.embedding_norms * query_norm)
        if scores.size == 0:
            return []

        top_k = min(k, scores.shape[0])
        candidate_indices = np.argpartition(scores, -top_k)[-top_k:]
        ordered_indices = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]

        return [(float(scores[idx]), self.valid_chunks[idx]) for idx in ordered_indices]
    
    def realtime_wikipedia_search(self, query, max_results=2):
        """Real-time Wikipedia search"""
        try:
            search_results = wikipedia.search(query, results=max_results)
            articles = []
            
            for title in search_results:
                try:
                    page = wikipedia.page(title)
                    articles.append({
                        'title': page.title,
                        'url': page.url,
                        'content': page.content[:2000]
                    })
                except:
                    continue
                    
            return articles
        except:
            return []
    
    def search_and_retrieve(self, query):
        """Complete search and retrieval"""
        context = ""
        sources = []
        search_method = "local_search"
        
        # Search local data
        local_results = self.similarity_search(query, self.data, k=2)
        
        # Use local results if good enough
        if local_results and local_results[0][0] > 0.3:
            for similarity, chunk in local_results:
                context += f"Content: {chunk['content']}\n\n"
                sources.append(chunk['metadata'].get('source', 'Unknown'))
            search_method = "local_chunks"
        elif not self.fast_mode:
            # Fallback to real-time Wikipedia
            wiki_articles = self.realtime_wikipedia_search(query)
            
            if wiki_articles:
                for article in wiki_articles:
                    context += f"Source: {article['title']}\nContent: {article['content']}\n\n"
                    sources.append(article['url'])
                search_method = "realtime_wikipedia"
        else:
            search_method = "fast_local_only"
        
        return context, sources, search_method
    
    def generate_response(self, query, context):
        """Generate response"""
        if context:
            prompt = f"""You are a helpful assistant. Use context and respond clearly in concise form.

Context:
{context}

Question: {query}

Give a direct answer. If context is insufficient, say what is missing in one short line.

Answer:"""
        else:
            prompt = f"""User question: {query}

No specific local context found. Provide a concise helpful response.

Response:"""
        
        response = self.llm.invoke(prompt)
        return response.content

    def _safe_json_parse(self, raw_text):
        """Parse JSON from model output safely"""
        if not raw_text:
            return None

        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.replace("```json", "").replace("```", "").strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            start_idx = cleaned.find("{")
            end_idx = cleaned.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                try:
                    return json.loads(cleaned[start_idx:end_idx + 1])
                except json.JSONDecodeError:
                    return None
        return None

    def extract_text_from_pdf(self, uploaded_file):
        """Extract text content from an uploaded PDF file"""
        if not uploaded_file:
            return ""

        try:
            reader = PdfReader(uploaded_file)
            pages_text = []
            for page in reader.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages_text.append(page_text)
            return "\n\n".join(pages_text).strip()
        except Exception:
            return ""

    def _extract_ats_checks(self, resume_text):
        """Heuristic ATS checks from extracted resume text"""
        text = resume_text or ""
        lowered = text.lower()
        issues = []

        has_email = bool(re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text))
        has_phone = bool(re.search(r"(\+?\d[\d\s\-()]{7,}\d)", text))

        if not has_email:
            issues.append("Missing email in contact information.")
        if not has_phone:
            issues.append("Missing phone number in contact information.")

        standard_headings = [
            "summary",
            "professional summary",
            "work experience",
            "experience",
            "education",
            "skills",
            "certifications",
            "projects",
        ]
        if not any(h in lowered for h in standard_headings):
            issues.append("Resume sections are unclear or missing standard headings.")

        likely_too_long = len(text.split()) > 1400
        if likely_too_long:
            issues.append("Resume appears long; keep it within 1-2 pages for ATS friendliness.")

        status = "Pass" if not issues else "Needs Improvement"
        return {"status": status, "issues": issues}

    def _industry_skill_baseline(self, industry, target_role):
        """Lightweight skill baseline used when JD is missing"""
        industry_map = {
            "IT": ["Python", "SQL", "Git", "Agile", "REST API", "Cloud", "Docker", "CI/CD"],
            "Healthcare": ["EHR", "HIPAA", "Patient Care", "Clinical Documentation", "Communication"],
            "Finance": ["Excel", "Financial Modeling", "Risk Analysis", "Compliance", "Data Analysis"],
            "Marketing": ["SEO", "SEM", "Google Analytics", "Content Strategy", "A/B Testing"],
            "Sales": ["CRM", "Lead Generation", "Negotiation", "Pipeline Management", "Forecasting"],
        }
        role_map = {
            "software engineer": ["System Design", "Testing", "Microservices", "Kubernetes", "TypeScript"],
            "data scientist": ["Machine Learning", "Pandas", "Statistics", "Model Deployment", "Feature Engineering"],
            "marketing manager": ["Campaign Management", "Brand Strategy", "Performance Marketing", "Attribution"],
            "product manager": ["Roadmapping", "Stakeholder Management", "Metrics", "Experimentation"],
        }

        base = industry_map.get(industry, ["Communication", "Problem Solving", "Leadership", "Teamwork"])
        role_extra = role_map.get((target_role or "").strip().lower(), [])
        return list(dict.fromkeys(base + role_extra))

    def _build_resume_rag_context(self, resume_text, target_role, industry, experience_level, job_description=None, company_name=None):
        """Build RAG context for resume analysis"""
        query = (
            f"Resume analysis best practices for {target_role} in {industry} at {experience_level} level. "
            "Include ATS, keywords, measurable impact, and required skills."
        )

        context_parts = []
        sources = []

        local_results = self.similarity_search(query, self.data, k=4)
        for similarity, chunk in local_results:
            if similarity > 0.2:
                context_parts.append(f"[Local Guide] {chunk.get('content', '')}")
                sources.append(chunk.get("metadata", {}).get("source", "local_dataset"))

        if job_description:
            context_parts.append(f"[Job Description]\n{job_description}")
            sources.append("job_description")

        if company_name:
            company_articles = self.realtime_wikipedia_search(f"{company_name} company career jobs", max_results=1)
            for article in company_articles:
                context_parts.append(f"[Company Context: {article['title']}]\n{article['content'][:1200]}")
                sources.append(article["url"])

        baseline_skills = self._industry_skill_baseline(industry, target_role)
        context_parts.append("[Industry Skill Baseline] " + ", ".join(baseline_skills))
        sources.append("industry_skill_baseline")

        return "\n\n".join(context_parts), list(dict.fromkeys(sources)), baseline_skills

    def analyze_resume(self, resume_text, target_role, industry, experience_level, job_description=None, company_name=None):
        """Analyze resume and return strict JSON payload"""
        start_time = time.perf_counter()
        ats_precheck = self._extract_ats_checks(resume_text)
        rag_context, sources, baseline_skills = self._build_resume_rag_context(
            resume_text=resume_text,
            target_role=target_role,
            industry=industry,
            experience_level=experience_level,
            job_description=job_description,
            company_name=company_name,
        )

        prompt = f"""You are an expert Resume Analyst AI.

Your task: analyze the resume using the provided context and return ONLY valid JSON.
Do not include markdown or code fences.

Inputs:
- Target Role: {target_role}
- Industry: {industry}
- Experience Level: {experience_level}
- Company Name: {company_name or 'N/A'}

ATS Precheck:
{json.dumps(ats_precheck, ensure_ascii=False)}

Retrieved RAG Context:
{rag_context if rag_context else 'No external context available.'}

Resume Text:
{resume_text}

Output JSON schema:
{{
  "resume_score": 0,
  "summary": "",
  "feedback": {{
    "strengths": [""],
    "weaknesses": [""],
    "grammar_spelling": [""],
    "ats_compatibility": {{
      "status": "Pass",
      "issues": [""]
    }}
  }},
  "suggestions": {{
    "skills_to_add": [""],
    "keywords_to_include": [""],
    "rewrite_bullets": {{
      "original": "",
      "rewritten": ""
    }},
    "professional_summary": ""
  }},
  "resources": [
    {{"name": "", "link": ""}}
    ],
    "confidence": 0.0,
    "evidence": [""]
}}

Requirements:
1) Score must be integer 0-100 computed from ATS compatibility (30%), content quality (40%), skill relevance (20%), formatting (10%).
2) Use concise, actionable feedback.
3) Include 5-10 missing keywords in suggestions.keywords_to_include.
4) Include upskilling resources links.
5) Respect ATS precheck findings inside feedback.ats_compatibility.
6) confidence must be a number between 0 and 1.
7) evidence must include 3-8 concise evidence points tied to resume/JD/context.
"""

        analysis_schema_hint = """{
    "resume_score": 0,
    "summary": "",
    "feedback": {
        "strengths": [""],
        "weaknesses": [""],
        "grammar_spelling": [""],
        "ats_compatibility": {"status": "Pass", "issues": [""]}
    },
    "suggestions": {
        "skills_to_add": [""],
        "keywords_to_include": [""],
        "rewrite_bullets": {"original": "", "rewritten": ""},
        "professional_summary": ""
    },
    "resources": [{"name": "", "link": "https://..."}],
    "confidence": 0.0,
    "evidence": [""]
}"""

        parsed, meta = self._invoke_json_task(
            task_name="resume_analysis",
            prompt=prompt,
            schema_hint=analysis_schema_hint,
            validator=self._validate_analysis_payload,
            max_attempts=2,
        )

        if parsed and self._is_low_quality_analysis(parsed):
            quality_retry_prompt = f"""Your previous output was too generic or empty.
Re-run the resume analysis and return ONLY valid JSON with meaningful non-empty content.

Hard quality constraints:
- resume_score must be between 35 and 95 (realistic)
- summary must be at least 20 words
- feedback.strengths must contain at least 2 non-empty items
- feedback.weaknesses must contain at least 2 non-empty items
- suggestions.skills_to_add must contain at least 3 items
- suggestions.keywords_to_include must contain at least 5 items
- suggestions.rewrite_bullets.rewritten must be at least 12 words
- evidence must contain at least 3 non-empty items

Use this same context and resume:
Target Role: {target_role}
Industry: {industry}
Experience Level: {experience_level}
ATS Precheck: {json.dumps(ats_precheck, ensure_ascii=False)}
RAG Context: {rag_context if rag_context else 'No external context available.'}
Resume Text: {resume_text}

Return JSON only."""

            try:
                retry_model = self._create_llm(self._get_backup_model_name())
                retry_response = retry_model.invoke(quality_retry_prompt)
                retry_parsed = self._safe_json_parse(getattr(retry_response, "content", ""))
                if retry_parsed:
                    retry_validated = self._validate_analysis_payload(retry_parsed)
                    if not self._is_low_quality_analysis(retry_validated):
                        parsed = retry_validated
                        meta = {
                            "task": "resume_analysis",
                            "model_used": self._get_backup_model_name(),
                            "attempt": max(meta.get("attempt", 1), 2),
                            "used_fallback_model": True,
                            "used_repair_prompt": True,
                            "validation_error": "",
                        }
            except Exception:
                pass

        if not parsed or self._is_low_quality_analysis(parsed):
            parsed = {
                "resume_score": 65,
                "summary": f"Resume needs measurable impact and stronger role alignment for {target_role}.",
                "feedback": {
                    "strengths": [
                        "Resume includes baseline structure that can be improved for ATS and relevance.",
                        "Role and industry context are available for tailoring."
                    ],
                    "weaknesses": [
                        "Experience bullets need stronger action + impact phrasing.",
                        "Keywords are not sufficiently aligned with target role and job context.",
                        "Quantifiable outcomes are missing or under-specified."
                    ],
                    "grammar_spelling": [
                        "Review for tense consistency and remove vague phrases like 'helped with'."
                    ],
                    "ats_compatibility": ats_precheck,
                },
                "suggestions": {
                    "skills_to_add": baseline_skills[:5],
                    "keywords_to_include": baseline_skills[:8],
                    "rewrite_bullets": {
                        "original": "Helped with projects",
                        "rewritten": "Led cross-functional delivery of key projects, improving delivery predictability and reducing turnaround time through process optimization.",
                    },
                    "professional_summary": f"Results-driven {target_role} candidate with experience in {industry}, focused on measurable outcomes and continuous improvement.",
                },
                "resources": [
                    {"name": "Coursera Career Academy", "link": "https://www.coursera.org/career-academy"},
                    {"name": "LinkedIn Learning", "link": "https://www.linkedin.com/learning/"},
                ],
                "confidence": 0.72,
                "evidence": [
                    "ATS precheck completed on uploaded resume text.",
                    "Industry baseline skills used for gap estimation.",
                    "Quality gate replaced low-information model output with actionable guidance.",
                ],
            }
            meta = {
                "task": "resume_analysis",
                "model_used": "quality_guard_fallback",
                "attempt": 2,
                "used_fallback_model": False,
                "used_repair_prompt": True,
                "validation_error": "Low-quality content detected and replaced",
            }

        parsed.setdefault("feedback", {})
        parsed["feedback"]["ats_compatibility"] = ats_precheck
        parsed["summary"] = (parsed.get("summary") or "") + (
            f" (RAG sources used: {len(sources)})" if sources else ""
        )
        parsed["_meta"] = meta

        latency_sec = round(time.perf_counter() - start_time, 3)
        self._log_evaluation_event(
            "resume_analysis",
            {
                "model_requested": self.chat_model,
                "model_used": meta.get("model_used"),
                "attempt": meta.get("attempt", 0),
                "used_fallback_model": meta.get("used_fallback_model", False),
                "latency_sec": latency_sec,
                "resume_score": parsed.get("resume_score", 0),
                "confidence": parsed.get("confidence", 0.0),
                "ats_issues": len(ats_precheck.get("issues", [])),
                "keyword_count": len(parsed.get("suggestions", {}).get("keywords_to_include", [])),
            },
        )

        return parsed

    def generate_resume_revision(
        self,
        resume_text,
        target_role,
        industry,
        experience_level,
        analysis_result,
        job_description=None,
        company_name=None,
    ):
        """Generate revised resume draft + concrete improvement roadmap"""
        start_time = time.perf_counter()
        prompt = f"""You are an expert resume writer and career coach.

Rewrite the resume to be stronger, ATS-friendly, and aligned to the target role.
Return ONLY valid JSON. No markdown fences.

Inputs:
- Target Role: {target_role}
- Industry: {industry}
- Experience Level: {experience_level}
- Company Name: {company_name or 'N/A'}

Prior Analysis JSON:
{json.dumps(analysis_result, ensure_ascii=False)}

Optional Job Description:
{job_description or 'N/A'}

Original Resume Text:
{resume_text}

Output JSON schema:
{{
  "revised_resume": "Full revised resume text with clear sections: Professional Summary, Skills, Work Experience, Education, Certifications/Projects if relevant.",
  "improvement_plan": [
    "Actionable improvement item 1",
    "Actionable improvement item 2"
  ],
  "high_impact_changes": [
    "What changed and why"
    ],
    "confidence": 0.0,
    "evidence": [""]
}}

Rules:
1) Keep claims realistic; do not invent fake employers or degrees.
2) Convert weak bullet points into action + impact style with metrics where possible.
3) Keep language concise and ATS-friendly.
4) improvement_plan must include at least 6 actionable steps.
5) confidence must be a number between 0 and 1.
6) evidence must include 3-8 concise points linked to changes made.
"""

        revision_schema_hint = """{
  "revised_resume": "",
  "improvement_plan": ["", "", "", "", "", ""],
  "high_impact_changes": [""],
  "confidence": 0.0,
  "evidence": [""]
}"""

        parsed, meta = self._invoke_json_task(
            task_name="resume_revision",
            prompt=prompt,
            schema_hint=revision_schema_hint,
            validator=self._validate_revision_payload,
            max_attempts=2,
        )

        if not parsed:
            parsed = {
                "revised_resume": (
                    f"Professional Summary\n"
                    f"Results-driven {target_role} professional in {industry} with {experience_level.lower()} experience, "
                    "focused on measurable business impact, collaboration, and continuous improvement.\n\n"
                    "Skills\n"
                    "- Communication\n- Problem Solving\n- Team Collaboration\n\n"
                    "Work Experience\n"
                    "- Reframe each bullet using action + scope + measurable outcome.\n\n"
                    "Education\n"
                    "- Include degree, institution, year, and relevant coursework/certifications."
                ),
                "improvement_plan": [
                    "Rewrite summary to match target role keywords.",
                    "Add 5-10 missing role-specific keywords from JD.",
                    "Convert each experience bullet to action + metric format.",
                    "Prioritize most relevant projects for the target role.",
                    "Add certifications or training aligned with missing skills.",
                    "Standardize formatting and section headings for ATS.",
                ],
                "high_impact_changes": [
                    "Improved clarity and ATS alignment across major sections.",
                    "Added stronger impact-oriented bullet style.",
                ],
                "confidence": 0.45,
                "evidence": [
                    "Revision based on prior analysis weaknesses.",
                    "Bullet style standardized to action-impact format.",
                    "Fallback template used after validation failure.",
                ],
            }
            meta = {
                "task": "resume_revision",
                "model_used": "fallback_template",
                "attempt": 2,
                "used_fallback_model": False,
                "used_repair_prompt": True,
                "validation_error": "Generated fallback payload",
            }

        parsed["_meta"] = meta
        self._log_evaluation_event(
            "resume_revision",
            {
                "model_requested": self.chat_model,
                "model_used": meta.get("model_used"),
                "attempt": meta.get("attempt", 0),
                "used_fallback_model": meta.get("used_fallback_model", False),
                "latency_sec": round(time.perf_counter() - start_time, 3),
                "confidence": parsed.get("confidence", 0.0),
                "improvement_items": len(parsed.get("improvement_plan", [])),
            },
        )

        return parsed

    def build_improvement_plan_from_analysis(self, analysis_result):
        """Create guaranteed improvement guidance from analysis JSON"""
        analysis_result = analysis_result or {}
        feedback = analysis_result.get("feedback", {})
        suggestions = analysis_result.get("suggestions", {})

        plan = []

        for weakness in feedback.get("weaknesses", [])[:4]:
            if weakness:
                plan.append(f"Address weakness: {weakness}")

        missing_skills = suggestions.get("skills_to_add", [])[:4]
        if missing_skills:
            plan.append("Add or strengthen these skills: " + ", ".join(missing_skills))

        missing_keywords = suggestions.get("keywords_to_include", [])[:6]
        if missing_keywords:
            plan.append("Include ATS keywords naturally: " + ", ".join(missing_keywords))

        rewrite_tip = suggestions.get("rewrite_bullets", {}).get("rewritten")
        if rewrite_tip:
            plan.append("Use impact-style bullets like: " + rewrite_tip)

        grammar_items = feedback.get("grammar_spelling", [])
        if grammar_items:
            plan.append("Fix grammar/spelling issues highlighted in feedback.")

        while len(plan) < 6:
            defaults = [
                "Tailor the professional summary to the target role and industry.",
                "Quantify achievements with metrics, scope, and outcomes.",
                "Keep section headings ATS-friendly and consistent.",
                "Prioritize recent, role-relevant experience and projects.",
                "Add certifications or courses to close high-priority skill gaps.",
                "Keep formatting simple, readable, and consistent across sections.",
            ]
            plan.append(defaults[len(plan) % len(defaults)])

        return plan[:8]

    def build_professional_report(self, analysis_result, revision_result, request_payload):
        """Create a business-ready assessment report"""
        analysis = analysis_result or {}
        revision = revision_result or {}
        payload = request_payload or {}

        role = payload.get("target_role", "N/A")
        industry = payload.get("industry", "N/A")
        level = payload.get("experience_level", "N/A")
        company = payload.get("company_name") or "N/A"

        score = analysis.get("resume_score", 0)
        confidence = analysis.get("confidence", 0.0)
        summary = analysis.get("summary", "")

        feedback = analysis.get("feedback", {})
        strengths = feedback.get("strengths", []) or []
        weaknesses = feedback.get("weaknesses", []) or []
        grammar = feedback.get("grammar_spelling", []) or []
        ats = feedback.get("ats_compatibility", {}) or {}

        suggestions = analysis.get("suggestions", {})
        skills_to_add = suggestions.get("skills_to_add", []) or []
        keywords = suggestions.get("keywords_to_include", []) or []
        professional_summary = suggestions.get("professional_summary", "")

        improvement_plan = revision.get("improvement_plan", []) or self.build_improvement_plan_from_analysis(analysis)
        high_impact_changes = revision.get("high_impact_changes", []) or []

        resources = analysis.get("resources", []) or []
        evidence = analysis.get("evidence", []) or []

        def bulletize(items, default="No items available."):
            cleaned = [f"- {item}" for item in items if isinstance(item, str) and item.strip()]
            return "\n".join(cleaned) if cleaned else f"- {default}"

        report = f"""# Resume Assessment Report

## Candidate Target Profile
- Target Role: {role}
- Industry: {industry}
- Experience Level: {level}
- Company Context: {company}

## Executive Summary
- Overall Score: {score}/100
- Confidence: {confidence:.2f}
- Assessment Summary: {summary or 'Resume assessed with role-specific RAG context.'}

## ATS Compliance
- Status: {ats.get('status', 'Needs Improvement')}
{bulletize(ats.get('issues', []), default='No ATS issues detected.')}

## Key Strengths
{bulletize(strengths, default='Strengths not explicitly identified.')}

## Improvement Priorities
{bulletize(weaknesses, default='No high-priority weaknesses detected.')}

## Grammar & Clarity Notes
{bulletize(grammar, default='No explicit grammar issues identified.')}

## Skill Gap & Keyword Optimization
### Skills to Add
{bulletize(skills_to_add, default='No additional skills suggested.')}

### Priority Keywords
{bulletize(keywords, default='No additional keywords suggested.')}

## Recommended Professional Summary
{professional_summary or 'Professional summary recommendation not available.'}

## Action Plan
{bulletize(improvement_plan, default='No action plan generated.')}

## High-Impact Revisions
{bulletize(high_impact_changes, default='No revision highlights available.')}

## Evidence Basis
{bulletize(evidence, default='Evidence not available.')}

## Upskilling Resources
{bulletize([f"{r.get('name', 'Resource')}: {r.get('link', '')}" for r in resources], default='No resources available.')}
"""

        return report

def main():
    """Main Streamlit application"""
    st.set_page_config(page_title="Multi-Model RAG Chatbot", page_icon="🤖", layout="wide")
    inject_custom_css()
    
    # Model selection in sidebar
    with st.sidebar:
        st.title("🤖 Model Selection")

        app_mode = st.radio(
            "Mode",
            ["General Chat", "Resume Analyst"],
            index=0,
            help="Switch between normal RAG chat and structured resume analysis"
        )

        fast_mode = st.toggle(
            "⚡ Fast Mode",
            value=True,
            help="Faster responses by using local retrieval first and skipping live web fallback"
        )
        
        # Available models
        available_models = [
            "llama3:latest",
            "llama2:latest", 
            "gemma2:2b",
            "gpt-oss:120b-cloud",
            "qwen3:4b",
            "minimax-m2.5:cloud"
        ]
        
        # Model selector
        selected_model = st.selectbox(
            "Choose Chat Model:",
            available_models,
            index=0,
            help="Select different AI models for varied responses"
        )
        
        # Initialize chatbot with selected model
        if (
            'chatbot' not in st.session_state
            or st.session_state.get('current_model') != selected_model
            or st.session_state.get('current_fast_mode') != fast_mode
        ):
            st.session_state.chatbot = RAGChatbot(chat_model=selected_model, fast_mode=fast_mode)
            st.session_state.current_model = selected_model
            st.session_state.current_fast_mode = fast_mode
            st.success(f"🔄 Switched to: {selected_model}")
        
        chatbot = st.session_state.chatbot
        
        st.markdown("---")
        
        # Model info
        model_info = {
            "llama3:latest": "8B parameters, balanced performance",
            "llama2:latest": "7B parameters, stable version", 
            "gemma2:2b": "2B parameters, fast responses",
            "gpt-oss:120b-cloud": "120B parameters, most capable",
            "qwen3:4b": "4B parameters, multilingual",
            "minimax-m2.5:cloud": "Cloud-based, optimized"
        }
        
        st.info(f"""
**Current Model:** {selected_model}
{model_info.get(selected_model, '')}
        """)
        
        st.markdown("---")
    
    # Title
    if app_mode == "Resume Analyst":
        st.markdown(
            """
            <div class="hero-card">
                <h2 style="margin:0;">📄 Resume Analyst AI</h2>
                <p style="margin:6px 0 0 0;">RAG-assisted ATS, skill-gap, keyword optimization, and resume revision.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="hero-card">
                <h2 style="margin:0;">🤖 RAG Chatbot</h2>
                <p style="margin:6px 0 0 0;">Multi-source assistant powered by local chunks and optional live retrieval.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    # Sidebar (continuing from model selection)
    with st.sidebar:
        st.title("🔍 System Info")
        
        # Data statistics
        total_chunks = len(chatbot.data.get("chunks", []))
        st.info(f"""
**📊 Data Status:**
- Total chunks: {total_chunks}
- Current model: {chatbot.chat_model}
- Mode: {'Fast' if chatbot.fast_mode else 'Balanced'}
- Embeddings: {os.getenv('EMBEDDING_MODEL')}
""")
        
        # Source breakdown
        if "source_stats" in chatbot.data:
            st.write("**📚 Sources:**")
            for source, count in chatbot.data["source_stats"].items():
                percentage = (count / total_chunks) * 100
                st.write(f"- {source}: {count} ({percentage:.1f}%)")
        
        st.markdown("---")
        
        # Model comparison feature
        st.markdown("### 🔄 Model Comparison")
        
        if st.button("🆚 Compare Models"):
            with st.spinner("Testing models..."):
                test_query = "What is Python programming?"
                comparison_results = {}
                
                for model in ["llama3:latest", "gpt-oss:120b-cloud", "gemma2:2b"]:
                    try:
                        temp_chatbot = RAGChatbot(chat_model=model)
                        response = temp_chatbot.generate_response(test_query, "")
                        comparison_results[model] = response[:100] + "..."
                    except Exception as e:
                        comparison_results[model] = f"Error: {str(e)}"
                
                st.write("**Model Comparison Results:**")
                for model, response in comparison_results.items():
                    st.write(f"**{model}:** {response}")
        
        st.markdown("---")
        
        # Data management
        st.markdown("### 📚 Data Management")
        
        if st.button("🔄 Refresh Data"):
            chatbot.data = chatbot.load_data()
            st.rerun()
        
        if st.button("🌐 Bright Data Optimize"):
            with st.spinner("Running Bright Data optimization..."):
                try:
                    from data_processor import DataProcessor
                    processor = DataProcessor()
                    keywords = processor.get_optimal_keywords()
                    chunk_count = processor.brightdata_scrape(keywords)
                    processor.create_unified_data()
                    chatbot.data = chatbot.load_data()
                    st.success(f"Added {chunk_count} chunks!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        
        if st.button("📚 Process All Sources"):
            with st.spinner("Processing all sources..."):
                try:
                    from data_processor import DataProcessor
                    processor = DataProcessor()
                    processor.create_unified_data()
                    chatbot.data = chatbot.load_data()
                    st.success("Data processed!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
    
    if app_mode == "Resume Analyst":
        st.subheader("Resume Inputs")

        if "resume_analysis_result" not in st.session_state:
            st.session_state.resume_analysis_result = None
        if "resume_revision_result" not in st.session_state:
            st.session_state.resume_revision_result = None

        uploaded_resume = st.file_uploader(
            "Drag and drop your resume PDF here (or click to browse)",
            type=["pdf"],
            accept_multiple_files=False,
            help="Upload a PDF resume to auto-extract text"
        )

        if "resume_upload_signature" not in st.session_state:
            st.session_state.resume_upload_signature = ""
        if "resume_text_input" not in st.session_state:
            st.session_state.resume_text_input = ""

        if uploaded_resume is not None:
            upload_signature = f"{uploaded_resume.name}:{uploaded_resume.size}"
            if st.session_state.resume_upload_signature != upload_signature:
                with st.spinner("Extracting text from PDF..."):
                    extracted_resume_text = chatbot.extract_text_from_pdf(uploaded_resume)
                st.session_state.resume_upload_signature = upload_signature
                if extracted_resume_text:
                    st.session_state.resume_text_input = extracted_resume_text
                    st.success("PDF text extracted successfully.")
                else:
                    st.warning("Could not extract readable text from this PDF. Please paste resume text manually.")

        st.text_area(
            "Resume Text (required)",
            height=260,
            key="resume_text_input",
            placeholder="Paste extracted resume text from PDF/DOCX..."
        )
        resume_text = st.session_state.resume_text_input

        col1, col2 = st.columns(2)
        with col1:
            target_role = st.text_input("Target Job Role", placeholder="Software Engineer")
            industry = st.selectbox(
                "Industry",
                ["IT", "Healthcare", "Finance", "Marketing", "Sales", "Other"],
                index=0
            )
        with col2:
            experience_level = st.selectbox(
                "Experience Level",
                ["Entry-Level", "Mid-Level", "Senior-Level", "Executive"],
                index=1
            )
            company_name = st.text_input("Company Name (optional)", placeholder="Google")

        job_description = st.text_area(
            "Job Description (optional)",
            height=180,
            placeholder="Paste the target job description for tailored feedback..."
        )

        action_col1, action_col2 = st.columns([1, 1])

        with action_col1:
            analyze_clicked = st.button("🔍 Analyze Resume", type="primary")
        with action_col2:
            revise_clicked = st.button(
                "✨ Generate Revised Resume",
                disabled=st.session_state.resume_analysis_result is None,
                help="Runs a second generation step; use after analysis"
            )

        if analyze_clicked:
            if not resume_text.strip() or not target_role.strip():
                st.error("Please provide both Resume Text and Target Job Role.")
            else:
                start_time = time.perf_counter()
                with st.spinner("Analyzing resume with RAG context..."):
                    result = chatbot.analyze_resume(
                        resume_text=resume_text,
                        target_role=target_role,
                        industry=industry,
                        experience_level=experience_level,
                        job_description=job_description.strip() if job_description else None,
                        company_name=company_name.strip() if company_name else None,
                    )

                st.session_state.resume_analysis_result = result
                st.session_state.resume_revision_result = None
                st.session_state.last_resume_payload = {
                    "resume_text": resume_text,
                    "target_role": target_role,
                    "industry": industry,
                    "experience_level": experience_level,
                    "job_description": job_description.strip() if job_description else None,
                    "company_name": company_name.strip() if company_name else None,
                }
                st.caption(f"Analysis completed in {time.perf_counter() - start_time:.1f}s")

        if revise_clicked and st.session_state.resume_analysis_result is not None:
            payload = st.session_state.get("last_resume_payload", {})
            if not payload:
                st.warning("Run Analyze Resume first.")
            else:
                start_time = time.perf_counter()
                with st.spinner("Generating revised resume draft..."):
                    revision_result = chatbot.generate_resume_revision(
                        resume_text=payload.get("resume_text", ""),
                        target_role=payload.get("target_role", ""),
                        industry=payload.get("industry", "Other"),
                        experience_level=payload.get("experience_level", "Mid-Level"),
                        analysis_result=st.session_state.resume_analysis_result,
                        job_description=payload.get("job_description"),
                        company_name=payload.get("company_name"),
                    )
                st.session_state.resume_revision_result = revision_result
                st.caption(f"Revision completed in {time.perf_counter() - start_time:.1f}s")

        if st.session_state.resume_analysis_result:
            result = st.session_state.resume_analysis_result
            revision_result = st.session_state.resume_revision_result or {}
            analysis_meta = result.get("_meta", {})

            st.subheader("Final Resume Score")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Score", f"{result.get('resume_score', 0)}/100")
            with c2:
                st.metric("Confidence", f"{result.get('confidence', 0.0):.2f}")
            st.write(result.get("summary", ""))
            if analysis_meta:
                st.caption(
                    f"Analysis model: {analysis_meta.get('model_used', 'unknown')} | "
                    f"attempt: {analysis_meta.get('attempt', 0)} | "
                    f"fallback: {analysis_meta.get('used_fallback_model', False)}"
                )

            tab1, tab2, tab3, tab4 = st.tabs(["Structured JSON", "Revised Resume", "Improvement Plan", "Professional Report"])

            with tab1:
                st.subheader("Structured JSON Output")
                st.json(result)

            with tab2:
                revised_resume = revision_result.get("revised_resume", "")
                if revised_resume:
                    st.text_area("Revised Resume Draft", value=revised_resume, height=380)
                    revision_meta = revision_result.get("_meta", {})
                    if revision_meta:
                        st.caption(
                            f"Revision model: {revision_meta.get('model_used', 'unknown')} | "
                            f"attempt: {revision_meta.get('attempt', 0)} | "
                            f"fallback: {revision_meta.get('used_fallback_model', False)}"
                        )
                    st.download_button(
                        "⬇️ Download Revised Resume (.txt)",
                        data=revised_resume,
                        file_name="revised_resume.txt",
                        mime="text/plain",
                    )
                else:
                    st.info("Revised resume draft will appear here after analysis.")

            with tab3:
                plan_items = revision_result.get("improvement_plan", [])
                high_impact_changes = revision_result.get("high_impact_changes", [])

                if not plan_items:
                    plan_items = chatbot.build_improvement_plan_from_analysis(result)

                st.write("### Recommended Improvement Steps")
                if plan_items:
                    for idx, item in enumerate(plan_items, start=1):
                        st.write(f"{idx}. {item}")
                else:
                    st.info("Improvement steps will appear here after analysis.")

                if high_impact_changes:
                    st.write("### High-Impact Changes Made")
                    for change in high_impact_changes:
                        st.write(f"- {change}")

            with tab4:
                report_payload = st.session_state.get("last_resume_payload", {})
                professional_report = chatbot.build_professional_report(
                    analysis_result=result,
                    revision_result=revision_result,
                    request_payload=report_payload,
                )
                st.markdown(professional_report)
                st.download_button(
                    "⬇️ Download Professional Report (.md)",
                    data=professional_report,
                    file_name="resume_assessment_report.md",
                    mime="text/markdown",
                )

        return

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message.get("sources"):
                with st.expander("📖 Sources"):
                    for source in message["sources"]:
                        st.markdown(f"- {source}")
            
            if message.get("search_method"):
                st.caption(f"🔍 Method: {message['search_method']}")
    
    # Chat input
    user_question = st.chat_input("Ask any question...")
    
    if user_question:
        # Add user message
        st.session_state.messages.append({
            "role": "user", 
            "content": user_question
        })
        
        with st.chat_message("user"):
            st.markdown(user_question)
        
        # Process query
        with st.spinner("🔍 Searching..."):
            context, sources, search_method = chatbot.search_and_retrieve(user_question)
        
        # Generate response
        with st.spinner("🤖 Thinking..."):
            response = chatbot.generate_response(user_question, context)
        
        # Add assistant response
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": sources,
            "search_method": search_method
        })
        
        with st.chat_message("assistant"):
            st.markdown(response)
            
            if sources:
                with st.expander("📖 Sources"):
                    for source in set(sources):
                        st.markdown(f"- {source}")
            
            st.caption(f"🔍 Method: {search_method}")

if __name__ == "__main__":
    main()
