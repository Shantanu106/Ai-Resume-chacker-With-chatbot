"""
Complete RAG Chatbot with Multi-Source Support
Wikipedia + Bright Data + Real-time Search
"""

import os
import json
import re
import time
import hashlib
import html
import requests
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import quote_plus
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


HTML_TEMPLATE_CLASSIC = """
<style>
    .resume-sheet { background: #fff; padding: 24px; border: 1px solid #ddd; border-radius: 8px; }
    .classic-header { text-align: left; border-bottom: 2px solid #000; padding-bottom: 10px; margin-bottom: 16px; }
    .classic-name { font-size: 28px; font-weight: 700; font-family: Helvetica, Arial, sans-serif; color: #000; }
    .classic-contact { font-size: 13px; font-family: Helvetica, Arial, sans-serif; color: #333; margin-top: 6px; }
    .classic-section { margin-bottom: 16px; }
    .classic-heading { font-size: 15px; font-weight: 700; font-family: Helvetica, Arial, sans-serif; text-transform: uppercase; color: #000; border-bottom: 1px solid #ccc; padding-bottom: 5px; margin-bottom: 8px; }
    .classic-meta { font-size: 13px; font-weight: 600; font-family: Helvetica, Arial, sans-serif; color: #111; margin-bottom: 5px; }
    .classic-bullet { font-family: Helvetica, Arial, sans-serif; font-size: 13px; line-height: 1.45; color: #111; margin: 2px 0; }
</style>
"""

HTML_TEMPLATE_ATS_COMPACT = """
<style>
    .resume-sheet { background: #fff; padding: 24px; border: 1px solid #dbe3ff; border-radius: 10px; }
    .modern-header { text-align: left; padding: 10px 0; border-bottom: 2px solid #2563eb; margin-bottom: 16px; }
    .modern-name { font-size: 28px; font-weight: 700; font-family: 'Segoe UI', Arial, sans-serif; color: #1e40af; }
    .modern-contact { font-size: 13px; color: #334155; margin-top: 6px; font-family: 'Segoe UI', Arial, sans-serif; }
    .modern-section { margin-bottom: 14px; }
    .modern-heading { font-size: 16px; font-weight: 700; font-family: 'Segoe UI', Arial, sans-serif; color: #2563eb; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; }
    .modern-bullet { font-family: 'Segoe UI', Arial, sans-serif; font-size: 13px; line-height: 1.5; color: #334155; margin: 2px 0; }
    .modern-meta { font-size: 13px; font-weight: 600; color: #1d4ed8; margin: 8px 0 4px 0; }
</style>
"""

HTML_TEMPLATE_MINIMAL = """
<style>
    .resume-sheet { background: #fff; padding: 28px; border: 1px solid #e5e7eb; border-radius: 8px; }
    .minimal-header { text-align: center; border-bottom: 1px solid #111; padding-bottom: 10px; margin-bottom: 16px; }
    .minimal-name { font-size: 24px; font-weight: 300; color: #000; letter-spacing: 1.8px; text-transform: uppercase; font-family: Arial, sans-serif; }
    .minimal-contact { font-size: 12px; color: #1f2937; margin-top: 6px; font-family: Arial, sans-serif; }
    .minimal-section { margin-bottom: 14px; }
    .minimal-heading { font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; color: #111; margin-bottom: 6px; font-family: Arial, sans-serif; }
    .minimal-meta { font-size: 12px; font-weight: 700; color: #111; margin: 5px 0 2px 0; font-family: Arial, sans-serif; }
    .minimal-bullet { font-size: 12px; line-height: 1.55; color: #111; margin-left: 12px; font-family: Arial, sans-serif; }
</style>
"""

ATS_REQUIRED_SECTIONS = [
    "Contact Information",
    "Professional Summary",
    "Work Experience",
    "Education",
    "Skills",
    "Certifications",
]

ATS_TEMPLATE_CATALOG: List[Dict[str, str]] = [
    {
    "id": "jobscan_classic",
    "name": "Jobscan Classic ATS",
    "provider": "Jobscan",
    "source_url": "https://www.jobscan.co/resume-templates",
    "format": ".docx",
    "notes": "Chronological, keyword-friendly, simple text-first structure.",
    "style_focus": "classic-professional",
    },
    {
    "id": "jobscan_hybrid",
    "name": "Jobscan Hybrid ATS",
    "provider": "Jobscan",
    "source_url": "https://www.jobscan.co/resume-templates",
    "format": ".docx",
    "notes": "Balanced summary + skills + experience flow for ATS parsing.",
    "style_focus": "hybrid-keyword",
    },
    {
    "id": "myresume_chrono",
    "name": "MyResume Chronological",
    "provider": "My Resume Templates",
    "source_url": "https://my-resume-templates.com/free-ats-friendly-resume-templates/",
    "format": ".docx",
    "notes": "Single-column format optimized for standard ATS ingestion.",
    "style_focus": "chronological",
    },
    {
    "id": "myresume_skills",
    "name": "MyResume Skills-Forward",
    "provider": "My Resume Templates",
    "source_url": "https://my-resume-templates.com/free-ats-friendly-resume-templates/",
    "format": ".docx",
    "notes": "Skills-first ordering while keeping section titles ATS-standard.",
    "style_focus": "skills-first",
    },
    {
    "id": "resume_llm_markdown",
    "name": "Resume-LLM Markdown DOCX",
    "provider": "GitHub resume-llm/resume-ai",
    "source_url": "https://github.com/resume-llm/resume-ai",
    "format": "Markdown -> DOCX",
    "notes": "Privacy-first markdown pipeline with ATS-safe structured output.",
    "style_focus": "markdown-pipeline",
    },
    {
    "id": "resume_lm_open_source",
    "name": "Resume-LM Open Source",
    "provider": "GitHub olyaiy/resume-lm",
    "source_url": "https://github.com/olyaiy/resume-lm",
    "format": "Text/Markdown",
    "notes": "Open-source clean output references for ATS-compatible authoring.",
    "style_focus": "open-source-clean",
    },
]


def get_ats_template_catalog() -> List[Dict[str, str]]:
    return ATS_TEMPLATE_CATALOG


def get_ats_template_by_id(template_id: str) -> Dict[str, str]:
    for template in ATS_TEMPLATE_CATALOG:
        if template.get("id") == template_id:
            return template
    return ATS_TEMPLATE_CATALOG[0]


def evaluate_ats_template_text(template_text: str) -> Dict[str, Any]:
    text = (template_text or "").strip()
    lowered = text.lower()
    issues: List[str] = []

    disallowed_patterns = [
        (r"<table|</table|<tr|</tr|<td|</td>", "Avoid table-based formatting."),
        (r"<img|!\[[^\]]*\]\([^\)]*\)", "Avoid images/icons in the resume body."),
        (r"\|\s*---\s*\|", "Avoid markdown table syntax."),
        (r"\S+\s{4,}\S+", "Detected multi-column-like spacing; keep a single reading flow."),
        (r"\t", "Avoid tab-aligned layout; use normal line flow."),
    ]

    for pattern, message in disallowed_patterns:
        if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
            issues.append(message)

    missing_sections = []
    for section in ATS_REQUIRED_SECTIONS:
        has_section = bool(re.search(rf"(?im)^\s*{re.escape(section)}\s*$", text))
        if not has_section:
            missing_sections.append(section)
    if missing_sections:
        issues.append("Missing standard ATS headings: " + ", ".join(missing_sections) + ".")

    if len(text.split()) < 80:
        issues.append("Template content is too short; include enough guidance to be useful.")

    status = "Pass" if not issues else "Needs Improvement"
    checks = {
        "single_column_flow": not any("single reading flow" in i.lower() for i in issues),
        "no_tables_or_graphics": not any("table" in i.lower() or "images" in i.lower() for i in issues),
        "standard_section_headings": len(missing_sections) == 0,
        "text_only_labels": not any("tab-aligned" in i.lower() for i in issues),
    }

    return {
        "status": status,
        "issues": issues,
        "checks": checks,
    }


def _escape(text: Any) -> str:
        return html.escape(str(text or "")).replace("\n", "<br>")


def _split_csv_like(value: str) -> List[str]:
        if not value:
                return []
        parts = re.split(r"[,;|]", value)
        cleaned = [p.strip() for p in parts if p and p.strip()]
        return list(dict.fromkeys(cleaned))


def _extract_resume_sections_from_text(text: str) -> Dict[str, Any]:
        """Parse plain resume text/template into sectioned data for HTML templates"""
        lines = [ln.strip() for ln in (text or "").splitlines()]
        lines = [ln for ln in lines if ln]

        section_map = {
                "professional summary": "professional_summary",
                "work experience": "work_experience",
                "education": "education",
                "skills": "skills",
                "certifications": "certifications",
                "contact information": "contact_information",
        }

        section_data: Dict[str, List[str]] = {
                "contact_information": [],
                "professional_summary": [],
                "work_experience": [],
                "education": [],
                "skills": [],
                "certifications": [],
        }

        full_name = lines[0] if lines else "[Full Name]"
        contact_line = lines[1] if len(lines) > 1 and ("@" in lines[1] or "|" in lines[1]) else "[Phone] | [Email] | [LinkedIn URL] | [City, State]"

        current_section = None
        for ln in lines:
                key = section_map.get(ln.lower().rstrip(":"))
                if key:
                        current_section = key
                        continue
                if current_section:
                        section_data[current_section].append(ln)

        experience_entries: List[Dict[str, Any]] = []
        current_entry: Dict[str, Any] | None = None
        for ln in section_data["work_experience"]:
                if ln.startswith("-") or ln.startswith("•"):
                        bullet = ln.lstrip("-• ").strip()
                        if current_entry is None:
                                current_entry = {"title": "Experience", "bullets": []}
                        if bullet:
                                current_entry["bullets"].append(bullet)
                else:
                        if current_entry is not None:
                                experience_entries.append(current_entry)
                        current_entry = {"title": ln, "bullets": []}
        if current_entry is not None:
                experience_entries.append(current_entry)

        skills_value = ", ".join(section_data["skills"])
        certifications_value = ", ".join(section_data["certifications"])

        return {
                "full_name": full_name,
                "contact": contact_line,
                "professional_summary": " ".join(section_data["professional_summary"]).strip(),
                "experience": experience_entries,
                "education": "<br>".join(_escape(x) for x in section_data["education"]) if section_data["education"] else "Degree in [Field] | [University] | [Year]",
                "skills": _split_csv_like(skills_value),
                "certifications": _split_csv_like(certifications_value),
        }


def format_resume_to_html(template_type: str, resume_data: Dict[str, Any]) -> str:
    """Generate HTML preview for selected visual template"""
    normalized_template = "ATS Compact" if template_type == "Modern (Sidebar Style)" else template_type
    template_map = {
        "Classic Professional": HTML_TEMPLATE_CLASSIC,
        "ATS Compact": HTML_TEMPLATE_ATS_COMPACT,
        "Minimalist Clean": HTML_TEMPLATE_MINIMAL,
    }
    css = template_map.get(normalized_template, HTML_TEMPLATE_CLASSIC)

    full_name = _escape(resume_data.get("full_name", "[Full Name]"))
    contact = _escape(resume_data.get("contact", "[Phone] | [Email] | [LinkedIn URL] | [City, State]"))
    summary = _escape(resume_data.get("professional_summary", "Professional summary goes here."))
    education = resume_data.get("education", "Degree in [Field] | [University] | [Year]")
    skills = ", ".join([str(s) for s in resume_data.get("skills", []) if str(s).strip()]) or "[Skill 1], [Skill 2], [Skill 3]"
    certifications = ", ".join([str(c) for c in resume_data.get("certifications", []) if str(c).strip()]) or "[Certification 1], [Certification 2]"

    experience_html = ""
    for exp in resume_data.get("experience", []) or []:
        title = _escape(exp.get("title", "[Job Title] | [Company] | [Start Date] - [End Date]"))
        bullets = exp.get("bullets", []) or []
        bullet_html = "".join(
            f"<div class='classic-bullet'>• {_escape(b)}</div>" if normalized_template == "Classic Professional" else
            f"<div class='modern-bullet'>• {_escape(b)}</div>" if normalized_template == "ATS Compact" else
            f"<div class='minimal-bullet'>• {_escape(b)}</div>"
            for b in bullets
        )

        if normalized_template == "Classic Professional":
            experience_html += f"<div class='classic-meta'>{title}</div>{bullet_html}"
        elif normalized_template == "ATS Compact":
            experience_html += f"<div class='modern-meta'>{title}</div>{bullet_html}"
        else:
            experience_html += f"<div class='minimal-meta'>{title}</div>{bullet_html}"

    if not experience_html:
        experience_html = (
            "<div class='classic-meta'>[Job Title] | [Company] | [Start Date] - [End Date]</div><div class='classic-bullet'>• Impact bullet with metric</div>"
            if normalized_template == "Classic Professional"
            else "<div class='modern-meta'>[Job Title] | [Company] | [Start Date] - [End Date]</div><div class='modern-bullet'>• Impact bullet with metric</div>"
            if normalized_template == "ATS Compact"
            else "<div class='minimal-meta'>[Job Title] | [Company] | [Start Date] - [End Date]</div><div class='minimal-bullet'>• Impact bullet with metric</div>"
        )

    if normalized_template == "ATS Compact":
        return f"""
{css}
<div class="resume-sheet">
    <div class="modern-header">
        <div class="modern-name">{full_name}</div>
        <div class="modern-contact">{contact}</div>
    </div>
    <div class="modern-section">
        <div class="modern-heading">Professional Summary</div>
        <div class="modern-bullet">{summary}</div>
    </div>
    <div class="modern-section">
        <div class="modern-heading">Work Experience</div>
        {experience_html}
    </div>
    <div class="modern-section">
        <div class="modern-heading">Education</div>
        <div class="modern-bullet">{education}</div>
    </div>
    <div class="modern-section">
        <div class="modern-heading">Skills</div>
        <div class="modern-bullet">{_escape(skills)}</div>
    </div>
    <div class="modern-section">
        <div class="modern-heading">Certifications</div>
        <div class="modern-bullet">{_escape(certifications)}</div>
    </div>
</div>
"""

    if normalized_template == "Minimalist Clean":
        return f"""
{css}
<div class="resume-sheet">
    <div class="minimal-header">
        <div class="minimal-name">{full_name}</div>
        <div class="minimal-contact">{contact}</div>
    </div>
    <div class="minimal-section">
        <div class="minimal-heading">Professional Summary</div>
        <div class="minimal-bullet">{summary}</div>
    </div>
    <div class="minimal-section">
        <div class="minimal-heading">Work Experience</div>
        {experience_html}
    </div>
    <div class="minimal-section">
        <div class="minimal-heading">Education</div>
        <div class="minimal-bullet">{education}</div>
    </div>
    <div class="minimal-section">
        <div class="minimal-heading">Skills</div>
        <div class="minimal-bullet">{_escape(skills)}</div>
    </div>
    <div class="minimal-section">
        <div class="minimal-heading">Certifications</div>
        <div class="minimal-bullet">{_escape(certifications)}</div>
    </div>
</div>
"""

    return f"""
{css}
<div class="resume-sheet">
    <div class="classic-header">
        <div class="classic-name">{full_name}</div>
        <div class="classic-contact">{contact}</div>
    </div>
    <div class="classic-section">
        <div class="classic-heading">Professional Summary</div>
        <div class="classic-bullet">{summary}</div>
    </div>
    <div class="classic-section">
        <div class="classic-heading">Work Experience</div>
        {experience_html}
    </div>
    <div class="classic-section">
        <div class="classic-heading">Education</div>
        <div class="classic-bullet">{education}</div>
    </div>
    <div class="classic-section">
        <div class="classic-heading">Skills</div>
        <div class="classic-bullet">{_escape(skills)}</div>
    </div>
    <div class="classic-section">
        <div class="classic-heading">Certifications</div>
        <div class="classic-bullet">{_escape(certifications)}</div>
    </div>
</div>
"""


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


class ResumeTemplateOutputModel(BaseModel):
    resume_template: str
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    evidence: List[str] = Field(default_factory=list)

class RAGChatbot:
    """Complete RAG Chatbot with all features"""
    
    def __init__(self, chat_model=None, fast_mode=True):
        self.embeddings = OllamaEmbeddings(model=os.getenv("EMBEDDING_MODEL"))
        self.chat_model = chat_model or os.getenv("CHAT_MODEL")
        self.fast_mode = fast_mode
        self.llm = self._create_llm(self.chat_model)
        self._analysis_cache: Dict[str, Dict[str, Any]] = {}
        self.data = self.load_data()
        self.embedding_matrix = None
        self.embedding_norms = None
        self.valid_chunks = []
        self._prepare_embeddings_index()

    def _create_llm(self, model_name):
        """Create tuned LLM instance for faster responses"""
        timeout_raw = os.getenv("OLLAMA_TIMEOUT_SEC", "90")
        try:
            timeout_sec = max(15, int(float(timeout_raw)))
        except (TypeError, ValueError):
            timeout_sec = 90

        return ChatOllama(
            model=model_name,
            temperature=0,
            num_ctx=2048,
            num_predict=300 if self.fast_mode else 420,
            client_kwargs={"timeout": timeout_sec},
        )

    def _invoke_with_timeout(self, model_client, prompt: str, timeout_sec: int):
        """Invoke model with a strict wall-time cap to avoid UI hanging"""
        if not timeout_sec or timeout_sec <= 0:
            return model_client.invoke(prompt)

        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(model_client.invoke, prompt)
        try:
            return future.result(timeout=timeout_sec)
        except FuturesTimeoutError as err:
            future.cancel()
            raise TimeoutError(f"LLM call timed out after {timeout_sec}s") from err
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

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

    def _validate_template_payload(self, payload: Dict[str, Any]):
        """Validate ATS template payload against strict schema"""
        model = ResumeTemplateOutputModel.model_validate(payload)
        validated = model.model_dump()
        template_check = evaluate_ats_template_text(validated.get("resume_template", ""))
        if template_check.get("status") != "Pass":
            raise ValueError("ATS template validation failed: " + "; ".join(template_check.get("issues", [])))
        return validated

    def _extract_resume_skills(self, resume_text: str) -> List[str]:
        """Extract probable skills from resume text using lightweight heuristics"""
        known_skills = [
            "Python", "Java", "C++", "JavaScript", "TypeScript", "SQL", "NoSQL", "AWS", "Azure", "GCP",
            "Docker", "Kubernetes", "CI/CD", "Git", "REST API", "Microservices", "Machine Learning",
            "Data Analysis", "Pandas", "NumPy", "TensorFlow", "PyTorch", "Linux", "Agile", "Scrum",
            "Leadership", "Communication", "Problem Solving", "Project Management", "Stakeholder Management",
        ]
        text = (resume_text or "")
        lowered = text.lower()
        found = []
        for skill in known_skills:
            if skill.lower() in lowered:
                found.append(skill)
        return list(dict.fromkeys(found))[:20]

    def _extract_resume_certifications(self, resume_text: str) -> List[str]:
        """Extract likely certifications from resume text"""
        text = resume_text or ""
        lines = [line.strip(" -•\t") for line in text.splitlines() if line.strip()]
        cert_markers = ["certification", "certified", "certificate", "aws", "azure", "google cloud", "pmp"]
        certs = []
        for line in lines:
            low = line.lower()
            if any(marker in low for marker in cert_markers) and len(line) <= 140:
                certs.append(line)
        return list(dict.fromkeys(certs))[:8]

    def _default_industry_certs(self, industry: str) -> List[str]:
        """Suggest baseline certifications by industry"""
        cert_map = {
            "IT": ["AWS Certified Solutions Architect", "Microsoft Azure Fundamentals"],
            "Healthcare": ["Certified Professional in Healthcare Quality (CPHQ)", "HIPAA Compliance Training"],
            "Finance": ["CFA Level I", "FRM Part I"],
            "Marketing": ["Google Ads Certification", "HubSpot Content Marketing Certification"],
            "Sales": ["Certified Professional Sales Person (CPSP)", "HubSpot Sales Software Certification"],
        }
        return cert_map.get(industry, ["Industry-relevant professional certification"])

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
        timeout_default = 18 if self.fast_mode else 35
        timeout_raw = os.getenv("LLM_JSON_TASK_TIMEOUT_SEC", str(timeout_default))
        try:
            task_timeout = max(8, int(float(timeout_raw)))
        except (TypeError, ValueError):
            task_timeout = timeout_default

        for attempt in range(1, max_attempts + 1):
            for idx, model_name in enumerate(model_sequence):
                model_client = self.llm if model_name == self.chat_model else self._create_llm(model_name)
                try:
                    response = self._invoke_with_timeout(model_client, repair_prompt, task_timeout)
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
        
        timeout_default = 14 if self.fast_mode else 30
        timeout_raw = os.getenv("LLM_CHAT_TIMEOUT_SEC", str(timeout_default))
        try:
            chat_timeout = max(6, int(float(timeout_raw)))
        except (TypeError, ValueError):
            chat_timeout = timeout_default

        try:
            response = self._invoke_with_timeout(self.llm, prompt, chat_timeout)
            return getattr(response, "content", "") or "I couldn't generate a full answer right now. Please retry."
        except Exception:
            return "The model took too long to respond. Try again, switch to a lighter model, or keep Fast Mode on."

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

    def _compact_text(self, text: str, max_chars: int) -> str:
        """Trim long text blocks to reduce prompt size and latency"""
        value = (text or "").strip()
        if len(value) <= max_chars:
            return value
        return value[:max_chars].rstrip() + "\n...[truncated for speed]"

    def _analysis_cache_key(self, payload: Dict[str, Any]) -> str:
        """Create stable cache key for repeated analysis requests"""
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.md5(encoded.encode("utf-8", errors="ignore")).hexdigest()

    def _put_analysis_cache(self, key: str, value: Dict[str, Any]):
        """Store analysis result with a small bounded in-memory cache"""
        if len(self._analysis_cache) >= 24:
            self._analysis_cache.pop(next(iter(self._analysis_cache)))
        self._analysis_cache[key] = value

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

    def _google_realtime_search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Fetch real-time web results using Google Custom Search JSON API"""
        api_key = (os.getenv("GOOGLE_API_KEY") or "").strip()
        cse_id = (os.getenv("GOOGLE_CSE_ID") or "").strip()
        if not api_key or not cse_id:
            return []

        try:
            response = requests.get(
                "https://www.googleapis.com/customsearch/v1",
                params={
                    "key": api_key,
                    "cx": cse_id,
                    "q": query,
                    "num": max(1, min(max_results, 10)),
                },
                timeout=10,
            )
            response.raise_for_status()
            payload = response.json() if response.content else {}
            items = payload.get("items", []) if isinstance(payload, dict) else []
            results: List[Dict[str, str]] = []
            for item in items[:max_results]:
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title") or "").strip()
                snippet = str(item.get("snippet") or "").strip()
                link = str(item.get("link") or "").strip()
                if title or snippet:
                    results.append({"title": title, "snippet": snippet, "url": link})
            return results
        except Exception:
            return []

    def _naukri_realtime_search(self, target_role: str, industry: str, experience_level: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Fetch real-time job insights from a user-configured Naukri API endpoint"""
        api_url = (os.getenv("NAUKRI_API_URL") or "").strip()
        api_key = (os.getenv("NAUKRI_API_KEY") or "").strip()
        app_id = (os.getenv("NAUKRI_APP_ID") or "").strip()
        if not api_url or not api_key:
            return []

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        }
        if app_id:
            headers["X-App-Id"] = app_id

        params = {
            "q": target_role,
            "industry": industry,
            "experience_level": experience_level,
            "limit": max(1, min(max_results, 20)),
        }

        try:
            response = requests.get(api_url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            payload = response.json() if response.content else {}
        except Exception:
            return []

        records: List[Any] = []
        if isinstance(payload, dict):
            for key in ("jobs", "results", "data", "items"):
                value = payload.get(key)
                if isinstance(value, list):
                    records = value
                    break
                if isinstance(value, dict):
                    nested = value.get("jobs") or value.get("items") or value.get("results")
                    if isinstance(nested, list):
                        records = nested
                        break
        elif isinstance(payload, list):
            records = payload

        normalized: List[Dict[str, str]] = []
        for item in records[:max_results]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or item.get("jobTitle") or item.get("position") or "").strip()
            company = str(item.get("company") or item.get("companyName") or "").strip()
            skills_raw = item.get("skills") or item.get("keySkills") or []
            if isinstance(skills_raw, list):
                skills = ", ".join([str(s).strip() for s in skills_raw if str(s).strip()][:8])
            else:
                skills = str(skills_raw or "").strip()
            description = str(item.get("description") or item.get("summary") or "").strip()
            job_url = str(item.get("url") or item.get("jobUrl") or item.get("link") or "").strip()
            snippet_parts = [p for p in [company, skills, description] if p]
            if title or snippet_parts:
                normalized.append(
                    {
                        "title": title or "Naukri Job Insight",
                        "snippet": " | ".join(snippet_parts)[:420],
                        "url": job_url,
                    }
                )
        return normalized

    def _build_realtime_job_market_context(self, target_role: str, industry: str, experience_level: str, enabled: bool = True) -> Tuple[str, List[str]]:
        """Aggregate real-time context from Google and Naukri APIs when configured"""
        if not enabled:
            return "", []

        max_results = 3 if self.fast_mode else 5
        google_query = f"{target_role} {industry} skills requirements ATS resume site:naukri.com"
        google_results = self._google_realtime_search(google_query, max_results=max_results)
        naukri_results = self._naukri_realtime_search(
            target_role=target_role,
            industry=industry,
            experience_level=experience_level,
            max_results=max_results,
        )

        context_parts: List[str] = []
        sources: List[str] = []

        if google_results:
            snippets = []
            for item in google_results:
                snippets.append(f"- {item.get('title', 'Result')}: {item.get('snippet', '')}")
                if item.get("url"):
                    sources.append(item["url"])
            context_parts.append("[Google Real-time Signals]\n" + "\n".join(snippets[:max_results]))
            sources.append("google_custom_search_api")

        if naukri_results:
            snippets = []
            for item in naukri_results:
                snippets.append(f"- {item.get('title', 'Job')}: {item.get('snippet', '')}")
                if item.get("url"):
                    sources.append(item["url"])
            context_parts.append("[Naukri Real-time Signals]\n" + "\n".join(snippets[:max_results]))
            sources.append("naukri_api")

        return "\n\n".join(context_parts), list(dict.fromkeys(sources))

    def _build_job_recommendations(
        self,
        target_role: str,
        industry: str,
        experience_level: str,
        baseline_skills: List[str],
        enabled: bool = True,
    ) -> List[Dict[str, str]]:
        """Build explicit job recommendations from real-time APIs with safe fallback"""
        recommendations: List[Dict[str, str]] = []

        if enabled:
            naukri_jobs = self._naukri_realtime_search(
                target_role=target_role,
                industry=industry,
                experience_level=experience_level,
                max_results=4 if self.fast_mode else 6,
            )
            for item in naukri_jobs:
                title = (item.get("title") or "Recommended Role").strip()
                snippet = (item.get("snippet") or "").strip()
                recommendations.append(
                    {
                        "title": title,
                        "source": "Naukri",
                        "url": item.get("url", ""),
                        "rationale": snippet[:220] or f"Matches {target_role} intent and current market keywords.",
                    }
                )

            google_query = f"{target_role} {industry} jobs"
            google_hits = self._google_realtime_search(google_query, max_results=4 if self.fast_mode else 6)
            for item in google_hits:
                title = (item.get("title") or "Live Job Signal").strip()
                snippet = (item.get("snippet") or "").strip()
                recommendations.append(
                    {
                        "title": title,
                        "source": "Google",
                        "url": item.get("url", ""),
                        "rationale": snippet[:220] or f"Live listing signal related to {target_role}.",
                    }
                )

        if not recommendations:
            skills_hint = ", ".join((baseline_skills or [])[:4]) or "core role skills"
            fallback_queries = [
                f"{target_role} jobs {industry}",
                f"{target_role} remote jobs",
                f"{target_role} {experience_level} openings",
            ]
            for query in fallback_queries:
                recommendations.append(
                    {
                        "title": f"{target_role} Opportunities ({industry})",
                        "source": "Generated",
                        "url": f"https://www.naukri.com/{quote_plus(query).replace('+', '-')}-jobs",
                        "rationale": f"Suggested search aligned to {experience_level} profile and skills: {skills_hint}.",
                    }
                )

        deduped: List[Dict[str, str]] = []
        seen = set()
        for rec in recommendations:
            key = (rec.get("title", "").lower(), rec.get("url", "").lower())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(rec)

        return deduped[:6]

    def _build_resume_rag_context(self, resume_text, target_role, industry, experience_level, job_description=None, company_name=None, include_realtime_apis=True):
        """Build RAG context for resume analysis"""
        query = (
            f"Resume analysis best practices for {target_role} in {industry} at {experience_level} level. "
            "Include ATS, keywords, measurable impact, and required skills."
        )

        context_parts = []
        sources = []

        local_k = 2 if self.fast_mode else 4
        local_results = self.similarity_search(query, self.data, k=local_k)
        for similarity, chunk in local_results:
            if similarity > 0.2:
                local_content = self._compact_text(chunk.get("content", ""), 700 if self.fast_mode else 1100)
                context_parts.append(f"[Local Guide] {local_content}")
                sources.append(chunk.get("metadata", {}).get("source", "local_dataset"))

        if job_description:
            compact_jd = self._compact_text(job_description, 2200 if self.fast_mode else 3200)
            context_parts.append(f"[Job Description]\n{compact_jd}")
            sources.append("job_description")

        if company_name and not self.fast_mode:
            company_articles = self.realtime_wikipedia_search(f"{company_name} company career jobs", max_results=1)
            for article in company_articles:
                context_parts.append(f"[Company Context: {article['title']}]\n{article['content'][:1200]}")
                sources.append(article["url"])

        baseline_skills = self._industry_skill_baseline(industry, target_role)
        context_parts.append("[Industry Skill Baseline] " + ", ".join(baseline_skills))
        sources.append("industry_skill_baseline")

        realtime_context, realtime_sources = self._build_realtime_job_market_context(
            target_role=target_role,
            industry=industry,
            experience_level=experience_level,
            enabled=bool(include_realtime_apis),
        )
        if realtime_context:
            context_parts.append(realtime_context)
            sources.extend(realtime_sources)

        return "\n\n".join(context_parts), list(dict.fromkeys(sources)), baseline_skills

    def analyze_resume(self, resume_text, target_role, industry, experience_level, job_description=None, company_name=None, include_realtime_apis=True):
        """Analyze resume and return strict JSON payload"""
        start_time = time.perf_counter()
        compact_resume = self._compact_text(resume_text, 5500 if self.fast_mode else 9000)
        compact_jd = self._compact_text(job_description or "", 2200 if self.fast_mode else 3200) if job_description else None

        cache_key = None
        try:
            cache_payload = {
                "resume_text": compact_resume,
                "target_role": target_role,
                "industry": industry,
                "experience_level": experience_level,
                "job_description": compact_jd,
                "company_name": company_name,
                "include_realtime_apis": bool(include_realtime_apis),
                "fast_mode": self.fast_mode,
                "model": self.chat_model,
            }
            cache_key = self._analysis_cache_key(cache_payload)
            cached_result = self._analysis_cache.get(cache_key)
            if cached_result:
                return dict(cached_result)
        except Exception:
            cache_key = None

        ats_precheck = self._extract_ats_checks(resume_text)
        try:
            rag_context, sources, baseline_skills = self._build_resume_rag_context(
                resume_text=resume_text,
                target_role=target_role,
                industry=industry,
                experience_level=experience_level,
                job_description=compact_jd,
                company_name=company_name,
                include_realtime_apis=include_realtime_apis,
            )
        except Exception:
            rag_context, sources = "", []
            baseline_skills = self._industry_skill_baseline(industry, target_role)

        compact_context = self._compact_text(rag_context, 1800 if self.fast_mode else 3200)

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
{compact_context if compact_context else 'No external context available.'}

Resume Text:
{compact_resume}

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

        try:
            parsed, meta = self._invoke_json_task(
                task_name="resume_analysis",
                prompt=prompt,
                schema_hint=analysis_schema_hint,
                validator=self._validate_analysis_payload,
                max_attempts=1 if self.fast_mode else 2,
            )
        except Exception as invoke_err:
            parsed, meta = {}, {
                "task": "resume_analysis",
                "model_used": "none",
                "attempt": 1,
                "used_fallback_model": False,
                "used_repair_prompt": False,
                "validation_error": f"Analyzer invocation failed: {invoke_err}",
            }

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
RAG Context: {compact_context if compact_context else 'No external context available.'}
Resume Text: {compact_resume}

Return JSON only."""

            try:
                retry_model = self._create_llm(self._get_backup_model_name())
                timeout_default = 16 if self.fast_mode else 30
                timeout_raw = os.getenv("LLM_JSON_TASK_TIMEOUT_SEC", str(timeout_default))
                try:
                    retry_timeout = max(8, int(float(timeout_raw)))
                except (TypeError, ValueError):
                    retry_timeout = timeout_default
                retry_response = self._invoke_with_timeout(retry_model, quality_retry_prompt, retry_timeout)
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
                "attempt": int(meta.get("attempt", 1) or 1),
                "used_fallback_model": False,
                "used_repair_prompt": True,
                "validation_error": "Low-quality content detected and replaced",
            }

        parsed.setdefault("feedback", {})
        parsed["feedback"]["ats_compatibility"] = ats_precheck
        parsed["job_recommendations"] = self._build_job_recommendations(
            target_role=target_role,
            industry=industry,
            experience_level=experience_level,
            baseline_skills=baseline_skills,
            enabled=bool(include_realtime_apis),
        )
        parsed["summary"] = (parsed.get("summary") or "") + (
            f" (RAG sources used: {len(sources)})" if sources else ""
        )
        if meta.get("task") == "resume_analysis":
            meta["cache_hit"] = False
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

        if cache_key:
            self._put_analysis_cache(cache_key, dict(parsed))

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

    def generate_resume_template(
        self,
        resume_text,
        analysis_results,
        target_role,
        industry,
        experience_level,
        selected_template_id="jobscan_classic",
    ):
        """Generate ATS-friendly plain-text resume template using analysis insights"""
        start_time = time.perf_counter()
        compact_resume = self._compact_text(resume_text, 4200 if self.fast_mode else 6500)
        compact_analysis = self._compact_text(json.dumps(analysis_results or {}, ensure_ascii=False), 2600 if self.fast_mode else 4200)
        selected_template = get_ats_template_by_id(selected_template_id)

        suggestions = (analysis_results or {}).get("suggestions", {}) or {}
        professional_summary = (suggestions.get("professional_summary") or "").strip()
        rewrite_bullets = suggestions.get("rewrite_bullets", {}) or {}
        keyword_list = [k for k in suggestions.get("keywords_to_include", []) if isinstance(k, str) and k.strip()]
        skills_to_add = [s for s in suggestions.get("skills_to_add", []) if isinstance(s, str) and s.strip()]

        extracted_skills = self._extract_resume_skills(resume_text)
        extracted_certs = self._extract_resume_certifications(resume_text)
        suggested_certs = self._default_industry_certs(industry)

        prompt = f"""System Role:
You are an Expert Resume Architect specializing in ATS-optimized resumes.
Generate a custom resume template using user resume + analysis, aligned to LinkedIn/Naukri ATS standards.

Input Context:
1) User Resume Text:
{compact_resume}

2) Analysis Results:
{compact_analysis}

3) Preferences:
- Target Role: {target_role}
- Industry: {industry}
- Experience Level: {experience_level}

4) Template Profile:
- Template Name: {selected_template.get('name')}
- Provider: {selected_template.get('provider')}
- Preferred Output Format: {selected_template.get('format')}
- Style Focus: {selected_template.get('style_focus')}
- Notes: {selected_template.get('notes')}

Instructions:
- Generate ATS-friendly resume template with strict rules:
    1. Single-column layout only.
    2. No tables, images, icons, graphics.
    3. Use standard section names: Contact Information, Professional Summary, Work Experience, Education, Skills, Certifications.
    4. Keep content ATS-safe and concise.

Content Rules:
- Contact Information line format:
    [Full Name]
    [Phone] | [Email] | [LinkedIn URL] | [City, State]
- Professional Summary:
    Use this if available: {professional_summary or '[missing]'}
    If missing, write 3 lines highlighting: {experience_level} {target_role} in {industry}, top skills, quantifiable result placeholder.
- Work Experience:
    Rewrite bullets in action+impact style using analysis rewrite guidance.
    Original sample bullet: {rewrite_bullets.get('original', 'Contributed to projects')}
    Rewritten sample bullet: {rewrite_bullets.get('rewritten', 'Led cross-functional work that improved performance by X%')}
    Include measurable impact placeholders where needed.
    Prioritize role-relevant keywords: {', '.join(keyword_list[:10]) if keyword_list else 'Use role-relevant ATS keywords'}
- Skills:
    Combine and deduplicate original + analysis additions + keywords.
    Original extracted skills: {', '.join(extracted_skills[:15]) if extracted_skills else 'N/A'}
    Skills to add: {', '.join(skills_to_add[:15]) if skills_to_add else 'N/A'}
    Keywords: {', '.join(keyword_list[:15]) if keyword_list else 'N/A'}
    If space allows, split into Technical and Soft Skills.
- Education:
    Format exactly: Degree in [Field] | University | Year
- Certifications:
    Include original certifications when present: {', '.join(extracted_certs) if extracted_certs else 'N/A'}
    Add industry-relevant certs if missing: {', '.join(suggested_certs)}

Output Requirements:
- Return ONLY valid JSON with this schema:
{{
    "resume_template": "plain text resume template only",
    "confidence": 0.0,
    "evidence": ["..."]
}}
- resume_template must be plain text with the exact section order:
    Contact Information, Professional Summary, Work Experience, Education, Skills, Certifications
- Ensure each section heading is a standalone line and matches case exactly.
- Do not include markdown fences.
"""

        schema_hint = """{
    "resume_template": "",
    "confidence": 0.0,
    "evidence": [""]
}"""

        parsed, meta = self._invoke_json_task(
            task_name="resume_template",
            prompt=prompt,
            schema_hint=schema_hint,
            validator=self._validate_template_payload,
            max_attempts=1 if self.fast_mode else 2,
        )

        if not parsed or not (parsed.get("resume_template") or "").strip():
            merged_skills = list(dict.fromkeys(extracted_skills + skills_to_add + keyword_list))
            if not merged_skills:
                merged_skills = self._industry_skill_baseline(industry, target_role)[:10]

            fallback_summary = professional_summary or (
                f"{experience_level} {target_role} with X+ years in {industry}. "
                f"Skilled in {', '.join(merged_skills[:3])}. Achieved measurable business outcomes with impact up to Y%."
            )

            fallback_template = (
                "Contact Information\n"
                "[Full Name]\n"
                "[Phone] | [Email] | [LinkedIn URL] | [City, State]\n\n"
                "Professional Summary\n"
                f"{fallback_summary}\n\n"
                "Work Experience\n"
                "[Job Title 1] | [Company 1] | [Start Date] - [End Date]\n"
                f"  - {rewrite_bullets.get('rewritten', 'Led key initiatives that improved process efficiency by X% and reduced turnaround time by Y%.')}\n"
                "  - Delivered role-aligned outcomes using ATS-relevant skills and cross-functional collaboration.\n\n"
                "[Job Title 2] | [Company 2] | [Start Date] - [End Date]\n"
                "  - Built and improved core workflows, increasing quality and reliability through measurable improvements.\n"
                "  - Supported strategic goals through data-driven execution and stakeholder communication.\n\n"
                "Education\n"
                "Degree in [Field] | [University] | [Year]\n\n"
                "Skills\n"
                + ", ".join(merged_skills[:20])
                + "\n\n"
                "Certifications\n"
                + ", ".join(list(dict.fromkeys(extracted_certs + suggested_certs))[:8])
            )

            template_check = evaluate_ats_template_text(fallback_template)
            if template_check.get("status") != "Pass":
                fallback_template += "\n[Certification 1], [Certification 2]"

            parsed = {
                "resume_template": fallback_template,
                "confidence": 0.62,
                "evidence": [
                    "Template built using analysis insights and ATS-safe section structure.",
                    "Skills merged from original resume extraction and analysis suggestions.",
                    "Fallback output generated to guarantee quick plain-text template delivery.",
                ],
            }
            meta = {
                "task": "resume_template",
                "model_used": "fallback_template",
                "attempt": max(int((meta or {}).get("attempt", 1) or 1), 1),
                "used_fallback_model": False,
                "used_repair_prompt": True,
                "validation_error": "Generated fallback template payload",
            }

        parsed["_meta"] = meta
        self._log_evaluation_event(
            "resume_template",
            {
                "model_requested": self.chat_model,
                "model_used": meta.get("model_used"),
                "attempt": meta.get("attempt", 0),
                "used_fallback_model": meta.get("used_fallback_model", False),
                "latency_sec": round(time.perf_counter() - start_time, 3),
                "confidence": parsed.get("confidence", 0.0),
                "template_chars": len(parsed.get("resume_template", "")),
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
        default_model = os.getenv("DEFAULT_CHAT_MODEL") or ("gemma2:2b" if fast_mode else "llama3:latest")
        default_index = available_models.index(default_model) if default_model in available_models else 0
        selected_model = st.selectbox(
            "Choose Chat Model:",
            available_models,
            index=default_index,
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
        if "resume_template_result" not in st.session_state:
            st.session_state.resume_template_result = None

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

        use_realtime_apis = st.checkbox(
            "Use Naukri + Google real-time API context",
            value=True,
            help="Requires API keys in .env; if not configured, app safely falls back to local context.",
        )

        job_description = st.text_area(
            "Job Description (optional)",
            height=180,
            placeholder="Paste the target job description for tailored feedback..."
        )

        template_catalog = get_ats_template_catalog()
        if "selected_ats_template_id" not in st.session_state:
            st.session_state.selected_ats_template_id = template_catalog[0]["id"]

        template_label_map = {
            f"{template['name']} ({template['provider']})": template["id"]
            for template in template_catalog
        }
        selected_template_label = st.selectbox(
            "ATS Template Library (Free)",
            options=list(template_label_map.keys()),
            index=max(
                0,
                [
                    i
                    for i, template in enumerate(template_catalog)
                    if template["id"] == st.session_state.selected_ats_template_id
                ][0]
                if any(template["id"] == st.session_state.selected_ats_template_id for template in template_catalog)
                else 0,
            ),
            help="Choose an ATS-focused template profile for generation.",
        )
        st.session_state.selected_ats_template_id = template_label_map[selected_template_label]
        selected_template_meta = get_ats_template_by_id(st.session_state.selected_ats_template_id)
        st.caption(
            f"Format: {selected_template_meta.get('format')} | Focus: {selected_template_meta.get('style_focus')}"
        )
        st.markdown(
            f"Source: [{selected_template_meta.get('provider')}]({selected_template_meta.get('source_url')})"
        )

        action_col1, action_col2 = st.columns([1, 1])
        action_col3 = st.columns([1])[0]

        with action_col1:
            analyze_clicked = st.button("🔍 Analyze Resume", type="primary")
        with action_col2:
            revise_clicked = st.button(
                "✨ Generate Revised Resume",
                disabled=st.session_state.resume_analysis_result is None,
                help="Runs a second generation step; use after analysis"
            )
        with action_col3:
            template_clicked = st.button(
                "📄 Generate ATS Template",
                disabled=st.session_state.resume_analysis_result is None,
                help="Creates an ATS-friendly plain-text resume template from analysis insights"
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
                        include_realtime_apis=use_realtime_apis,
                    )

                st.session_state.resume_analysis_result = result
                st.session_state.resume_revision_result = None
                st.session_state.resume_template_result = None
                st.session_state.last_resume_payload = {
                    "resume_text": resume_text,
                    "target_role": target_role,
                    "industry": industry,
                    "experience_level": experience_level,
                    "job_description": job_description.strip() if job_description else None,
                    "company_name": company_name.strip() if company_name else None,
                    "include_realtime_apis": use_realtime_apis,
                }

                try:
                    with st.spinner("Preparing ATS template..."):
                        st.session_state.resume_template_result = chatbot.generate_resume_template(
                            resume_text=resume_text,
                            analysis_results=result,
                            target_role=target_role,
                            industry=industry,
                            experience_level=experience_level,
                            selected_template_id=st.session_state.selected_ats_template_id,
                        )
                except Exception as template_err:
                    st.session_state.resume_template_result = {
                        "resume_template": (
                            "[Full Name]\n"
                            "[Phone] | [Email] | [LinkedIn URL] | [City, State]\n\n"
                            "Professional Summary\n"
                            f"{experience_level} {target_role} in {industry} with measurable impact.\n\n"
                            "Work Experience\n"
                            "[Job Title] | [Company] | [Start Date] - [End Date]\n"
                            "  - Delivered role-relevant outcomes with quantifiable metrics.\n\n"
                            "Education\n"
                            "Degree in [Field] | [University] | [Year]\n\n"
                            "Skills\n"
                            "[Skill 1], [Skill 2], [Skill 3]\n\n"
                            "Certifications\n"
                            "[Certification 1], [Certification 2]"
                        ),
                        "confidence": 0.45,
                        "evidence": ["Auto-template fallback used after generation error."],
                        "_meta": {
                            "task": "resume_template",
                            "model_used": "ui_fallback",
                            "attempt": 1,
                            "used_fallback_model": False,
                            "used_repair_prompt": False,
                            "validation_error": str(template_err),
                        },
                    }
                    st.warning("Template generation hit an issue, so a fallback ATS template was created.")
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

        if template_clicked and st.session_state.resume_analysis_result is not None:
            payload = st.session_state.get("last_resume_payload", {})
            if not payload:
                st.warning("Run Analyze Resume first.")
            else:
                start_time = time.perf_counter()
                with st.spinner("Generating ATS-friendly resume template..."):
                    template_result = chatbot.generate_resume_template(
                        resume_text=payload.get("resume_text", ""),
                        analysis_results=st.session_state.resume_analysis_result,
                        target_role=payload.get("target_role", ""),
                        industry=payload.get("industry", "Other"),
                        experience_level=payload.get("experience_level", "Mid-Level"),
                        selected_template_id=st.session_state.selected_ats_template_id,
                    )
                st.session_state.resume_template_result = template_result
                st.caption(f"Template completed in {time.perf_counter() - start_time:.1f}s")

        if st.session_state.resume_analysis_result:
            result = st.session_state.resume_analysis_result
            revision_result = st.session_state.resume_revision_result or {}
            template_result = st.session_state.resume_template_result or {}
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

            job_recommendations = result.get("job_recommendations", []) or []
            st.write("### Job Recommendations")
            if job_recommendations:
                for rec in job_recommendations[:6]:
                    title = rec.get("title", "Recommended Role")
                    source = rec.get("source", "Unknown")
                    rationale = rec.get("rationale", "")
                    url = rec.get("url", "")
                    if url:
                        st.markdown(f"- **{title}** ({source}) — [{url}]({url})")
                    else:
                        st.markdown(f"- **{title}** ({source})")
                    if rationale:
                        st.caption(rationale)
            else:
                st.info("No live recommendations found yet. Add API keys in .env or disable/re-enable real-time context and re-run analysis.")

            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "Structured JSON",
                "Revised Resume",
                "ATS Template",
                "Improvement Plan",
                "Professional Report",
                "Visual Templates",
            ])

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
                template_text = template_result.get("resume_template", "")
                if template_text:
                    st.text_area("ATS-Friendly Resume Template", value=template_text, height=380)
                    template_meta = template_result.get("_meta", {})
                    if template_meta:
                        st.caption(
                            f"Template model: {template_meta.get('model_used', 'unknown')} | "
                            f"attempt: {template_meta.get('attempt', 0)} | "
                            f"fallback: {template_meta.get('used_fallback_model', False)}"
                        )
                    st.download_button(
                        "⬇️ Download ATS Template (.txt)",
                        data=template_text,
                        file_name="ats_resume_template.txt",
                        mime="text/plain",
                    )
                else:
                    st.info("ATS template will appear here after clicking 'Generate ATS Template'.")
                    if st.button("Generate ATS Template Now", key="tab_generate_ats_template"):
                        payload = st.session_state.get("last_resume_payload", {})
                        if payload and st.session_state.resume_analysis_result is not None:
                            with st.spinner("Generating ATS-friendly resume template..."):
                                st.session_state.resume_template_result = chatbot.generate_resume_template(
                                    resume_text=payload.get("resume_text", ""),
                                    analysis_results=st.session_state.resume_analysis_result,
                                    target_role=payload.get("target_role", ""),
                                    industry=payload.get("industry", "Other"),
                                    experience_level=payload.get("experience_level", "Mid-Level"),
                                    selected_template_id=st.session_state.selected_ats_template_id,
                                )
                            st.rerun()

                if template_text:
                    compliance = evaluate_ats_template_text(template_text)
                    if compliance.get("status") == "Pass":
                        st.success("ATS structure check: Pass")
                    else:
                        st.warning("ATS structure check: Needs Improvement")
                    for issue in compliance.get("issues", []):
                        st.write(f"- {issue}")

            with tab4:
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

            with tab5:
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

            with tab6:
                st.subheader("🎨 Visualize & Download")
                payload = st.session_state.get("last_resume_payload", {})
                original_resume_text = payload.get("resume_text", "")

                source_text = (
                    template_result.get("resume_template", "")
                    or revision_result.get("revised_resume", "")
                    or original_resume_text
                )

                visual_data = _extract_resume_sections_from_text(source_text)

                suggested_summary = ((result.get("suggestions", {}) or {}).get("professional_summary") or "").strip()
                if suggested_summary and not visual_data.get("professional_summary"):
                    visual_data["professional_summary"] = suggested_summary

                merged_skills = list(dict.fromkeys(
                    visual_data.get("skills", [])
                    + ((result.get("suggestions", {}) or {}).get("skills_to_add", []) or [])
                    + ((result.get("suggestions", {}) or {}).get("keywords_to_include", []) or [])
                ))
                visual_data["skills"] = [s for s in merged_skills if isinstance(s, str) and s.strip()][:24]

                try:
                    industry_value = payload.get("industry", "Other")
                    suggested_certs = chatbot._default_industry_certs(industry_value)
                except Exception:
                    suggested_certs = []
                merged_certs = list(dict.fromkeys((visual_data.get("certifications", []) or []) + suggested_certs))
                visual_data["certifications"] = merged_certs[:10]

                col_left, col_right = st.columns(2)

                with col_left:
                    st.markdown("**Original Resume (Text)**")
                    st.text_area(
                        "Original",
                        value=original_resume_text,
                        height=320,
                        key="visual_original_resume",
                    )

                with col_right:
                    selected_template = st.selectbox(
                        "Select ATS-Friendly Template:",
                        ["Classic Professional", "ATS Compact", "Minimalist Clean"],
                        key="visual_template_selector",
                    )

                    html_content = format_resume_to_html(selected_template, visual_data)
                    st.markdown(f"**Preview: {selected_template}**")
                    st.markdown(html_content, unsafe_allow_html=True)

                    st.download_button(
                        label="Download Resume (HTML)",
                        data=html_content,
                        file_name="resume_visual_template.html",
                        mime="text/html",
                        key="download_visual_html",
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
