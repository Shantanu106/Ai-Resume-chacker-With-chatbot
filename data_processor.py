"""
Complete Data Processor for Local RAG System
Handles Wikipedia scraping, Bright Data integration, and data processing
"""

import os
import json
import wikipedia
import pandas as pd
import requests
import time
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

load_dotenv()

class DataProcessor:
    """Complete data processing with multiple sources"""
    
    def __init__(self):
        self.embeddings = OllamaEmbeddings(model=os.getenv("EMBEDDING_MODEL"))
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.api_key = os.getenv("BRIGHTDATA_API_KEY")
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
    
    def scrape_wikipedia(self, keywords_file="keywords.xlsx"):
        """Scrape Wikipedia articles based on keywords"""
        print("🔍 Starting Wikipedia scraping...")
        
        if not os.path.exists(os.getenv("DATASET_STORAGE_FOLDER")):
            os.makedirs(os.getenv("DATASET_STORAGE_FOLDER"))
        
        keywords = pd.read_excel(keywords_file)
        all_content = []
        
        for index, row in keywords.iterrows():
            keyword = row['Keyword']
            pages = row['Pages']
            
            try:
                print(f"Searching for: {keyword}")
                search_results = wikipedia.search(keyword, results=pages)
                
                for result in search_results:
                    try:
                        page = wikipedia.page(result)
                        content = {
                            'title': page.title,
                            'url': page.url,
                            'content': page.content
                        }
                        all_content.append(content)
                        print(f"  - Scraped: {page.title}")
                    except Exception as e:
                        print(f"  - Error scraping {result}: {e}")
            except Exception as e:
                print(f"Error searching for {keyword}: {e}")
        
        output_file = os.path.join(os.getenv("DATASET_STORAGE_FOLDER"), "data.txt")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in all_content:
                f.write(f"TITLE: {item['title']}\n")
                f.write(f"URL: {item['url']}\n")
                f.write(f"CONTENT:\n{item['content']}\n")
                f.write("="*80 + "\n\n")
        
        print(f"✅ Scraping completed! Saved {len(all_content)} articles to {output_file}")
        return len(all_content)
    
    def brightdata_scrape(self, keywords, pages_per_keyword=2):
        """Bright Data web scraping"""
        if not self.api_key or self.api_key == "[YOUR API KEY HERE]":
            print("⚠️ Bright Data API key not configured")
            return 0
        
        print("🌐 Starting Bright Data scraping...")
        
        json_data = []
        for keyword in keywords:
            json_data.append({
                "keyword": keyword,
                "pages_load": str(pages_per_keyword)
            })
        
        params = {
            "dataset_id": "gd_lr9978962kkjr3nx49",
            "include_errors": "true",
            "type": "discover_new",
            "discover_by": "keyword",
        }
        
        try:
            response = requests.post(
                'https://api.brightdata.com/datasets/v3/trigger',
                params=params,
                headers=self.headers,
                json=json_data
            )
            
            result = response.json()
            snapshot_id = result["snapshot_id"]
            
            with open(os.getenv("SNAPSHOT_STORAGE_FILE"), "w") as f:
                f.write(snapshot_id)
            
            print(f"✅ Snapshot created: {snapshot_id}")
            
            # Wait for completion
            max_wait_time = 300
            for i in range(0, max_wait_time, 10):
                try:
                    status_response = requests.get(
                        f'https://api.brightdata.com/datasets/v3/progress/{snapshot_id}',
                        headers={'Authorization': f'Bearer {self.api_key}'}
                    )
                    status = status_response.json()['status']
                    print(f"📊 Status: {status}")
                    
                    if status == "ready":
                        break
                    print(f"⏱️ Waiting... ({i + 10}/{max_wait_time}s)")
                    time.sleep(10)
                except:
                    pass
            
            # Download data
            download_response = requests.get(
                f'https://api.brightdata.com/datasets/v3/snapshot/{snapshot_id}',
                headers=self.headers
            )
            
            data = download_response.json()
            documents = []
            
            for item in data:
                if 'content' in item and item['content'].strip():
                    doc = Document(
                        page_content=item['content'],
                        metadata={
                            'source': item.get('url', 'Unknown'),
                            'title': item.get('title', 'Untitled'),
                            'scrape_method': 'brightdata'
                        }
                    )
                    documents.append(doc)
            
            chunks = self.text_splitter.split_documents(documents)
            chunk_texts = [chunk.page_content for chunk in chunks]
            embeddings_list = self.embeddings.embed_documents(chunk_texts)
            
            processed_data = {
                "chunks": [
                    {
                        "content": chunk.page_content,
                        "metadata": chunk.metadata,
                        "embedding": embedding
                    }
                    for chunk, embedding in zip(chunks, embeddings_list)
                ]
            }
            
            output_file = os.path.join(os.getenv("DATASET_STORAGE_FOLDER"), "brightdata_chunks.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2)
            
            print(f"✅ Bright Data completed! {len(chunks)} chunks saved")
            return len(chunks)
            
        except Exception as e:
            print(f"❌ Bright Data error: {e}")
            return 0
    
    def process_documents(self, data_file=None):
        """Process scraped documents into chunks with embeddings"""
        if data_file is None:
            data_file = os.path.join(os.getenv("DATASET_STORAGE_FOLDER"), "data.txt")
        
        if not os.path.exists(data_file):
            print("❌ No data file found")
            return 0
        
        print("📚 Reading and processing documents...")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        documents = []
        sections = content.split('=' * 80)
        
        for section in sections:
            if section.strip():
                lines = section.strip().split('\n')
                if len(lines) >= 3:
                    title = lines[0].replace('TITLE: ', '')
                    url = lines[1].replace('URL: ', '')
                    content_text = '\n'.join(lines[2:]).replace('CONTENT:\n', '')
                    
                    doc = Document(
                        page_content=content_text,
                        metadata={"source": url, "title": title}
                    )
                    documents.append(doc)
        
        print(f"Created {len(documents)} documents")
        
        chunks = self.text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks")
        
        print("🧠 Creating embeddings...")
        embedding_texts = [chunk.page_content for chunk in chunks]
        embeddings_list = self.embeddings.embed_documents(embedding_texts)
        
        processed_data = {
            "chunks": [
                {
                    "content": chunk.page_content,
                    "metadata": chunk.metadata,
                    "embedding": embedding
                }
                for chunk, embedding in zip(chunks, embeddings_list)
            ]
        }
        
        output_file = os.path.join(os.getenv("DATASET_STORAGE_FOLDER"), "processed_chunks.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2)
        
        print(f"✅ Processing completed! {len(chunks)} chunks saved to {output_file}")
        return len(chunks)
    
    def get_optimal_keywords(self):
        """Get optimal keywords for diverse knowledge base"""
        return [
            "Python programming language complete guide",
            "LangChain framework tutorial",
            "React hooks and components",
            "OpenAI GPT models",
            "Machine learning algorithms",
            "JavaScript modern features ES6+",
            "TypeScript programming language",
            "Node.js backend development",
            "Docker containerization tutorial",
            "Kubernetes orchestration guide",
            "TensorFlow vs PyTorch comparison",
            "AWS cloud services guide",
            "Vue.js vs React comparison",
            "Deep learning neural networks",
            "Data science with Pandas",
            "Web3 blockchain development",
            "Quantum computing basics",
            "AI ethics and safety"
        ]
    
    def create_unified_data(self):
        """Create unified dataset from all sources"""
        print("🔗 Creating unified dataset...")
        
        all_chunks = []
        source_stats = {}
        
        # Load Wikipedia data
        wiki_file = os.path.join(os.getenv("DATASET_STORAGE_FOLDER"), "processed_chunks.json")
        if os.path.exists(wiki_file):
            with open(wiki_file, 'r', encoding='utf-8') as f:
                wiki_data = json.load(f)
            all_chunks.extend(wiki_data.get("chunks", []))
            source_stats['wikipedia'] = len(wiki_data.get("chunks", []))
        
        # Load Bright Data
        bd_file = os.path.join(os.getenv("DATASET_STORAGE_FOLDER"), "brightdata_chunks.json")
        if os.path.exists(bd_file):
            with open(bd_file, 'r', encoding='utf-8') as f:
                bd_data = json.load(f)
            all_chunks.extend(bd_data.get("chunks", []))
            source_stats['brightdata'] = len(bd_data.get("chunks", []))
        
        # Scrape additional Wikipedia topics
        print("🔍 Scraping additional topics...")
        additional_topics = [
            "Artificial intelligence", "Machine learning", "Deep learning",
            "JavaScript", "React (JavaScript library)", "Docker (software)",
            "Kubernetes", "Cloud computing", "Blockchain", "LangChain"
        ]
        
        additional_chunks = []
        for topic in additional_topics:
            try:
                search_results = wikipedia.search(topic, results=1)
                if search_results:
                    page = wikipedia.page(search_results[0])
                    doc = Document(
                        page_content=page.content[:2000],
                        metadata={
                            "source": page.url,
                            "title": page.title,
                            "scrape_method": "additional_wikipedia"
                        }
                    )
                    additional_chunks.append(doc)
            except:
                pass
        
        if additional_chunks:
            chunks = self.text_splitter.split_documents(additional_chunks)
            chunk_texts = [chunk.page_content for chunk in chunks]
            embeddings_list = self.embeddings.embed_documents(chunk_texts)
            
            for chunk, embedding in zip(chunks, embeddings_list):
                all_chunks.append({
                    "content": chunk.page_content,
                    "metadata": chunk.metadata,
                    "embedding": embedding
                })
            source_stats['additional_wikipedia'] = len(chunks)
        
        # Save unified data
        unified_data = {
            "chunks": all_chunks,
            "source_stats": source_stats,
            "total_chunks": len(all_chunks),
            "created_at": time.time()
        }
        
        output_file = os.path.join(os.getenv("DATASET_STORAGE_FOLDER"), "unified_chunks.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(unified_data, f, indent=2)
        
        print(f"✅ Unified dataset created: {len(all_chunks)} chunks")
        for source, count in source_stats.items():
            print(f"  {source}: {count} chunks")
        
        return len(all_chunks)
    
    def realtime_wikipedia_search(self, query, max_results=3):
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
                        'content': page.content[:2000],
                        'summary': page.summary
                    })
                except:
                    continue
                    
            return articles
        except Exception as e:
            print(f"Wikipedia search error: {e}")
            return []

def main():
    """Main function with options"""
    processor = DataProcessor()
    
    print("🚀 LOCAL RAG DATA PROCESSOR")
    print("=" * 50)
    print("1. Process Wikipedia data (existing)")
    print("2. Bright Data scraping (optimal)")
    print("3. Create unified dataset (recommended)")
    print("4. Complete pipeline (all)")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        processor.scrape_wikipedia()
        processor.process_documents()
    elif choice == "2":
        keywords = processor.get_optimal_keywords()
        processor.brightdata_scrape(keywords)
    elif choice == "3":
        processor.create_unified_data()
    elif choice == "4":
        processor.scrape_wikipedia()
        processor.process_documents()
        keywords = processor.get_optimal_keywords()
        processor.brightdata_scrape(keywords)
        processor.create_unified_data()
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()
