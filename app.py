from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import io
import base64
from datetime import datetime
import json
import re
from typing import List, Dict, Tuple
import tempfile

# Document processing
import PyPDF2
from docx import Document
import pytesseract
from PIL import Image

# AI and NLP
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Text-to-Speech
from gtts import gTTS

# Environment setup
app = Flask(__name__)

# CORS configuration for Vercel frontend
CORS(app, resources={
    r"/api/*": {
        "origins": ["*"],  # Update with your Vercel domain in production
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Configuration
UPLOAD_FOLDER = tempfile.gettempdir()
AUDIO_FOLDER = tempfile.gettempdir()

app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max


class DocumentProcessingAgent:
    """Enhanced agent for extracting and preprocessing text"""
    
    def __init__(self):
        self.name = "DocumentProcessor"
    
    def extract_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF with better formatting"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
        except Exception as e:
            raise Exception(f"PDF extraction error: {str(e)}")
        return self.clean_text(text)
    
    def extract_from_docx(self, file_path: str) -> str:
        """Extract text from Word document with structure"""
        try:
            doc = Document(file_path)
            text_parts = []
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    # Identify headings
                    if paragraph.style.name.startswith('Heading'):
                        text_parts.append(f"\n## {paragraph.text}\n")
                    else:
                        text_parts.append(paragraph.text)
            
            text = "\n".join(text_parts)
        except Exception as e:
            raise Exception(f"DOCX extraction error: {str(e)}")
        return self.clean_text(text)
    
    def extract_from_txt(self, file_path: str) -> str:
        """Extract text from plain text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as file:
                text = file.read()
        except Exception as e:
            raise Exception(f"TXT extraction error: {str(e)}")
        return self.clean_text(text)
    
    def extract_from_image(self, file_path: str) -> str:
        """Extract text from image using OCR"""
        try:
            image = Image.open(file_path)
            # Enhance image for better OCR
            image = image.convert('L')  # Grayscale
            text = pytesseract.image_to_string(image, config='--psm 6')
        except Exception as e:
            raise Exception(f"OCR extraction error: {str(e)}")
        return self.clean_text(text)
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        text = text.strip()
        return text
    
    def process(self, file_path: str, file_type: str) -> str:
        """Main processing method"""
        print(f"[{self.name}] Processing {file_type} document...")
        
        if file_type == 'pdf':
            return self.extract_from_pdf(file_path)
        elif file_type in ['doc', 'docx']:
            return self.extract_from_docx(file_path)
        elif file_type == 'txt':
            return self.extract_from_txt(file_path)
        elif file_type in ['jpg', 'jpeg', 'png', 'bmp', 'gif']:
            return self.extract_from_image(file_path)
        else:
            raise Exception(f"Unsupported file type: {file_type}")


class TextToSpeechAgent:
    """Enhanced TTS agent with Nigerian English support"""
    
    def __init__(self):
        self.name = "TextToSpeech"
    
    def split_text_for_tts(self, text: str, max_length: int = 4000) -> List[str]:
        """Split long text into manageable chunks for TTS"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < max_length:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def generate_audio(self, text: str, output_path: str, speed: float = 1.0) -> str:
        """Generate speech using gTTS with Nigerian English"""
        print(f"[{self.name}] Generating audio...")
        
        try:
            # Limit text length for gTTS
            if len(text) > 5000:
                text = text[:5000] + "... (Audio truncated due to length)"
            
            # Generate with Nigerian English accent
            tts = gTTS(
                text=text,
                lang='en',
                tld='com.ng',  # Nigerian English
                slow=(speed < 0.8)
            )
            tts.save(output_path)
            return output_path
            
        except Exception as e:
            raise Exception(f"TTS generation error: {str(e)}")


class QuestionGenerationAgent:
    """Advanced agent for generating study questions"""
    
    def __init__(self):
        self.name = "QuestionGenerator"
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def extract_key_concepts(self, text: str, num_concepts: int = 20) -> List[str]:
        """Extract key concepts from text"""
        # Split into sentences
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 20]
        
        if not sentences:
            return []
        
        # Get embeddings
        embeddings = self.model.encode(sentences)
        
        # Simple diversity sampling
        selected_indices = []
        selected_indices.append(0)  # Start with first sentence
        
        while len(selected_indices) < min(num_concepts, len(sentences)):
            # Find sentence most different from already selected
            max_min_distance = -1
            best_idx = -1
            
            for i in range(len(sentences)):
                if i in selected_indices:
                    continue
                
                # Calculate minimum distance to selected sentences
                min_distance = min([
                    cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                    for j in selected_indices
                ])
                
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_idx = i
            
            if best_idx != -1:
                selected_indices.append(best_idx)
            else:
                break
        
        return [sentences[i] for i in selected_indices]
    
    def generate_mcq(self, text: str, num_questions: int = 5) -> List[Dict]:
        """Generate multiple choice questions"""
        concepts = self.extract_key_concepts(text, num_questions * 2)
        questions = []
        
        templates = [
            "According to the text, what is stated about {concept}?",
            "Which of the following best describes {concept}?",
            "What can be inferred about {concept} from the passage?",
            "The text suggests that {concept} is:",
            "Which statement about {concept} is correct?",
        ]
        
        for i, concept in enumerate(concepts[:num_questions]):
            # Extract key phrase
            words = concept.split()
            if len(words) > 8:
                key_phrase = ' '.join(words[2:6])
            else:
                key_phrase = ' '.join(words[:4])
            
            template = templates[i % len(templates)]
            question_text = template.format(concept=key_phrase)
            
            # Create answer from the concept
            correct_answer = concept[:100] + "..." if len(concept) > 100 else concept
            
            # Generate plausible distractors
            distractors = [
                "This information is not mentioned in the text",
                "The opposite of what is stated in the passage",
                "A related but incorrect interpretation"
            ]
            
            options = [correct_answer] + distractors[:3]
            
            # Shuffle but remember correct position
            import random
            correct_index = 0
            combined = list(enumerate(options))
            random.shuffle(combined)
            
            shuffled_options = [opt for _, opt in combined]
            correct_answer_index = [i for i, (orig_idx, _) in enumerate(combined) if orig_idx == 0][0]
            
            questions.append({
                'id': f'mcq_{i+1}',
                'type': 'multiple_choice',
                'question': question_text,
                'options': shuffled_options,
                'correct_answer': correct_answer_index,
                'explanation': f"This is based on: {concept[:200]}..."
            })
        
        return questions
    
    def generate_essay(self, text: str, num_questions: int = 3) -> List[Dict]:
        """Generate essay questions"""
        concepts = self.extract_key_concepts(text, num_questions * 2)
        questions = []
        
        templates = [
            "Discuss the main points presented about {concept} in the text.",
            "Explain in detail what the text reveals about {concept}.",
            "Analyze the information provided regarding {concept}.",
            "Describe the key aspects of {concept} as presented in the document.",
            "Critically examine what the text says about {concept}.",
        ]
        
        for i, concept in enumerate(concepts[:num_questions]):
            words = concept.split()
            key_phrase = ' '.join(words[:6]) if len(words) > 6 else concept
            
            template = templates[i % len(templates)]
            question_text = template.format(concept=key_phrase)
            
            questions.append({
                'id': f'essay_{i+1}',
                'type': 'essay',
                'question': question_text,
                'context': concept,
                'suggested_points': [
                    "Introduction to the topic",
                    "Main points from the text",
                    "Supporting details and examples",
                    "Conclusion"
                ],
                'marking_guide': f"Look for: understanding of {key_phrase}, clarity of explanation, use of evidence from text"
            })
        
        return questions
    
    def generate_theory(self, text: str, num_questions: int = 4) -> List[Dict]:
        """Generate theory/short answer questions"""
        concepts = self.extract_key_concepts(text, num_questions * 2)
        questions = []
        
        templates = [
            "What does the text say about {concept}?",
            "Define {concept} as used in the text.",
            "Explain the significance of {concept}.",
            "List the key features of {concept} mentioned in the document.",
            "How is {concept} described in the text?",
        ]
        
        for i, concept in enumerate(concepts[:num_questions]):
            words = concept.split()
            key_phrase = ' '.join(words[:5]) if len(words) > 5 else concept
            
            template = templates[i % len(templates)]
            question_text = template.format(concept=key_phrase)
            
            questions.append({
                'id': f'theory_{i+1}',
                'type': 'theory',
                'question': question_text,
                'expected_answer': concept[:300],
                'keywords': key_phrase.split()[:5],
                'points': 5
            })
        
        return questions


class AnswerEvaluationAgent:
    """Agent for evaluating student answers"""
    
    def __init__(self):
        self.name = "AnswerEvaluator"
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def evaluate_mcq(self, question: Dict, user_answer: int) -> Dict:
        """Evaluate multiple choice answer"""
        is_correct = user_answer == question['correct_answer']
        
        return {
            'correct': is_correct,
            'user_answer': user_answer,
            'correct_answer': question['correct_answer'],
            'explanation': question.get('explanation', ''),
            'feedback': "Correct! Well done." if is_correct else f"Incorrect. The correct answer is option {question['correct_answer'] + 1}."
        }
    
    def evaluate_theory(self, question: Dict, user_answer: str) -> Dict:
        """Evaluate theory/short answer"""
        expected = question['expected_answer']
        keywords = question.get('keywords', [])
        
        # Calculate similarity
        user_embedding = self.model.encode([user_answer])
        expected_embedding = self.model.encode([expected])
        similarity = cosine_similarity(user_embedding, expected_embedding)[0][0]
        
        # Check for keywords
        user_lower = user_answer.lower()
        keywords_found = sum(1 for kw in keywords if kw.lower() in user_lower)
        keyword_score = keywords_found / len(keywords) if keywords else 0
        
        # Combined score
        overall_score = (similarity * 0.6 + keyword_score * 0.4)
        points_earned = int(overall_score * question.get('points', 5))
        
        feedback = ""
        if overall_score >= 0.8:
            feedback = "Excellent answer! You've captured the key points well."
        elif overall_score >= 0.6:
            feedback = "Good answer, but could be more comprehensive."
        elif overall_score >= 0.4:
            feedback = "Fair attempt. Review the expected answer for improvements."
        else:
            feedback = "Needs improvement. Please refer to the expected answer."
        
        return {
            'score': round(overall_score * 100, 2),
            'points_earned': points_earned,
            'total_points': question.get('points', 5),
            'keywords_found': keywords_found,
            'total_keywords': len(keywords),
            'feedback': feedback,
            'expected_answer': expected[:500],
            'similarity': round(similarity, 3)
        }
    
    def evaluate_essay(self, question: Dict, user_answer: str) -> Dict:
        """Evaluate essay answer"""
        context = question.get('context', '')
        
        # Basic metrics
        word_count = len(user_answer.split())
        
        # Semantic similarity
        if context:
            user_embedding = self.model.encode([user_answer])
            context_embedding = self.model.encode([context])
            relevance = cosine_similarity(user_embedding, context_embedding)[0][0]
        else:
            relevance = 0.5
        
        # Length score
        length_score = min(word_count / 150, 1.0)  # Optimal around 150+ words
        
        # Combined evaluation
        overall_score = (relevance * 0.6 + length_score * 0.4)
        
        feedback_parts = []
        
        if word_count < 50:
            feedback_parts.append("Your essay is too short. Aim for at least 100-150 words.")
        elif word_count < 100:
            feedback_parts.append("Your essay could be more detailed.")
        else:
            feedback_parts.append("Good length and detail.")
        
        if relevance >= 0.7:
            feedback_parts.append("Your answer is highly relevant to the topic.")
        elif relevance >= 0.5:
            feedback_parts.append("Your answer is somewhat relevant but could focus more on the key points.")
        else:
            feedback_parts.append("Try to stay more focused on the main topic.")
        
        return {
            'score': round(overall_score * 100, 2),
            'word_count': word_count,
            'relevance_score': round(relevance * 100, 2),
            'feedback': " ".join(feedback_parts),
            'suggested_points': question.get('suggested_points', []),
            'marking_guide': question.get('marking_guide', '')
        }


class RAGAgent:
    """Enhanced RAG system for intelligent Q&A"""
    
    def __init__(self):
        self.name = "RAGSystem"
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chunks = []
        self.embeddings = None
        self.full_text = ""
        self.metadata = {}
    
    def chunk_text(self, text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_words = sentence.split()
            sentence_length = len(sentence_words)
            
            if current_length + sentence_length > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                # Keep overlap
                overlap_words = sum([len(s.split()) for s in current_chunk[-2:]], 0)
                if overlap_words < overlap:
                    current_chunk = current_chunk[-2:]
                else:
                    current_chunk = current_chunk[-1:]
                current_length = sum([len(s.split()) for s in current_chunk], 0)
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def build_knowledge_base(self, text: str):
        """Build vector database with metadata"""
        print(f"[{self.name}] Building knowledge base...")
        
        self.full_text = text
        self.chunks = self.chunk_text(text)
        
        # Create embeddings
        self.embeddings = self.model.encode(self.chunks, show_progress_bar=False)
        
        # Extract metadata
        self.metadata = {
            'total_words': len(text.split()),
            'total_chunks': len(self.chunks),
            'avg_chunk_length': np.mean([len(c.split()) for c in self.chunks])
        }
        
        print(f"[{self.name}] Knowledge base: {len(self.chunks)} chunks")
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Retrieve most relevant chunks with scores"""
        if not self.embeddings.size:
            return []
        
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = [(self.chunks[i], similarities[i]) for i in top_indices]
        return results
    
    def answer_question(self, question: str) -> Dict:
        """Intelligent Q&A with context"""
        print(f"[{self.name}] Answering: {question[:50]}...")
        
        relevant = self.retrieve_relevant_chunks(question, top_k=3)
        
        if not relevant:
            return {
                'answer': "I don't have enough information to answer this question from the document.",
                'confidence': 0.0,
                'sources': []
            }
        
        # Combine top contexts
        combined_context = " ".join([chunk for chunk, _ in relevant[:2]])
        confidence = float(relevant[0][1])
        
        # Generate answer
        if confidence > 0.5:
            answer = f"Based on the document: {combined_context[:400]}..."
        else:
            answer = f"The document mentions: {combined_context[:300]}... However, this may not fully answer your question."
        
        return {
            'answer': answer,
            'confidence': round(confidence, 3),
            'sources': [{'text': chunk[:200] + '...', 'relevance': round(float(score), 3)} 
                       for chunk, score in relevant]
        }
    
    def get_summary(self, max_sentences: int = 5) -> str:
        """Generate document summary"""
        if not self.chunks:
            return ""
        
        # Select diverse chunks
        num_chunks = min(max_sentences, len(self.chunks))
        step = len(self.chunks) // num_chunks
        
        summary_chunks = [self.chunks[i * step] for i in range(num_chunks)]
        summary = " ".join(summary_chunks)
        
        return summary[:1000]


class OrchestratorAgent:
    """Main orchestrator coordinating all agents"""
    
    def __init__(self):
        self.name = "Orchestrator"
        self.doc_agent = DocumentProcessingAgent()
        self.tts_agent = TextToSpeechAgent()
        self.rag_agent = RAGAgent()
        self.question_agent = QuestionGenerationAgent()
        self.eval_agent = AnswerEvaluationAgent()
        self.current_document = None
        self.current_questions = {}
    
    def process_document(self, file_path: str, file_type: str) -> Dict:
        """Complete document processing pipeline"""
        print(f"[{self.name}] Processing document...")
        
        # Extract text
        text = self.doc_agent.process(file_path, file_type)
        
        if len(text) < 100:
            raise Exception("Document text too short or extraction failed")
        
        # Build knowledge base
        self.rag_agent.build_knowledge_base(text)
        
        # Store document info
        self.current_document = {
            'text': text,
            'file_path': file_path,
            'processed_at': datetime.now().isoformat(),
            'word_count': len(text.split()),
            'summary': self.rag_agent.get_summary()
        }
        
        return self.current_document
    
    def generate_audio(self, text: str, speed: float = 1.0) -> str:
        """Generate audio file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(AUDIO_FOLDER, f'audio_{timestamp}.mp3')
        
        return self.tts_agent.generate_audio(text, output_path, speed)
    
    def generate_all_questions(self, question_type: str = 'all', num_questions: int = 5) -> Dict:
        """Generate questions of specified type"""
        text = self.current_document['text']
        
        questions = {}
        
        if question_type in ['all', 'mcq']:
            questions['mcq'] = self.question_agent.generate_mcq(text, num_questions)
        
        if question_type in ['all', 'essay']:
            questions['essay'] = self.question_agent.generate_essay(text, max(3, num_questions // 2))
        
        if question_type in ['all', 'theory']:
            questions['theory'] = self.question_agent.generate_theory(text, num_questions)
        
        # Store for later evaluation
        self.current_questions = questions
        
        return questions
    
    def evaluate_answer(self, question_id: str, question_type: str, user_answer) -> Dict:
        """Evaluate user's answer"""
        # Find question
        question = None
        for qtype, qlist in self.current_questions.items():
            for q in qlist:
                if q['id'] == question_id:
                    question = q
                    break
        
        if not question:
            return {'error': 'Question not found'}
        
        # Evaluate based on type
        if question_type == 'multiple_choice':
            return self.eval_agent.evaluate_mcq(question, user_answer)
        elif question_type == 'theory':
            return self.eval_agent.evaluate_theory(question, user_answer)
        elif question_type == 'essay':
            return self.eval_agent.evaluate_essay(question, user_answer)
        
        return {'error': 'Unknown question type'}


# Initialize orchestrator
orchestrator = OrchestratorAgent()


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'version': '2.0',
        'agents': {
            'document_processor': 'active',
            'tts_engine': 'active',
            'rag_system': 'active',
            'question_generator': 'active',
            'answer_evaluator': 'active',
            'orchestrator': 'active'
        },
        'features': [
            'PDF/DOCX/TXT/Image processing',
            'Nigerian English TTS',
            'MCQ/Essay/Theory questions',
            'Automated answer evaluation',
            'RAG-based Q&A'
        ]
    })


@app.route('/api/upload', methods=['POST'])
def upload_document():
    """Upload and process document"""
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_extension = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else 'txt'
    filename = f"{timestamp}_{file.filename}"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)
    
    try:
        # Process document
        doc_info = orchestrator.process_document(file_path, file_extension)
        
        return jsonify({
            'success': True,
            'document_id': timestamp,
            'text_preview': doc_info['text'][:500] + '...',
            'word_count': doc_info['word_count'],
            'summary': doc_info['summary'],
            'processed_at': doc_info['processed_at']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up uploaded file
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass


@app.route('/api/generate-audio', methods=['POST'])
def generate_audio():
    """Generate Nigerian English audio"""
    
    data = request.json
    text = data.get('text', '')
    speed = float(data.get('speed', 1.0))
    use_summary = data.get('use_summary', False)
    
    if not text and not use_summary:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        # Use summary if requested
        if use_summary and orchestrator.current_document:
            text = orchestrator.current_document.get('summary', text)
        
        if not text:
            return jsonify({'error': 'No text available'}), 400
        
        # Generate audio
        audio_path = orchestrator.generate_audio(text, speed)
        
        # Read audio file and encode to base64
        with open(audio_path, 'rb') as audio_file:
            audio_data = base64.b64encode(audio_file.read()).decode('utf-8')
        
        # Clean up
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        return jsonify({
            'success': True,
            'audio_data': audio_data,
            'audio_format': 'mp3',
            'text_length': len(text)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate-questions', methods=['POST'])
def generate_questions():
    """Generate study questions"""
    
    data = request.json
    question_type = data.get('type', 'all')  # all, mcq, essay, theory
    num_questions = int(data.get('num_questions', 5))
    
    if not orchestrator.current_document:
        return jsonify({'error': 'No document processed yet'}), 400
    
    try:
        questions = orchestrator.generate_all_questions(question_type, num_questions)
        
        return jsonify({
            'success': True,
            'questions': questions,
            'total_questions': sum(len(q) for q in questions.values())
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/evaluate-answer', methods=['POST'])
def evaluate_answer():
    """Evaluate student's answer"""
    
    data = request.json
    question_id = data.get('question_id')
    question_type = data.get('question_type')
    user_answer = data.get('answer')
    
    if not all([question_id, question_type, user_answer is not None]):
        return jsonify({'error': 'Missing required fields'}), 400
    
    try:
        evaluation = orchestrator.evaluate_answer(question_id, question_type, user_answer)
        
        return jsonify({
            'success': True,
            'evaluation': evaluation
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ask', methods=['POST'])
def ask_question():
    """Ask question about document (RAG)"""
    
    data = request.json
    question = data.get('question', '')
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    if not orchestrator.current_document:
        return jsonify({'error': 'No document processed yet'}), 400
    
    try:
        answer = orchestrator.rag_agent.answer_question(question)
        
        return jsonify({
            'success': True,
            'question': question,
            'answer': answer['answer'],
            'confidence': answer['confidence'],
            'sources': answer['sources']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/get-summary', methods=['GET'])
def get_summary():
    """Get document summary"""
    
    if not orchestrator.current_document:
        return jsonify({'error': 'No document processed yet'}), 400
    
    try:
        return jsonify({
            'success': True,
            'summary': orchestrator.current_document.get('summary', ''),
            'word_count': orchestrator.current_document.get('word_count', 0)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/explain-concept', methods=['POST'])
def explain_concept():
    """Explain a specific concept from the document"""
    
    data = request.json
    concept = data.get('concept', '')
    
    if not concept:
        return jsonify({'error': 'No concept provided'}), 400
    
    if not orchestrator.current_document:
        return jsonify({'error': 'No document processed yet'}), 400
    
    try:
        # Use RAG to find relevant information
        explanation = orchestrator.rag_agent.answer_question(
            f"Explain {concept} in detail based on the document"
        )
        
        return jsonify({
            'success': True,
            'concept': concept,
            'explanation': explanation['answer'],
            'confidence': explanation['confidence'],
            'related_sections': explanation['sources']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/batch-evaluate', methods=['POST'])
def batch_evaluate():
    """Evaluate multiple answers at once"""
    
    data = request.json
    answers = data.get('answers', [])  # List of {question_id, question_type, answer}
    
    if not answers:
        return jsonify({'error': 'No answers provided'}), 400
    
    try:
        results = []
        total_score = 0
        max_score = 0
        
        for answer_data in answers:
            evaluation = orchestrator.evaluate_answer(
                answer_data['question_id'],
                answer_data['question_type'],
                answer_data['answer']
            )
            
            results.append({
                'question_id': answer_data['question_id'],
                'evaluation': evaluation
            })
            
            # Calculate scores
            if 'score' in evaluation:
                total_score += evaluation['score']
                max_score += 100
            elif 'correct' in evaluation:
                total_score += 100 if evaluation['correct'] else 0
                max_score += 100
        
        percentage = (total_score / max_score * 100) if max_score > 0 else 0
        
        return jsonify({
            'success': True,
            'results': results,
            'summary': {
                'total_questions': len(answers),
                'total_score': round(total_score, 2),
                'max_score': max_score,
                'percentage': round(percentage, 2)
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/study-tips', methods=['GET'])
def get_study_tips():
    """Get personalized study tips based on document"""
    
    if not orchestrator.current_document:
        return jsonify({'error': 'No document processed yet'}), 400
    
    try:
        word_count = orchestrator.current_document.get('word_count', 0)
        
        # Estimate reading time (average 200 words per minute)
        reading_time = max(1, word_count // 200)
        
        # Generate study tips
        tips = [
            f"📚 This document has approximately {word_count} words and will take about {reading_time} minutes to read.",
            "🎧 Listen to the audio version while reading to improve retention.",
            "📝 Start with multiple choice questions to test basic understanding.",
            "✍️ Move to theory questions to reinforce key concepts.",
            "📖 Try essay questions to demonstrate comprehensive understanding.",
            "🔄 Review incorrect answers and their explanations carefully.",
            "⏰ Take breaks every 25-30 minutes to maintain focus.",
            "🎯 Use the Q&A feature to clarify confusing concepts.",
        ]
        
        return jsonify({
            'success': True,
            'tips': tips,
            'estimated_study_time': f"{reading_time + 15}-{reading_time + 30} minutes",
            'document_stats': {
                'word_count': word_count,
                'reading_time_minutes': reading_time
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Error handlers
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large. Maximum size is 50MB'}), 413


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


# Health check for Render
@app.route('/', methods=['GET'])
def index():
    """Root endpoint"""
    return jsonify({
        'message': 'AI Study Assistant API',
        'version': '2.0',
        'status': 'active',
        'endpoints': {
            'health': '/api/health',
            'upload': '/api/upload',
            'generate_audio': '/api/generate-audio',
            'generate_questions': '/api/generate-questions',
            'evaluate_answer': '/api/evaluate-answer',
            'batch_evaluate': '/api/batch-evaluate',
            'ask': '/api/ask',
            'summary': '/api/get-summary',
            'explain': '/api/explain-concept',
            'study_tips': '/api/study-tips'
        }
    })


if __name__ == '__main__':
    print("=" * 70)
    print("Study_Buddy - Enhanced Version 2.0")
    print("=" * 70)
    print("\n🤖 Active AI Agents:")
    print("  1. DocumentProcessingAgent - Multi-format extraction")
    print("  2. TextToSpeechAgent - Nigerian English audio")
    print("  3. QuestionGenerationAgent - MCQ/Essay/Theory generation")
    print("  4. AnswerEvaluationAgent - Intelligent grading")
    print("  5. RAGAgent - Smart Q&A system")
    print("  6. OrchestratorAgent - System coordinator")
    print("\n✨ Key Features:")
    print("  • PDF, DOCX, TXT, Image support")
    print("  • Nigerian English text-to-speech")
    print("  • Automated question generation (MCQ, Essay, Theory)")
    print("  • AI-powered answer evaluation")
    print("  • Intelligent document Q&A")
    print("  • Study tips and summaries")
    print("\n🚀 Starting server...")
    print("=" * 70)
    
    # Use environment port for Render deployment
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
