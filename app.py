from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import io
import base64
from datetime import datetime
import json
import re
from typing import List, Dict
import tempfile

# Document processing
import PyPDF2
from docx import Document
from PIL import Image

# Text-to-Speech
from gtts import gTTS

# Lightweight NLP
import random
from collections import Counter
import math

app = Flask(__name__)

CORS(app, resources={
    r"/api/*": {
        "origins": ["*"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

UPLOAD_FOLDER = tempfile.gettempdir()
AUDIO_FOLDER = tempfile.gettempdir()
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024


class DocumentProcessingAgent:
    """Lightweight document extraction"""
    
    def __init__(self):
        self.name = "DocumentProcessor"
    
    def extract_from_pdf(self, file_path: str) -> str:
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            raise Exception(f"PDF extraction error: {str(e)}")
        return self.clean_text(text)
    
    def extract_from_docx(self, file_path: str) -> str:
        try:
            doc = Document(file_path)
            text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        except Exception as e:
            raise Exception(f"DOCX extraction error: {str(e)}")
        return self.clean_text(text)
    
    def extract_from_txt(self, file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as file:
                text = file.read()
        return self.clean_text(text)
    
    def clean_text(self, text: str) -> str:
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        return text.strip()
    
    def process(self, file_path: str, file_type: str) -> str:
        if file_type == 'pdf':
            return self.extract_from_pdf(file_path)
        elif file_type in ['doc', 'docx']:
            return self.extract_from_docx(file_path)
        elif file_type == 'txt':
            return self.extract_from_txt(file_path)
        else:
            raise Exception(f"Unsupported file type: {file_type}")


class TextToSpeechAgent:
    """Lightweight TTS"""
    
    def __init__(self):
        self.name = "TextToSpeech"
    
    def generate_audio(self, text: str, output_path: str, speed: float = 1.0) -> str:
        try:
            if len(text) > 5000:
                text = text[:5000] + "... (Audio truncated due to length)"
            
            tts = gTTS(text=text, lang='en', tld='com.ng', slow=(speed < 0.8))
            tts.save(output_path)
            return output_path
        except Exception as e:
            raise Exception(f"TTS generation error: {str(e)}")


class QuestionGenerationAgent:
    """Lightweight question generation using TF-IDF and rule-based methods"""
    
    def __init__(self):
        self.name = "QuestionGenerator"
    
    def extract_sentences(self, text: str) -> List[str]:
        """Extract meaningful sentences"""
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 30]
        return sentences[:50]  # Limit for performance
    
    def calculate_tfidf_scores(self, sentences: List[str]) -> List[float]:
        """Simple TF-IDF scoring"""
        # Tokenize
        docs = [[w.lower() for w in re.findall(r'\b\w+\b', s)] for s in sentences]
        
        # Calculate IDF
        word_doc_count = Counter()
        for doc in docs:
            word_doc_count.update(set(doc))
        
        num_docs = len(docs)
        idf = {w: math.log(num_docs / count) for w, count in word_doc_count.items()}
        
        # Calculate TF-IDF scores for each sentence
        scores = []
        for doc in docs:
            word_counts = Counter(doc)
            tfidf_sum = sum(word_counts[w] * idf[w] for w in doc)
            scores.append(tfidf_sum / len(doc) if doc else 0)
        
        return scores
    
    def extract_key_sentences(self, text: str, num_sentences: int = 20) -> List[str]:
        """Extract important sentences using TF-IDF"""
        sentences = self.extract_sentences(text)
        if not sentences:
            return []
        
        scores = self.calculate_tfidf_scores(sentences)
        
        # Get top sentences
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        top_indices = [i for i, _ in indexed_scores[:num_sentences]]
        return [sentences[i] for i in sorted(top_indices)]
    
    def generate_mcq(self, text: str, num_questions: int = 5) -> List[Dict]:
        """Generate MCQ questions"""
        key_sentences = self.extract_key_sentences(text, num_questions * 2)
        questions = []
        
        templates = [
            "According to the text, what is stated about {}?",
            "Which of the following best describes {}?",
            "What can be inferred about {} from the passage?",
            "The text suggests that {} is:",
            "Which statement about {} is correct?",
        ]
        
        for i, sentence in enumerate(key_sentences[:num_questions]):
            words = sentence.split()
            key_phrase = ' '.join(words[2:6]) if len(words) > 8 else ' '.join(words[:4])
            
            question_text = templates[i % len(templates)].format(key_phrase)
            correct_answer = sentence[:100] + "..." if len(sentence) > 100 else sentence
            
            distractors = [
                "This information is not mentioned in the text",
                "The opposite of what is stated in the passage",
                "A related but incorrect interpretation"
            ]
            
            options = [correct_answer] + distractors[:3]
            random.shuffle(options)
            correct_index = options.index(correct_answer)
            
            questions.append({
                'id': f'mcq_{i+1}',
                'type': 'multiple_choice',
                'question': question_text,
                'options': options,
                'correct_answer': correct_index,
                'explanation': f"Based on: {sentence[:200]}..."
            })
        
        return questions
    
    def generate_essay(self, text: str, num_questions: int = 3) -> List[Dict]:
        """Generate essay questions"""
        key_sentences = self.extract_key_sentences(text, num_questions * 2)
        questions = []
        
        templates = [
            "Discuss the main points about {} in the text.",
            "Explain what the text reveals about {}.",
            "Analyze the information regarding {}.",
            "Describe the key aspects of {} from the document.",
        ]
        
        for i, sentence in enumerate(key_sentences[:num_questions]):
            words = sentence.split()
            key_phrase = ' '.join(words[:6]) if len(words) > 6 else sentence
            
            question_text = templates[i % len(templates)].format(key_phrase)
            
            questions.append({
                'id': f'essay_{i+1}',
                'type': 'essay',
                'question': question_text,
                'context': sentence,
                'suggested_points': [
                    "Introduction to the topic",
                    "Main points from the text",
                    "Supporting details",
                    "Conclusion"
                ]
            })
        
        return questions
    
    def generate_theory(self, text: str, num_questions: int = 4) -> List[Dict]:
        """Generate theory questions"""
        key_sentences = self.extract_key_sentences(text, num_questions * 2)
        questions = []
        
        templates = [
            "What does the text say about {}?",
            "Define {} as used in the text.",
            "Explain the significance of {}.",
            "How is {} described?",
        ]
        
        for i, sentence in enumerate(key_sentences[:num_questions]):
            words = sentence.split()
            key_phrase = ' '.join(words[:5]) if len(words) > 5 else sentence
            
            question_text = templates[i % len(templates)].format(key_phrase)
            
            questions.append({
                'id': f'theory_{i+1}',
                'type': 'theory',
                'question': question_text,
                'expected_answer': sentence[:300],
                'keywords': [w for w in key_phrase.split() if len(w) > 4][:5],
                'points': 5
            })
        
        return questions


class AnswerEvaluationAgent:
    """Lightweight answer evaluation"""
    
    def __init__(self):
        self.name = "AnswerEvaluator"
    
    def calculate_word_overlap(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity"""
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def evaluate_mcq(self, question: Dict, user_answer: int) -> Dict:
        is_correct = user_answer == question['correct_answer']
        
        return {
            'correct': is_correct,
            'user_answer': user_answer,
            'correct_answer': question['correct_answer'],
            'explanation': question.get('explanation', ''),
            'feedback': "Correct! Well done." if is_correct else f"Incorrect. The correct answer is option {question['correct_answer'] + 1}."
        }
    
    def evaluate_theory(self, question: Dict, user_answer: str) -> Dict:
        expected = question['expected_answer']
        keywords = question.get('keywords', [])
        
        # Word overlap similarity
        similarity = self.calculate_word_overlap(user_answer, expected)
        
        # Keyword check
        user_lower = user_answer.lower()
        keywords_found = sum(1 for kw in keywords if kw.lower() in user_lower)
        keyword_score = keywords_found / len(keywords) if keywords else 0
        
        overall_score = (similarity * 0.6 + keyword_score * 0.4)
        points_earned = int(overall_score * question.get('points', 5))
        
        if overall_score >= 0.8:
            feedback = "Excellent answer!"
        elif overall_score >= 0.6:
            feedback = "Good answer, but could be more comprehensive."
        elif overall_score >= 0.4:
            feedback = "Fair attempt. Review the expected answer."
        else:
            feedback = "Needs improvement. Please refer to the expected answer."
        
        return {
            'score': round(overall_score * 100, 2),
            'points_earned': points_earned,
            'total_points': question.get('points', 5),
            'keywords_found': keywords_found,
            'total_keywords': len(keywords),
            'feedback': feedback,
            'expected_answer': expected[:500]
        }
    
    def evaluate_essay(self, question: Dict, user_answer: str) -> Dict:
        context = question.get('context', '')
        word_count = len(user_answer.split())
        
        # Word overlap with context
        relevance = self.calculate_word_overlap(user_answer, context) if context else 0.5
        
        # Length score
        length_score = min(word_count / 150, 1.0)
        
        overall_score = (relevance * 0.6 + length_score * 0.4)
        
        feedback_parts = []
        if word_count < 50:
            feedback_parts.append("Too short. Aim for 100+ words.")
        elif word_count < 100:
            feedback_parts.append("Could be more detailed.")
        else:
            feedback_parts.append("Good length.")
        
        if relevance >= 0.5:
            feedback_parts.append("Relevant to the topic.")
        else:
            feedback_parts.append("Stay focused on the main topic.")
        
        return {
            'score': round(overall_score * 100, 2),
            'word_count': word_count,
            'relevance_score': round(relevance * 100, 2),
            'feedback': " ".join(feedback_parts)
        }


class RAGAgent:
    """Lightweight RAG using keyword matching"""
    
    def __init__(self):
        self.name = "RAGSystem"
        self.chunks = []
        self.full_text = ""
    
    def chunk_text(self, text: str, chunk_size: int = 400) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current = []
        current_len = 0
        
        for sentence in sentences:
            words = len(sentence.split())
            if current_len + words > chunk_size and current:
                chunks.append(' '.join(current))
                current = [sentence]
                current_len = words
            else:
                current.append(sentence)
                current_len += words
        
        if current:
            chunks.append(' '.join(current))
        
        return chunks
    
    def build_knowledge_base(self, text: str):
        self.full_text = text
        self.chunks = self.chunk_text(text)
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 3) -> List[str]:
        """Keyword-based retrieval"""
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        
        scores = []
        for chunk in self.chunks:
            chunk_words = set(re.findall(r'\b\w+\b', chunk.lower()))
            overlap = len(query_words.intersection(chunk_words))
            scores.append(overlap)
        
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.chunks[i] for i in top_indices if scores[i] > 0]
    
    def answer_question(self, question: str) -> Dict:
        relevant = self.retrieve_relevant_chunks(question, 3)
        
        if not relevant:
            return {
                'answer': "I don't have enough information to answer this question.",
                'confidence': 0.0,
                'sources': []
            }
        
        answer = f"Based on the document: {relevant[0][:400]}..."
        
        return {
            'answer': answer,
            'confidence': 0.7,
            'sources': [{'text': chunk[:200] + '...'} for chunk in relevant]
        }
    
    def get_summary(self, max_sentences: int = 5) -> str:
        if not self.chunks:
            return ""
        
        num_chunks = min(max_sentences, len(self.chunks))
        step = max(1, len(self.chunks) // num_chunks)
        
        summary_chunks = [self.chunks[i * step] for i in range(num_chunks) if i * step < len(self.chunks)]
        return " ".join(summary_chunks)[:1000]


class OrchestratorAgent:
    """Main coordinator"""
    
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
        text = self.doc_agent.process(file_path, file_type)
        
        if len(text) < 100:
            raise Exception("Document text too short")
        
        self.rag_agent.build_knowledge_base(text)
        
        self.current_document = {
            'text': text,
            'processed_at': datetime.now().isoformat(),
            'word_count': len(text.split()),
            'summary': self.rag_agent.get_summary()
        }
        
        return self.current_document
    
    def generate_audio(self, text: str, speed: float = 1.0) -> str:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(AUDIO_FOLDER, f'audio_{timestamp}.mp3')
        return self.tts_agent.generate_audio(text, output_path, speed)
    
    def generate_all_questions(self, question_type: str = 'all', num_questions: int = 5) -> Dict:
        text = self.current_document['text']
        questions = {}
        
        if question_type in ['all', 'mcq']:
            questions['mcq'] = self.question_agent.generate_mcq(text, num_questions)
        
        if question_type in ['all', 'essay']:
            questions['essay'] = self.question_agent.generate_essay(text, max(3, num_questions // 2))
        
        if question_type in ['all', 'theory']:
            questions['theory'] = self.question_agent.generate_theory(text, num_questions)
        
        self.current_questions = questions
        return questions
    
    def evaluate_answer(self, question_id: str, question_type: str, user_answer) -> Dict:
        question = None
        for qlist in self.current_questions.values():
            for q in qlist:
                if q['id'] == question_id:
                    question = q
                    break
        
        if not question:
            return {'error': 'Question not found'}
        
        if question_type == 'multiple_choice':
            return self.eval_agent.evaluate_mcq(question, user_answer)
        elif question_type == 'theory':
            return self.eval_agent.evaluate_theory(question, user_answer)
        elif question_type == 'essay':
            return self.eval_agent.evaluate_essay(question, user_answer)
        
        return {'error': 'Unknown question type'}


orchestrator = OrchestratorAgent()


# API Endpoints
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'version': '2.0-lite',
        'features': ['PDF/DOCX/TXT processing', 'Nigerian TTS', 'Question generation', 'Answer evaluation', 'RAG Q&A']
    })


@app.route('/api/upload', methods=['POST'])
def upload_document():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_extension = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else 'txt'
    filename = f"{timestamp}_{file.filename}"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)
    
    try:
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
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass


@app.route('/api/generate-audio', methods=['POST'])
def generate_audio():
    data = request.json
    text = data.get('text', '')
    speed = float(data.get('speed', 1.0))
    use_summary = data.get('use_summary', False)
    
    if not text and not use_summary:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        if use_summary and orchestrator.current_document:
            text = orchestrator.current_document.get('summary', text)
        
        if not text:
            return jsonify({'error': 'No text available'}), 400
        
        audio_path = orchestrator.generate_audio(text, speed)
        
        with open(audio_path, 'rb') as audio_file:
            audio_data = base64.b64encode(audio_file.read()).decode('utf-8')
        
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
    data = request.json
    question_type = data.get('type', 'all')
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
    data = request.json
    question_id = data.get('question_id')
    question_type = data.get('question_type')
    user_answer = data.get('answer')
    
    if not all([question_id, question_type, user_answer is not None]):
        return jsonify({'error': 'Missing required fields'}), 400
    
    try:
        evaluation = orchestrator.evaluate_answer(question_id, question_type, user_answer)
        return jsonify({'success': True, 'evaluation': evaluation})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ask', methods=['POST'])
def ask_question():
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
    if not orchestrator.current_document:
        return jsonify({'error': 'No document processed yet'}), 400
    
    return jsonify({
        'success': True,
        'summary': orchestrator.current_document.get('summary', ''),
        'word_count': orchestrator.current_document.get('word_count', 0)
    })


@app.route('/api/batch-evaluate', methods=['POST'])
def batch_evaluate():
    data = request.json
    answers = data.get('answers', [])
    
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
            
            results.append({'question_id': answer_data['question_id'], 'evaluation': evaluation})
            
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


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'message': 'AI Study Assistant API - Lite Version',
        'version': '2.0-lite',
        'status': 'active'
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
