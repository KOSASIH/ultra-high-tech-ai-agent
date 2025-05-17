import re
import random
import logging
from typing import Dict, List, Any, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("NLPEngine")

class NLPEngine:
    """Natural Language Processing engine for the autonomous agent."""
    
    def __init__(self):
        # Initialize intent patterns
        self.intent_patterns = {
            "greeting": [
                r"hello|hi|hey|greetings|howdy",
                r"good\s+(morning|afternoon|evening)"
            ],
            "farewell": [
                r"bye|goodbye|see\s+you|farewell|later"
            ],
            "gratitude": [
                r"thanks|thank\s+you|appreciate"
            ],
            "weather": [
                r"weather|temperature|forecast|rain|sunny|cloudy"
            ],
            "time": [
                r"time|clock|hour"
            ],
            "date": [
                r"date|day|month|year|today"
            ],
            "help": [
                r"help|assist|support|guide"
            ],
            "about": [
                r"about\s+you|who\s+are\s+you|what\s+are\s+you|tell\s+me\s+about\s+yourself"
            ],
            "capabilities": [
                r"what\s+can\s+you\s+do|capabilities|features|functions"
            ],
            "joke": [
                r"joke|funny|humor|make\s+me\s+laugh"
            ],
            "fact": [
                r"fact|trivia|interesting"
            ],
            "math": [
                r"calculate|compute|math|sum|add|subtract|multiply|divide"
            ],
            "search": [
                r"search|find|look\s+up|google"
            ],
            "translate": [
                r"translate|translation|language"
            ],
            "reminder": [
                r"remind|reminder|remember"
            ],
            "news": [
                r"news|headlines|current\s+events"
            ],
            "music": [
                r"music|song|playlist|artist|album"
            ],
            "movie": [
                r"movie|film|watch|cinema|theater"
            ],
            "restaurant": [
                r"restaurant|food|eat|dining|dinner|lunch|breakfast"
            ],
            "travel": [
                r"travel|trip|vacation|flight|hotel"
            ],
            "shopping": [
                r"shop|buy|purchase|order|amazon"
            ],
            "sports": [
                r"sports|game|match|team|score|player"
            ],
            "stocks": [
                r"stocks|market|investment|shares|portfolio"
            ],
            "health": [
                r"health|medical|doctor|symptom|disease|condition"
            ],
            "definition": [
                r"define|definition|meaning|what\s+is|what\s+are"
            ],
            "learn": [
                r"learn|study|training|improve|better"
            ],
            "reset": [
                r"reset|restart|clear|start\s+over"
            ]
        }
        
        # Entity extraction patterns
        self.entity_patterns = {
            "location": [
                r"in\s+([A-Za-z\s]+)",
                r"at\s+([A-Za-z\s]+)",
                r"for\s+([A-Za-z\s]+)"
            ],
            "datetime": [
                r"at\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)",
                r"on\s+([A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?)",
                r"in\s+(\d+)\s+(minute|hour|day|week|month|year)s?"
            ],
            "number": [
                r"(\d+(?:\.\d+)?)"
            ],
            "person": [
                r"(?:about|for|to)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"
            ]
        }
        
        # Sentiment analysis keywords
        self.positive_words = [
            "good", "great", "excellent", "amazing", "wonderful", "fantastic",
            "terrific", "outstanding", "superb", "brilliant", "awesome",
            "fabulous", "incredible", "marvelous", "perfect", "pleasant",
            "delightful", "lovely", "nice", "happy", "glad", "positive",
            "beautiful", "impressive", "exceptional", "remarkable"
        ]
        
        self.negative_words = [
            "bad", "terrible", "awful", "horrible", "poor", "disappointing",
            "frustrating", "annoying", "irritating", "unpleasant", "dreadful",
            "appalling", "atrocious", "inadequate", "inferior", "unsatisfactory",
            "abysmal", "lousy", "mediocre", "miserable", "pathetic", "sad",
            "unhappy", "angry", "upset", "negative", "worst", "hate", "dislike"
        ]
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze the input text and extract intents, entities, and sentiment.
        
        Args:
            text: The input text to analyze
            
        Returns:
            A dictionary containing the analysis results
        """
        # Convert to lowercase for better matching
        text_lower = text.lower()
        
        # Extract intents
        intents = self._extract_intents(text_lower)
        
        # Extract entities
        entities = self._extract_entities(text)
        
        # Analyze sentiment
        sentiment = self._analyze_sentiment(text_lower)
        
        # Analyze complexity
        complexity = self._analyze_complexity(text)
        
        return {
            "intents": intents,
            "entities": entities,
            "sentiment": sentiment,
            "complexity": complexity,
            "original_text": text
        }
    
    def _extract_intents(self, text: str) -> List[str]:
        """Extract intents from the text."""
        found_intents = []
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    found_intents.append(intent)
                    break  # Found this intent, move to next one
        
        return found_intents
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from the text."""
        entities = {}
        
        for entity_type, patterns in self.entity_patterns.items():
            entities[entity_type] = []
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    entities[entity_type].extend(matches)
        
        # Remove empty entity types
        return {k: v for k, v in entities.items() if v}
    
    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze the sentiment of the text."""
        words = re.findall(r'\b\w+\b', text.lower())
        
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        
        # Calculate sentiment score (-1 to 1)
        total = positive_count + negative_count
        if total == 0:
            score = 0  # Neutral
        else:
            score = (positive_count - negative_count) / total
        
        # Determine sentiment label
        if score > 0.25:
            label = "positive"
        elif score < -0.25:
            label = "negative"
        else:
            label = "neutral"
        
        return {
            "score": score,
            "label": label,
            "positive_words": positive_count,
            "negative_words": negative_count
        }
    
    def _analyze_complexity(self, text: str) -> Dict[str, Any]:
        """Analyze the complexity of the text."""
        # Count words
        words = re.findall(r'\b\w+\b', text.lower())
        word_count = len(words)
        
        # Count sentences
        sentences = re.split(r'[.!?]+', text)
        sentence_count = sum(1 for s in sentences if s.strip())
        
        # Calculate average word length
        if word_count > 0:
            avg_word_length = sum(len(word) for word in words) / word_count
        else:
            avg_word_length = 0
        
        # Calculate average sentence length
        if sentence_count > 0:
            avg_sentence_length = word_count / sentence_count
        else:
            avg_sentence_length = 0
        
        # Determine complexity level
        if avg_sentence_length > 15 or avg_word_length > 6:
            complexity_level = "high"
        elif avg_sentence_length > 10 or avg_word_length > 5:
            complexity_level = "medium"
        else:
            complexity_level = "low"
        
        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_word_length": avg_word_length,
            "avg_sentence_length": avg_sentence_length,
            "complexity_level": complexity_level
        }
    
    def generate_response_template(self, analysis: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a response template based on the analysis.
        
        Args:
            analysis: The analysis results from analyze_text
            
        Returns:
            A tuple containing the response template and parameters
        """
        intents = analysis.get("intents", [])
        sentiment = analysis.get("sentiment", {}).get("label", "neutral")
        
        # Default response for when no specific intent is detected
        template = "I understand you're saying something about {topic}, but I'm not sure how to respond specifically."
        params = {"topic": "that"}
        
        # Check for specific intents
        if "greeting" in intents:
            if sentiment == "positive":
                template = "Hello there! It's wonderful to see you. How can I help you today?"
            elif sentiment == "negative":
                template = "Hello. I notice you might be feeling down. Is there something I can do to help?"
            else:
                template = "Hello! How can I assist you today?"
            params = {}
        
        elif "farewell" in intents:
            if sentiment == "positive":
                template = "Goodbye! It was a pleasure chatting with you. Have a wonderful day!"
            elif sentiment == "negative":
                template = "Goodbye. I hope things improve for you soon."
            else:
                template = "Goodbye! Feel free to chat again whenever you need assistance."
            params = {}
        
        elif "gratitude" in intents:
            template = "You're welcome! I'm always happy to help. Is there anything else you'd like to know?"
            params = {}
        
        elif "weather" in intents:
            locations = analysis.get("entities", {}).get("location", [])
            if locations:
                template = "I'll check the weather in {location} for you."
                params = {"location": locations[0]}
            else:
                template = "I'd be happy to check the weather for you. Could you specify which location you're interested in?"
                params = {}
        
        elif "time" in intents:
            template = "I'll tell you the current time."
            params = {}
        
        elif "help" in intents:
            template = "I'm here to help! I can provide information, answer questions, or assist with various tasks. What specifically do you need help with?"
            params = {}
        
        elif "about" in intents:
            template = "I'm an advanced autonomous AI agent designed to assist with a wide range of tasks. I can understand natural language, learn from interactions, and provide helpful responses."
            params = {}
        
        elif "capabilities" in intents:
            template = "I can help with many things including answering questions, providing information about weather and time, telling jokes, sharing facts, and learning from our interactions to improve over time."
            params = {}
        
        elif "joke" in intents:
            template = "I'll tell you a joke to brighten your day!"
            params = {}
        
        elif "fact" in intents:
            template = "Here's an interesting fact for you!"
            params = {}
        
        elif "learn" in intents:
            template = "I'm always learning from our interactions. Is there something specific you'd like me to learn about?"
            params = {}
        
        elif "reset" in intents:
            template = "Would you like me to reset my learning and start fresh? This will clear all my learned patterns."
            params = {}
        
        return template, params


if __name__ == "__main__":
    # Example usage
    nlp = NLPEngine()
    
    # Test with some example inputs
    test_inputs = [
        "Hello there! How are you today?",
        "What's the weather like in New York?",
        "Tell me a joke please",
        "I'm feeling really sad today",
        "Thanks for your help, you're amazing!",
        "Goodbye, talk to you later"
    ]
    
    for input_text in test_inputs:
        print(f"\nAnalyzing: '{input_text}'")
        analysis = nlp.analyze_text(input_text)
        print(f"Intents: {analysis['intents']}")
        print(f"Entities: {analysis['entities']}")
        print(f"Sentiment: {analysis['sentiment']['label']} (score: {analysis['sentiment']['score']:.2f})")
        print(f"Complexity: {analysis['complexity']['complexity_level']}")
        
        template, params = nlp.generate_response_template(analysis)
        print(f"Response template: '{template}'")
        print(f"Response params: {params}")