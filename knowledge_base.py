import json
import os
import logging
import time
import random
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("KnowledgeBase")

class KnowledgeBase:
    """Knowledge base for storing and retrieving information."""
    
    def __init__(self, storage_path: str = "knowledge_base.json"):
        self.storage_path = storage_path
        self.categories = {
            "general": {},
            "user_preferences": {},
            "facts": {},
            "definitions": {},
            "procedures": {},
            "entities": {}
        }
        self.load()
        
    def load(self) -> None:
        """Load knowledge base from disk."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    self.categories = data
                logger.info(f"Loaded knowledge base from {self.storage_path}")
            except Exception as e:
                logger.error(f"Error loading knowledge base: {e}")
        else:
            logger.info(f"No knowledge base file found at {self.storage_path}, starting with empty knowledge base")
            self._initialize_default_knowledge()
            
    def save(self) -> None:
        """Save knowledge base to disk."""
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(self.categories, f, indent=2)
            logger.info(f"Saved knowledge base to {self.storage_path}")
        except Exception as e:
            logger.error(f"Error saving knowledge base: {e}")
            
    def _initialize_default_knowledge(self) -> None:
        """Initialize the knowledge base with default information."""
        # General knowledge
        self.add_knowledge("general", "agent_purpose", "I am an autonomous AI agent designed to assist users with various tasks.")
        
        # Facts
        facts = [
            "The Earth is the third planet from the Sun.",
            "Water freezes at 0 degrees Celsius at standard atmospheric pressure.",
            "The human body has 206 bones.",
            "The speed of light in a vacuum is approximately 299,792,458 meters per second.",
            "The Great Wall of China is visible from space.",
            "Honey never spoils if stored properly.",
            "The shortest war in history was between Britain and Zanzibar on August 27, 1896, lasting only 38 minutes.",
            "The average person will spend six months of their life waiting for red lights to turn green.",
            "A day on Venus is longer than a year on Venus.",
            "Octopuses have three hearts, nine brains, and blue blood."
        ]
        for i, fact in enumerate(facts):
            self.add_knowledge("facts", f"fact_{i+1}", fact)
            
        # Definitions
        definitions = {
            "artificial intelligence": "The simulation of human intelligence processes by machines, especially computer systems.",
            "machine learning": "A subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed.",
            "neural network": "A computing system inspired by the biological neural networks that constitute animal brains.",
            "algorithm": "A step-by-step procedure for solving a problem or accomplishing a task.",
            "data science": "An interdisciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge and insights from structured and unstructured data."
        }
        for term, definition in definitions.items():
            self.add_knowledge("definitions", term, definition)
            
        # Procedures
        procedures = {
            "make_coffee": [
                "Boil water",
                "Add coffee grounds to a filter",
                "Pour hot water over the grounds",
                "Wait for the coffee to brew",
                "Enjoy your coffee"
            ],
            "change_password": [
                "Go to the account settings",
                "Find the security or password section",
                "Enter your current password",
                "Enter your new password",
                "Confirm your new password",
                "Save the changes"
            ]
        }
        for proc_name, steps in procedures.items():
            self.add_knowledge("procedures", proc_name, steps)
            
        # Save the default knowledge
        self.save()
            
    def add_knowledge(self, category: str, key: str, value: Any) -> bool:
        """
        Add or update knowledge in the specified category.
        
        Args:
            category: The category to add the knowledge to
            key: The key for the knowledge
            value: The value of the knowledge
            
        Returns:
            True if successful, False otherwise
        """
        if category not in self.categories:
            logger.warning(f"Unknown category: {category}")
            return False
            
        self.categories[category][key] = {
            "value": value,
            "timestamp": time.time(),
            "confidence": 1.0  # Default confidence
        }
        return True
        
    def get_knowledge(self, category: str, key: str) -> Optional[Any]:
        """
        Retrieve knowledge from the specified category.
        
        Args:
            category: The category to retrieve from
            key: The key for the knowledge
            
        Returns:
            The value if found, None otherwise
        """
        if category not in self.categories:
            logger.warning(f"Unknown category: {category}")
            return None
            
        if key not in self.categories[category]:
            return None
            
        return self.categories[category][key]["value"]
        
    def search_knowledge(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for knowledge across all categories.
        
        Args:
            query: The search query
            
        Returns:
            A list of matching knowledge items
        """
        results = []
        query = query.lower()
        
        for category, items in self.categories.items():
            for key, data in items.items():
                value = data["value"]
                
                # Check if query is in key
                if query in key.lower():
                    results.append({
                        "category": category,
                        "key": key,
                        "value": value,
                        "confidence": data.get("confidence", 1.0),
                        "timestamp": data.get("timestamp", 0)
                    })
                    continue
                    
                # Check if query is in value
                if isinstance(value, str) and query in value.lower():
                    results.append({
                        "category": category,
                        "key": key,
                        "value": value,
                        "confidence": data.get("confidence", 1.0),
                        "timestamp": data.get("timestamp", 0)
                    })
                    continue
                    
                # Check if value is a list and query is in any item
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, str) and query in item.lower():
                            results.append({
                                "category": category,
                                "key": key,
                                "value": value,
                                "confidence": data.get("confidence", 1.0),
                                "timestamp": data.get("timestamp", 0)
                            })
                            break
                            
                # Check if value is a dict and query is in any value
                if isinstance(value, dict):
                    for k, v in value.items():
                        if (isinstance(k, str) and query in k.lower()) or \
                           (isinstance(v, str) and query in v.lower()):
                            results.append({
                                "category": category,
                                "key": key,
                                "value": value,
                                "confidence": data.get("confidence", 1.0),
                                "timestamp": data.get("timestamp", 0)
                            })
                            break
        
        # Sort results by confidence (descending)
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results
        
    def learn_from_text(self, text: str, category: str = "general") -> Tuple[bool, str]:
        """
        Extract knowledge from text and add it to the knowledge base.
        
        Args:
            text: The text to learn from
            category: The category to add the knowledge to
            
        Returns:
            A tuple of (success, message)
        """
        # This is a simplified implementation
        # In a real system, this would use more sophisticated NLP techniques
        
        # Simple pattern: "X is Y" or "X are Y"
        import re
        
        # Try to extract definitions
        definition_patterns = [
            r"([A-Za-z\s]+) is ([^\.]+)\.",
            r"([A-Za-z\s]+) are ([^\.]+)\.",
            r"([A-Za-z\s]+) refers to ([^\.]+)\.",
            r"([A-Za-z\s]+) means ([^\.]+)\."
        ]
        
        for pattern in definition_patterns:
            matches = re.findall(pattern, text)
            if matches:
                for match in matches:
                    subject = match[0].strip().lower()
                    definition = match[1].strip()
                    
                    if len(subject) > 2 and len(definition) > 5:  # Minimum lengths to avoid noise
                        if category == "general":
                            # If it looks like a definition, put it in definitions category
                            self.add_knowledge("definitions", subject, definition)
                        else:
                            self.add_knowledge(category, subject, definition)
                        
                        return True, f"Learned that {subject} is {definition}"
        
        # If no patterns match, store as general knowledge with a generated key
        key = f"learned_{int(time.time())}"
        self.add_knowledge(category, key, text)
        
        return True, f"Stored as general knowledge"
        
    def get_random_fact(self) -> str:
        """Get a random fact from the knowledge base."""
        facts = self.categories.get("facts", {})
        if not facts:
            return "I don't have any facts stored yet."
            
        fact_key = random.choice(list(facts.keys()))
        return facts[fact_key]["value"]
        
    def get_definition(self, term: str) -> Optional[str]:
        """Get the definition of a term."""
        definitions = self.categories.get("definitions", {})
        
        # Try exact match
        if term in definitions:
            return definitions[term]["value"]
            
        # Try case-insensitive match
        term_lower = term.lower()
        for key, data in definitions.items():
            if key.lower() == term_lower:
                return data["value"]
                
        # Try partial match
        for key, data in definitions.items():
            if term_lower in key.lower() or key.lower() in term_lower:
                return data["value"]
                
        return None
        
    def get_procedure(self, name: str) -> Optional[List[str]]:
        """Get a procedure by name."""
        procedures = self.categories.get("procedures", {})
        
        # Try exact match
        if name in procedures:
            return procedures[name]["value"]
            
        # Try case-insensitive match
        name_lower = name.lower()
        for key, data in procedures.items():
            if key.lower() == name_lower:
                return data["value"]
                
        # Try partial match
        for key, data in procedures.items():
            if name_lower in key.lower() or key.lower() in name_lower:
                return data["value"]
                
        return None
        
    def add_user_preference(self, user_id: str, preference_key: str, preference_value: Any) -> None:
        """Add or update a user preference."""
        if "user_preferences" not in self.categories:
            self.categories["user_preferences"] = {}
            
        if user_id not in self.categories["user_preferences"]:
            self.categories["user_preferences"][user_id] = {}
            
        self.categories["user_preferences"][user_id][preference_key] = {
            "value": preference_value,
            "timestamp": time.time(),
            "confidence": 1.0
        }
        
    def get_user_preference(self, user_id: str, preference_key: str) -> Optional[Any]:
        """Get a user preference."""
        if "user_preferences" not in self.categories:
            return None
            
        if user_id not in self.categories["user_preferences"]:
            return None
            
        if preference_key not in self.categories["user_preferences"][user_id]:
            return None
            
        return self.categories["user_preferences"][user_id][preference_key]["value"]
        
    def get_all_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get all preferences for a user."""
        if "user_preferences" not in self.categories:
            return {}
            
        if user_id not in self.categories["user_preferences"]:
            return {}
            
        return {k: v["value"] for k, v in self.categories["user_preferences"][user_id].items()}


if __name__ == "__main__":
    # Example usage
    kb = KnowledgeBase()
    
    # Add some knowledge
    kb.add_knowledge("general", "greeting", "Hello, how can I help you?")
    kb.add_knowledge("facts", "earth_circumference", "The Earth's circumference is approximately 40,075 kilometers.")
    
    # Retrieve knowledge
    print(kb.get_knowledge("general", "greeting"))
    print(kb.get_knowledge("facts", "earth_circumference"))
    
    # Search for knowledge
    results = kb.search_knowledge("earth")
    for result in results:
        print(f"Found in {result['category']}: {result['value']}")
    
    # Learn from text
    success, message = kb.learn_from_text("Artificial intelligence is the simulation of human intelligence by machines.")
    print(message)
    
    # Get a definition
    print(kb.get_definition("artificial intelligence"))